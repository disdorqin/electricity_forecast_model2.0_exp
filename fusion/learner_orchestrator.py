"""Window 2 (学习器 + 动态路由) — 统一学习器编排器。

设计目标
--------
1. 把 ``dynamic_weights`` (SLSQP) / ``dynamic_router`` (Ridge 系数投影) /
   ``meta_learner_v3`` (Ridge with aux features) 三个学习器封装成同一套接口:
   - ``fit_*_learner(long_df, **kwargs)`` -> ``LearnerOutputs``
   - ``apply_learner_outputs(long_df, outputs, **kwargs)`` -> long_df + y_fused
2. 在 9_16 区间，把 dynamic_weights 的 base weight 与 dynamic_router 学到的
   spike-aware 偏置混合：SLSQP 给出 base (neutral) 权重 + Ridge 在
   spike 维度上的修正。
3. 输出标准化的 ``LearnerOutputs`` dataclass，包含:
   - ``weights``            : ``{(task, period): {model: weight, ...}}``
   - ``cv_scores``          : ``{(task, period): float}``
   - ``per_fold_weights``   : ``{(task, period): list[np.ndarray]}``
   - ``spike_templates``    : dynamic_weights 9_16 spike-aware 模板
   - ``meta_learner``       : ``dict[(task, period)] -> SegmentMetaModel``
   - ``router_weights``     : ``{(task, period): {model: weight, ...}}``
     来自 dynamic_router，可选 9_16 spike-aware 偏置
4. 包含 ``fit_learner_stage`` 函数 — 这是 W2 替代 staged_pipeline 中
   ``_run_learner_stage`` 的统一入口。会按顺序跑 dynamic_weights →
   dynamic_router → meta_learner_v3，输出统一 ``LearnerOutputs``。

约束
----
- 不修改 ``dynamic_weights.py`` / ``dynamic_router.py`` / ``meta_learner_v3.py``
  的核心算法，仅做"组合 + 包装"。
- 不触碰 spike_detector 内部。
- 9_16 混合: spike_aware_bias = (router_weight - dyn_weight) * alpha
  其中 alpha 来自 spike_prob (HIGH>0.6 → alpha→1, LOW<0.2 → alpha→-1)。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .dynamic_weights import (
    DynamicWeightResult,
    apply_dynamic_weights,
    fit_dynamic_weights,
    HIGH_SPIKE_THRESHOLD,
    LOW_SPIKE_THRESHOLD,
)
from .dynamic_router import (
    DynamicRouterConfig,
    apply_dynamic_router,
    fit_dynamic_router_from_long_table,
)
from .meta_learner_v3 import (
    SegmentMetaModel,
    apply_meta_learners,
    fit_meta_learners_from_long_table,
)

logger = logging.getLogger(__name__)


# ── Standardised outputs ──

@dataclass
class LearnerOutputs:
    """所有学习器的统一输出格式。

    字段
    ----
    weights : ``{(task, period): {model_name: weight, ...}}``
        时段感知的基础权重 (来自 dynamic_weights / SLSQP)
    cv_scores : ``{(task, period): float}``
        每个时段的 CV SMAPE 估计（来自 dynamic_weights 训练集）
    per_fold_weights : ``{(task, period): list[np.ndarray]}``
        每个时段每折的 SLSQP 权重 (用于稳定性诊断)
    spike_templates : ``{(task, period): {"neutral"|"sgdfnet_heavy"|"rt916_heavy": {...}}}``
        9_16 时段内 spike-aware 模板权重
    router_weights : ``{(task, period): {model_name: weight, ...}}``
        来自 dynamic_router 的 Ridge 系数投影权重 (可与 weights 混合)
    router_cv_scores : ``{(task, period): float}``
        router 在每段的 CV SMAPE
    router_use : ``{(task, period): bool}``
        router 的 use_router 门控结果
    meta_learner : ``dict[(task, period)] -> SegmentMetaModel``
        来自 meta_learner_v3 的逐段 Ridge 模型
    meta_report : ``pd.DataFrame``
        meta_learner_v3 的 report frame
    dynamic_weights_raw : ``DynamicWeightResult``
        原 dynamic_weights 完整结果 (含 affine calibration)
    dynamic_router_raw : ``tuple[pd.DataFrame, pd.DataFrame]``
        原 dynamic_router 完整结果 (weights_df, report_df)
    """

    weights: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    cv_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    per_fold_weights: dict[tuple[str, str], list[np.ndarray]] = field(default_factory=dict)
    spike_templates: dict[tuple[str, str], dict[str, dict[str, float]]] = field(default_factory=dict)
    router_weights: dict[tuple[str, str], dict[str, float]] = field(default_factory=dict)
    router_cv_scores: dict[tuple[str, str], float] = field(default_factory=dict)
    router_use: dict[tuple[str, str], bool] = field(default_factory=dict)
    meta_learner: dict[tuple[str, str], SegmentMetaModel] = field(default_factory=dict)
    meta_report: pd.DataFrame = field(default_factory=pd.DataFrame)
    dynamic_weights_raw: DynamicWeightResult | None = None
    dynamic_router_raw: tuple[pd.DataFrame, pd.DataFrame] | None = None


# ── Mixed-weight helpers (9_16 ridge-bias injection) ──

def _interp_alpha(spike_prob: float) -> float:
    """把 spike_prob 映射到 (-1, 1) 偏置 alpha。

    - spike_prob >= 0.6 → alpha -> 1 (倾向 router)
    - spike_prob <= 0.2 → alpha -> -1 (倾向 base weights)
    - 中间 → 0 (用 base weights)
    """
    if pd.isna(spike_prob):
        return 0.0
    if spike_prob >= HIGH_SPIKE_THRESHOLD:
        # 在 [0.6, 1.0] 之间从 0 线性上升到 1
        return float(min(1.0, (spike_prob - HIGH_SPIKE_THRESHOLD) / max(1.0 - HIGH_SPIKE_THRESHOLD, 1e-9)))
    if spike_prob <= LOW_SPIKE_THRESHOLD:
        # 在 [0.0, 0.2] 之间从 0 线性下降到 -1
        return float(-(1.0 - spike_prob / max(LOW_SPIKE_THRESHOLD, 1e-9)))
    return 0.0


def _normalise_weights(weights: dict[str, float]) -> dict[str, float]:
    """Project weights into the [-0.5, 1.2] bounds and renormalise to sum=1.

    Mirrors the normalisation used by dynamic_weights so the result can
    be fed straight to apply_dynamic_weights style logic.
    """
    models = list(weights.keys())
    if not models:
        return weights
    clipped = {m: float(np.clip(weights.get(m, 0.0), -0.5, 1.2)) for m in models}
    total = sum(clipped.values())
    if abs(total - 1.0) < 1e-9:
        return clipped
    if abs(total) < 1e-9:
        return {m: 1.0 / len(models) for m in models}
    scaled = {m: clipped[m] / total for m in models}
    for m in models:
        scaled[m] = float(np.clip(scaled[m], -0.5, 1.2))
    s = sum(scaled.values())
    if abs(s) < 1e-9:
        return {m: 1.0 / len(models) for m in models}
    return {m: v / s for m, v in scaled.items()}


def mix_base_with_router_bias(
    base_weights: dict[str, float],
    router_weights: dict[str, float],
    alpha: float,
) -> dict[str, float]:
    """把 router_weights 投影成 base_weights 的偏置，按 alpha 混合。

    math::
        w_mixed = base + alpha * (router - base)
        normalize to sum=1
    """
    models = list(base_weights.keys())
    if not models:
        return base_weights
    mixed = {m: float(base_weights.get(m, 0.0)) for m in models}
    for m in models:
        mixed[m] = mixed[m] + alpha * (router_weights.get(m, 0.0) - mixed[m])
    return _normalise_weights(mixed)


# ── Top-level orchestration ──

def fit_learner_stage(
    long_df: pd.DataFrame,
    *,
    fit_dynamic: bool = True,
    fit_router: bool = True,
    fit_meta: bool = True,
    router_config: DynamicRouterConfig | None = None,
    meta_kwargs: Mapping | None = None,
    dynamic_kwargs: Mapping | None = None,
) -> LearnerOutputs:
    """统一的 learner_stage: 在同一份 long_df 上同时训练 dynamic_weights,
    dynamic_router, meta_learner_v3, 合并到标准化的 ``LearnerOutputs``。

    任何一步失败都不会让整个 stage 失败 — 出错的子学习器会保留空
    结果，剩下的学习器照常工作 (保证 fuse_stage 至少有一个可用
    后备)。
    """
    out = LearnerOutputs()
    if long_df is None or long_df.empty:
        logger.warning("fit_learner_stage: empty long_df, returning empty outputs")
        return out

    # ── 1) dynamic_weights (SLSQP, spike-aware 9_16) ──
    if fit_dynamic:
        try:
            dyn = fit_dynamic_weights(long_df, **(dict(dynamic_kwargs) if dynamic_kwargs else {}))
            out.dynamic_weights_raw = dyn
            out.weights = {k: dict(v) for k, v in dyn.weights.items()}
            out.spike_templates = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in dyn.spike_interpolation.items()}
            for _, row in dyn.report.iterrows():
                out.cv_scores[(str(row["task"]), str(row["period"]))] = float(row.get("smape", float("nan")))
            logger.info(
                "fit_learner_stage: dynamic_weights fit on %d segments",
                len(out.weights),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fit_learner_stage: dynamic_weights failed: %s", exc)

    # ── 2) dynamic_router (Ridge 系数投影) ──
    if fit_router:
        try:
            cfg = router_config or DynamicRouterConfig()
            weights_df, report_df = fit_dynamic_router_from_long_table(long_df, config=cfg)
            out.dynamic_router_raw = (weights_df, report_df)
            if not weights_df.empty:
                for (task, period), grp in weights_df.groupby(["task", "period"]):
                    out.router_weights[(str(task), str(period))] = {
                        str(row["model_name"]): float(row["weight"]) for _, row in grp.iterrows()
                    }
            if not report_df.empty:
                for _, row in report_df.iterrows():
                    key = (str(row["task"]), str(row["period"]))
                    out.router_cv_scores[key] = float(row.get("smape_cv_ridge", float("nan")))
                    out.router_use[key] = bool(row.get("use_router", False))
            logger.info(
                "fit_learner_stage: dynamic_router fit on %d segments",
                len(out.router_weights),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fit_learner_stage: dynamic_router failed: %s", exc)

    # ── 3) meta_learner_v3 (Ridge + aux features + sub-916 split) ──
    if fit_meta:
        try:
            meta_models, meta_report = fit_meta_learners_from_long_table(
                long_df, **(dict(meta_kwargs) if meta_kwargs else {}),
            )
            out.meta_learner = meta_models
            out.meta_report = meta_report
            use_count = sum(1 for m in meta_models.values() if m.use_learner)
            logger.info(
                "fit_learner_stage: meta_learner_v3 fit on %d segments, use_learner=True for %d",
                len(meta_models), use_count,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fit_learner_stage: meta_learner_v3 failed: %s", exc)

    return out


# ── Top-level apply ──

def apply_learner_outputs(
    long_df: pd.DataFrame,
    outputs: LearnerOutputs,
    *,
    mode: str = "dynamic",
    spike_col: str = "spike_prob",
) -> pd.DataFrame:
    """把 ``LearnerOutputs`` 应用到 forecast long table，输出含 y_fused。

    mode
    ----
    - ``"dynamic"`` : 用 dynamic_weights (含 9_16 spike-aware 模板)
    - ``"router"``  : 用 dynamic_router 投影权重
    - ``"meta"``    : 用 meta_learner_v3 (Ridge)
    - ``"mixed"``   : 9_16 区间用 dynamic base + router bias 混合，
                     其他时段退到 dynamic base (默认行为)
    """
    if long_df is None or long_df.empty:
        return long_df
    if mode == "dynamic":
        if not outputs.weights:
            raise RuntimeError("apply_learner_outputs: mode=dynamic but no weights in outputs")
        return apply_dynamic_weights(
            long_df,
            outputs.weights,
            spike_templates=outputs.spike_templates or None,
            spike_col=spike_col,
        )
    if mode == "router":
        if not outputs.router_weights:
            raise RuntimeError("apply_learner_outputs: mode=router but no router_weights in outputs")
        weights_rows: list[dict[str, object]] = []
        for (task, period), wmap in outputs.router_weights.items():
            for model_name, w in wmap.items():
                weights_rows.append({
                    "task": task, "period": period, "model_name": model_name,
                    "weight": float(w),
                })
        weights_df = pd.DataFrame(weights_rows)
        # apply_dynamic_router 需要 task/test_start/test_end；这里从 long_df 推
        if "task" in long_df.columns:
            tasks = long_df["task"].dropna().unique().tolist()
            task = str(tasks[0]) if tasks else "realtime"
        else:
            task = "realtime"
        if "target_day" in long_df.columns:
            days = pd.to_datetime(long_df["target_day"], errors="coerce").dropna()
            if not days.empty:
                test_start = days.min().strftime("%Y-%m-%d")
                test_end = days.max().strftime("%Y-%m-%d")
            else:
                test_start = "1900-01-01"
                test_end = "2999-12-31"
        else:
            test_start = "1900-01-01"
            test_end = "2999-12-31"
        return apply_dynamic_router(
            long_df, weights_df,
            task=task, test_start=test_start, test_end=test_end,
        )
    if mode == "meta":
        if not outputs.meta_learner:
            raise RuntimeError("apply_learner_outputs: mode=meta but no meta_learner in outputs")
        if "task" in long_df.columns:
            tasks = long_df["task"].dropna().unique().tolist()
            task = str(tasks[0]) if tasks else "realtime"
        else:
            task = "realtime"
        if "target_day" in long_df.columns:
            days = pd.to_datetime(long_df["target_day"], errors="coerce").dropna()
            if not days.empty:
                test_start = days.min().strftime("%Y-%m-%d")
                test_end = days.max().strftime("%Y-%m-%d")
            else:
                test_start = "1900-01-01"
                test_end = "2999-12-31"
        else:
            test_start = "1900-01-01"
            test_end = "2999-12-31"
        return apply_meta_learners(
            long_df, outputs.meta_learner,
            task=task, test_start=test_start, test_end=test_end,
        )
    if mode == "mixed":
        # 9_16 区间：base (dynamic_weights) + router bias 混合
        # 其他时段：纯 dynamic_weights
        if not outputs.weights:
            raise RuntimeError("apply_learner_outputs: mode=mixed but no weights in outputs")
        if not outputs.router_weights:
            # 没有 router 权重 → 退到 dynamic
            return apply_dynamic_weights(
                long_df, outputs.weights,
                spike_templates=outputs.spike_templates or None,
                spike_col=spike_col,
            )
        # 先按 dynamic 跑一遍
        fused = apply_dynamic_weights(
            long_df, outputs.weights,
            spike_templates=outputs.spike_templates or None,
            spike_col=spike_col,
        )
        # 对 9_16 行用 router bias 重新算
        if "period" not in fused.columns:
            return fused
        mask_916 = fused["period"].astype(str) == "9_16"
        if not mask_916.any():
            return fused
        sp_series = fused[spike_col] if spike_col in fused.columns else pd.Series([np.nan] * len(fused), index=fused.index)
        router_default = outputs.router_weights.get(("realtime", "9_16")) or outputs.router_weights.get(("dayahead", "9_16"))
        if router_default is None:
            return fused
        # 把 (task, period) 分组
        for (task, period), grp in fused[mask_916].groupby(["task", "period"], sort=True):
            key = (str(task), str(period))
            base_w = outputs.weights.get(key)
            router_w = outputs.router_weights.get(key) or router_default
            if base_w is None or router_w is None:
                continue
            for idx in grp.index:
                sp = float(sp_series.loc[idx]) if pd.notna(sp_series.loc[idx]) else float("nan")
                alpha = _interp_alpha(sp)
                w_mixed = mix_base_with_router_bias(base_w, router_w, alpha)
                # 重新计算 y_fused
                row = fused.loc[idx]
                value = 0.0
                used = 0.0
                for model_name, w in w_mixed.items():
                    if model_name in fused.columns and pd.notna(row.get(model_name)):
                        value += float(w) * float(row[model_name])
                        used += float(w)
                if used > 1e-9 and abs(used - 1.0) > 0.01:
                    value /= used
                fused.at[idx, "y_fused"] = value
        return fused
    raise ValueError(f"apply_learner_outputs: unknown mode={mode!r}")


def evaluate_learner_outputs(
    long_df: pd.DataFrame,
    outputs: LearnerOutputs,
    *,
    modes: tuple[str, ...] = ("dynamic", "router", "meta", "mixed"),
    spike_col: str = "spike_prob",
) -> dict[str, dict[str, float]]:
    """对每个 mode 跑 apply_learner_outputs 并按 period 计算 SMAPE。

    返回 ``{mode: {smape_overall, smape_1_8, smape_9_16, smape_17_24, rows}}``
    """
    from .metrics import smape_floor50

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        try:
            fused = apply_learner_outputs(long_df, outputs, mode=mode, spike_col=spike_col)
        except Exception as exc:  # noqa: BLE001
            logger.warning("evaluate_learner_outputs: mode=%s failed: %s", mode, exc)
            results[mode] = {"error": 1.0}
            continue
        valid = fused.dropna(subset=["y_true", "y_fused"]) if "y_true" in fused.columns else fused.dropna(subset=["y_fused"])
        if valid.empty or "y_true" not in valid.columns:
            results[mode] = {"rows": int(len(fused))}
            continue
        out: dict[str, float] = {
            "rows": int(len(valid)),
            "smape_overall": smape_floor50(valid["y_true"].to_numpy(float), valid["y_fused"].to_numpy(float)),
        }
        if "period" in valid.columns:
            for p in ("1_8", "9_16", "17_24"):
                sub = valid[valid["period"].astype(str) == p]
                if not sub.empty:
                    out[f"smape_{p}"] = smape_floor50(
                        sub["y_true"].to_numpy(float), sub["y_fused"].to_numpy(float),
                    )
        results[mode] = out
    return results
