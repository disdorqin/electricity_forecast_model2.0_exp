"""
Time-period aware dynamic weight fitter for fusion ensemble.

Window1 (尖峰融合深度化) — 让 spike_detector 的输出真正影响融合权重。

设计要点
--------
1. 一键把一天的 24 小时按业务时段分（1_8 / 9_16 / 17_24），对每个时段
   用 ``fusion.weights.fit_segment_weights`` 跑约束 SLSQP，学一个
   权重向量（权重和=1, 范围 [-0.5, 1.2]）。
2. 9_16 时段内，可以根据 ``spike_prob`` 进一步在"标准融合权重"和
   "保守/激进权重"之间插值：
   - spike_prob > 0.6 → 倾向于 sgdfnet（被证明对极端值更鲁棒）
   - spike_prob < 0.2 → 倾向于 rt916（无尖峰时使用 spike-detector 自身）
3. 接口：
   - ``fit_dynamic_weights(long_df, ...)`` -> ``{(task, period): {model_name: float, ...}, ...}``
   - ``apply_dynamic_weights(long_df, weights_dict, spike_col='spike_prob')`` -> long_df + y_fused
4. ``long_df`` 是 ``fusion.contracts`` 规范表，至少包含
   ``task / target_day / ds / period / hour_business / y_true / y_pred / model_name``；
   9_16 时段融合需要 ``spike_prob`` 列（可以缺失，缺失时回退到标准权重）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import build_wide_frame, infer_period
from .metrics import smape_floor50
from .weights import fit_segment_weights

logger = logging.getLogger(__name__)

VALID_PERIODS = ("1_8", "9_16", "17_24")

# 9_16 时段内 spike_prob 调控的拐点
HIGH_SPIKE_THRESHOLD = 0.6     # 超过此值 → sgdfnet 主导
LOW_SPIKE_THRESHOLD = 0.2      # 低于此值 → rt916 主导
TRANSITION_WIDTH = 0.2         # 软过渡带宽

# 9_16 尖峰 "保守/激进" 模板（每个模型给一个基准权重）
# 通过把对应模型权重抬高、其他模型等比例压制实现
SGDFNET_HEAVY = {
    "sgdfnet": 1.5,    # 对极端值更鲁棒
    "rt916": 0.0,      # 尖峰时段不参与
    "timesfm": 0.2,
    "timemixer": 0.0,
    "lightgbm": 0.0,
}
RT916_HEAVY = {
    "rt916": 1.3,      # 平稳时段让 spike-detector 自身主导
    "sgdfnet": 0.2,
    "timesfm": 0.2,
    "timemixer": 0.0,
    "lightgbm": 0.0,
}


@dataclass
class DynamicWeightResult:
    weights: dict[tuple[str, str], dict[str, float]]
    report: pd.DataFrame
    spike_interpolation: dict[tuple[str, str], dict[str, dict[str, float]]]
    affine_calibration: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    affine_report: pd.DataFrame = field(default_factory=pd.DataFrame)


def _affine_search(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    scale_grid: tuple[float, ...] = (0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20),
    bias_grid: tuple[float, ...] = (-30.0, -15.0, -5.0, 0.0, 5.0, 15.0, 30.0),
) -> tuple[float, float, float]:
    """Grid-search (scale, bias) minimising SMAPE on (y_true, y_pred).

    Returns ``(scale, bias, smape)``.  Used to absorb the systematic
    bias that linear fusion cannot cancel (e.g. SGDFNet over-predicting
    on weekday peak hours).
    """
    best = (1.0, 0.0, float("inf"))
    for s in scale_grid:
        for b in bias_grid:
            y_hat = s * y_pred + b
            smp = smape_floor50(y_true, y_hat)
            if smp < best[2]:
                best = (s, b, smp)
    return best


def fit_affine_calibration(
    long_df: pd.DataFrame,
    fused_col: str = "y_fused",
    truth_col: str = "y_true",
) -> dict[tuple[str, str], tuple[float, float]]:
    """Per-(task, period) affine calibration that minimises SMAPE on val."""
    work = long_df.copy()
    if fused_col not in work.columns:
        raise ValueError(f"fused_col '{fused_col}' missing — call apply_dynamic_weights first")
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for (task, period), group in work.groupby(["task", "period"], sort=True):
        valid = group.dropna(subset=[truth_col, fused_col])
        if len(valid) < 4:
            continue
        s, b, _ = _affine_search(
            valid[truth_col].to_numpy(dtype=float),
            valid[fused_col].to_numpy(dtype=float),
        )
        out[(str(task), str(period))] = (s, b)
    return out


def _normalize_to_sum_one(weights: Mapping[str, float], models: list[str]) -> dict[str, float]:
    """对缺失模型补 0 后做最小-最大投影, 强制 sum=1。"""
    bounded_lower, bounded_upper = -0.5, 1.2
    out = {m: float(weights.get(m, 0.0)) for m in models}
    # clip
    for m in models:
        out[m] = float(np.clip(out[m], bounded_lower, bounded_upper))
    total = sum(out.values())
    if abs(total - 1.0) < 1e-9:
        return out
    # 若和不为 1，按比例缩放并 clip（极端情况下 clip 后再调整一次）
    scaled = {m: out[m] / total for m in models} if abs(total) > 1e-9 else {m: 1.0 / len(models) for m in models}
    for m in models:
        scaled[m] = float(np.clip(scaled[m], bounded_lower, bounded_upper))
    s = sum(scaled.values())
    if abs(s) < 1e-9:
        scaled = {m: 1.0 / len(models) for m in models}
    else:
        scaled = {m: v / s for m, v in scaled.items()}
    return scaled


def _extract_models(wide: pd.DataFrame, exclude: tuple[str, ...] = ("spike_prob", "is_spike")) -> list[str]:
    excluded = set(BASE_ID_COLS) | set(exclude)
    return [c for c in wide.columns if c not in excluded]


BASE_ID_COLS = ["task", "target_day", "ds", "period", "hour_business", "y_true"]


def _build_template_weights(period: str, models: list[str], base_weights: Mapping[str, float]) -> dict[str, float]:
    """构造 "9_16 尖峰模板权重": 只在当前实际存在的模型中分配。"""
    template = {m: 0.0 for m in models}
    for m in models:
        template[m] = float(base_weights.get(m, 0.0))
    if all(abs(v) < 1e-9 for v in template.values()):
        # 若模板一个权重都没匹配上（极端小模型集），退回到均分
        return {m: 1.0 / len(models) for m in models}
    return _normalize_to_sum_one(template, models)


def _interpolation_factor(spike_prob: float) -> tuple[float, float, str]:
    """根据 spike_prob 返回 (alpha_sgdfnet, alpha_rt916, 描述)。

    - spike_prob >= 0.6      → (1.0, 0.0, "sgdfnet-heavy")
    - 0.2 < spike_prob < 0.6 → (0.0, 0.0, "neutral")   走基线权重
    - spike_prob <= 0.2      → (0.0, 1.0, "rt916-heavy")
    """
    if spike_prob >= HIGH_SPIKE_THRESHOLD:
        # 在 [0.6, 1.0] 之间从 0 线性上升到 1（更尖峰越倾向 sgdfnet）
        alpha = min(1.0, (spike_prob - HIGH_SPIKE_THRESHOLD) / max(1.0 - HIGH_SPIKE_THRESHOLD, 1e-9))
        return float(alpha), 0.0, "sgdfnet-heavy"
    if spike_prob <= LOW_SPIKE_THRESHOLD:
        # 在 [0.0, 0.2] 之间从 1 线性下降到 0
        alpha = max(0.0, 1.0 - spike_prob / max(LOW_SPIKE_THRESHOLD, 1e-9))
        return 0.0, float(alpha), "rt916-heavy"
    return 0.0, 0.0, "neutral"


def _interpolate(base: dict[str, float], template_a: dict[str, float], template_b: dict[str, float],
                 alpha_a: float, alpha_b: float) -> dict[str, float]:
    """base + alpha_a*(template_a - base) + alpha_b*(template_b - base)，再 normalize。"""
    models = list(base.keys())
    out = {m: float(base[m]) for m in models}
    for m in models:
        out[m] = float(base[m]) + alpha_a * (template_a.get(m, 0.0) - float(base[m]))
        out[m] = out[m] + alpha_b * (template_b.get(m, 0.0) - float(base[m]))
    return _normalize_to_sum_one(out, models)


def fit_dynamic_weights(
    long_df: pd.DataFrame,
    *,
    reg: float = 0.1,
    reg_map: dict[str, float] | None = None,
    lower_bound: float = -0.5,
    upper_bound: float = 1.2,
) -> DynamicWeightResult:
    """训练时段感知动态权重。

    Parameters
    ----------
    long_df : ``fusion.contracts`` 规范表（含 y_true, y_pred, model_name, period 等）
    reg : 默认正则系数，传给 ``fit_segment_weights``
    reg_map : 按 period 自定义正则系数
    lower_bound / upper_bound : 权重边界

    Returns
    -------
    ``DynamicWeightResult`` 对象：
    - ``weights`` : ``{(task, period): {model_name: weight, ...}}``
    - ``report``  : 每段训练的 SMAPE / 样本数 / 主导模型
    - ``spike_interpolation`` : ``{(task, period): {"neutral": {...}, "sgdfnet_heavy": {...}, "rt916_heavy": {...}}}``
    """
    wide = build_wide_frame(long_df)
    models = _extract_models(wide)
    if not models:
        raise ValueError("dynamic_weights: no model columns found after pivoting long table")

    weights_dict: dict[tuple[str, str], dict[str, float]] = {}
    spike_dict: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    rows: list[dict[str, object]] = []

    for (task, period), group in wide.groupby(["task", "period"], sort=True):
        segment_models = [m for m in models if m in group.columns and group[m].notna().any()]
        if not segment_models:
            continue
        clean = group.dropna(subset=["y_true"]).copy()
        clean = clean.dropna(subset=segment_models, how="all")
        if clean.empty or len(clean) < 4:
            continue

        # Filter to rows where all segment_models are observed (per-row, not per-column),
        # otherwise SLSQP sees NaN and returns NaN weights.
        all_observed = clean[segment_models + ["y_true"]].notna().all(axis=1)
        clean_full = clean[all_observed].copy()
        if len(clean_full) < 4:
            logger.debug(
                "dynamic_weights: %s/%s has too few fully-observed rows (%d), skipping SLSQP",
                task, period, len(clean_full),
            )
            continue

        preds = clean_full[segment_models].to_numpy(dtype=float)
        y_true = clean_full["y_true"].to_numpy(dtype=float)

        seg_reg = float(reg_map.get(period, reg)) if reg_map else float(reg)
        try:
            w_vec, smape_val, iters = fit_segment_weights(
                preds,
                y_true,
                reg=seg_reg,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("fit_segment_weights failed for %s/%s: %s", task, period, exc)
            continue

        if np.isnan(w_vec).any():
            logger.warning(
                "dynamic_weights: SLSQP returned NaN for %s/%s (n=%d), falling back to inverse-MAE",
                task, period, len(clean_full),
            )
            # Fallback: 简单 1/MAE
            errors = np.mean(np.abs(preds - y_true[:, None]), axis=0)
            scores = 1.0 / np.maximum(errors, 1e-6)
            w_vec = scores / scores.sum()

        w_map = {m: float(w) for m, w in zip(segment_models, w_vec)}
        weights_dict[(str(task), str(period))] = w_map
        rows.append(
            {
                "task": str(task),
                "period": str(period),
                "sample_count": int(len(clean_full)),
                "smape": float(smape_val) if not np.isnan(smape_val) else float("nan"),
                "iterations": int(iters),
                "models": ",".join(segment_models),
                "reg": float(seg_reg),
            }
        )

        if str(period) == "9_16":
            spike_dict[(str(task), "9_16")] = {
                "neutral": w_map,
                "sgdfnet_heavy": _build_template_weights("9_16", segment_models, SGDFNET_HEAVY),
                "rt916_heavy": _build_template_weights("9_16", segment_models, RT916_HEAVY),
            }

    # ── Affine calibration per (task, period) on the val fused series ──
    affine_dict: dict[tuple[str, str], tuple[float, float]] = {}
    affine_rows: list[dict[str, object]] = []
    if weights_dict:
        try:
            val_fused = apply_dynamic_weights(
                long_df,
                weights_dict,
                spike_templates=spike_dict,
            )
            affine_dict = fit_affine_calibration(val_fused, fused_col="y_fused", truth_col="y_true")
            for (task, period), (s, b) in affine_dict.items():
                sub = val_fused[(val_fused["task"] == task) & (val_fused["period"] == period)].dropna(
                    subset=["y_true", "y_fused"]
                )
                if len(sub) >= 4:
                    raw_smape = smape_floor50(sub["y_true"].to_numpy(float), sub["y_fused"].to_numpy(float))
                    cal_smape = smape_floor50(
                        sub["y_true"].to_numpy(float),
                        (s * sub["y_fused"].to_numpy(float) + b).astype(float),
                    )
                    affine_rows.append(
                        {
                            "task": task,
                            "period": period,
                            "scale": float(s),
                            "bias": float(b),
                            "raw_smape": float(raw_smape),
                            "calibrated_smape": float(cal_smape),
                            "improvement_pct": float(raw_smape - cal_smape),
                            "n": int(len(sub)),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("affine calibration failed: %s", exc)

    return DynamicWeightResult(
        weights=weights_dict,
        report=pd.DataFrame(rows),
        spike_interpolation=spike_dict,
        affine_calibration=affine_dict,
        affine_report=pd.DataFrame(affine_rows),
    )


def apply_dynamic_weights(
    long_df: pd.DataFrame,
    weights_dict: Mapping[tuple[str, str], Mapping[str, float]],
    *,
    spike_templates: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]] | None = None,
    spike_col: str = "spike_prob",
    high_threshold: float = HIGH_SPIKE_THRESHOLD,
    low_threshold: float = LOW_SPIKE_THRESHOLD,
    affine_calibration: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    affine_fused_col: str = "y_fused_calibrated",
) -> pd.DataFrame:
    """对每行根据 period（+ 9_16 内的 spike_prob）应用动态权重，输出 y_fused 列。"""
    work = long_df.copy()
    if "period" not in work.columns:
        work["hour_business"] = pd.to_datetime(work["ds"]).dt.hour.replace({0: 24}).astype(int)
        work["period"] = work["hour_business"].map(infer_period)

    # 全部 long 表 -> wide（每个时间点一行，每个模型一列）
    pivot_index = ["task", "target_day", "ds", "period", "hour_business"]
    if "y_true" in work.columns:
        pivot_index = pivot_index + ["y_true"]
    if spike_col in work.columns:
        pivot_index = pivot_index + [spike_col]

    truth = work[pivot_index].drop_duplicates(subset=["task", "target_day", "ds", "period", "hour_business"])
    wide_pred = work.pivot_table(
        index=["task", "target_day", "ds", "period", "hour_business"],
        columns="model_name",
        values="y_pred",
        aggfunc="last",
    ).reset_index()
    wide_pred.columns.name = None
    fused = truth.merge(wide_pred, on=["task", "target_day", "ds", "period", "hour_business"], how="left")

    fused_values: list[float] = []
    used_regime: list[str] = []
    spike_used: list[float] = []

    spike_series = fused[spike_col] if spike_col in fused.columns else pd.Series([0.5] * len(fused))

    for _, row in fused.iterrows():
        key = (str(row["task"]), str(row["period"]))
        w_map = weights_dict.get(key)
        if not w_map:
            # 找不到该时段的权重 — 退到均值
            avail = [c for c in wide_pred.columns if c not in {"task", "target_day", "ds", "period", "hour_business"}]
            vals = [row[c] for c in avail if c in row.index and pd.notna(row[c])]
            fused_values.append(float(np.mean(vals)) if vals else float("nan"))
            used_regime.append("fallback_mean")
            spike_used.append(float(row[spike_col]) if spike_col in fused.columns and pd.notna(row.get(spike_col)) else float("nan"))
            continue

        regime = "neutral"
        weights_to_use = dict(w_map)
        if key[1] == "9_16" and spike_templates is not None and key in spike_templates:
            templates = spike_templates[key]
            sp = float(row[spike_col]) if spike_col in fused.columns and pd.notna(row.get(spike_col)) else 0.5
            alpha_sg, alpha_rt, regime = _interpolation_factor(sp)
            if alpha_sg > 0 or alpha_rt > 0:
                weights_to_use = _interpolate(
                    w_map,
                    templates.get("sgdfnet_heavy", w_map),
                    templates.get("rt916_heavy", w_map),
                    alpha_sg,
                    alpha_rt,
                )
            spike_used.append(sp)
        else:
            spike_used.append(float(row[spike_col]) if spike_col in fused.columns and pd.notna(row.get(spike_col)) else float("nan"))

        value = 0.0
        used_weight_sum = 0.0
        for model_name, w in weights_to_use.items():
            if model_name not in fused.columns or pd.isna(row.get(model_name)):
                continue
            value += float(w) * float(row[model_name])
            used_weight_sum += float(w)
        if used_weight_sum > 1e-9 and abs(used_weight_sum - 1.0) > 0.01:
            value /= used_weight_sum
        fused_values.append(value)
        used_regime.append(regime)

    fused["y_fused"] = fused_values
    fused["spike_regime"] = used_regime
    if spike_col in fused.columns:
        fused["spike_prob_used"] = spike_used

    if affine_calibration:
        cal_values: list[float] = []
        for _, row in fused.iterrows():
            sb = affine_calibration.get((str(row["task"]), str(row["period"])))
            if sb is None or pd.isna(row.get("y_fused")):
                cal_values.append(float(row["y_fused"]) if pd.notna(row.get("y_fused")) else float("nan"))
                continue
            cal_values.append(float(sb[0]) * float(row["y_fused"]) + float(sb[1]))
        fused[affine_fused_col] = cal_values
        # Make the calibrated output the canonical y_fused for downstream consumers.
        fused["y_fused"] = fused[affine_fused_col]
        fused["y_fused_raw"] = pd.NA

    return fused.sort_values(["target_day", "hour_business"]).reset_index(drop=True)


def evaluate_dynamic_weights(
    long_df: pd.DataFrame,
    weights_dict: Mapping[tuple[str, str], Mapping[str, float]],
    spike_templates: Mapping[tuple[str, str], Mapping[str, Mapping[str, float]]] | None = None,
    spike_col: str = "spike_prob",
) -> dict[str, float]:
    """对带真值的 long_df 跑 apply_dynamic_weights 并汇报 SMAPE。"""
    fused = apply_dynamic_weights(
        long_df,
        weights_dict,
        spike_templates=spike_templates,
        spike_col=spike_col,
    )
    valid = fused.dropna(subset=["y_true", "y_fused"])
    if valid.empty:
        return {"smape_overall": float("nan"), "rows": 0}

    out: dict[str, float] = {
        "rows": int(len(valid)),
        "smape_overall": smape_floor50(valid["y_true"].to_numpy(float), valid["y_fused"].to_numpy(float)),
    }
    if "period" in valid.columns:
        for period in VALID_PERIODS:
            sub = valid[valid["period"] == period]
            if not sub.empty:
                out[f"smape_{period}"] = smape_floor50(sub["y_true"].to_numpy(float), sub["y_fused"].to_numpy(float))
    if "spike_regime" in valid.columns:
        for regime in ("neutral", "sgdfnet_heavy", "rt916_heavy", "fallback_mean"):
            sub = valid[valid["spike_regime"] == regime]
            if not sub.empty:
                out[f"smape_regime_{regime}"] = smape_floor50(sub["y_true"].to_numpy(float), sub["y_fused"].to_numpy(float))
                out[f"count_regime_{regime}"] = int(len(sub))
    return out
