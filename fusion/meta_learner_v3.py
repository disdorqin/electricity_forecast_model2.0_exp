"""
Improved meta-learner: Ridge regression with cross-validation.
Ensures fusion is never worse than the best individual model.

v3.1 — Added optional `extra_feature_cols` parameter to allow feeding
auxiliary signals (spike_prob, hour, is_peak, …) as extra regression
features. The contract long table typically only carries model
predictions, so extra features are merged-in externally by the caller
before invoking this function (see :func:`augment_long_table_with_extras`).
The implementation is fully backward compatible: if the requested
extra columns are absent in the merged frame, the learner silently
falls back to using only the per-model prediction columns.

v3.2 — More auxiliary features (hour_sin/cos, period_onehot,
        is_high_spike/is_low_spike), a relaxed ``use_learner`` gate with
        an ``enable_meta_floor`` fallback (when the learner is within
        ~0.5 pp of the best single model and the pool has >=3 models,
        we trust the learner) and optional 9-16 sub-interval splitting
        (9-12, 13-16) for the trickiest window. Also hardened
        :func:`apply_meta_learners` against NaT / missing ``target_day``
        columns (fallback to ``ds.dt.normalize()``) so the function
        does not raise on forecast-day frames where ``y_true`` is NaN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .contracts import build_wide_frame, infer_period as _infer_period
from .metrics import smape_floor50


BASE_ID_COLS = ["task", "target_day", "ds", "period", "hour_business", "y_true"]
# 尖峰融合（窗口1）: spike_detector 输出的额外特征列，会在 long_df 中按行携带。
SPIKE_FEATURE_COLS = ("spike_prob", "is_spike")

# 9_16 子区间切分（窗口2 增强）：9-12 / 13-16 各自跑一个 ridge，调用时由 hour 决定子区间
SUB_916_BREAK_HOUR = 13
HIGH_SPIKE_FLOOR = 0.6
LOW_SPIKE_FLOOR = 0.2

# 9_16 整体放松门控：cv_smape 差距 < 0.5 个百分点但模型池 >= 3 个时仍允许开启
META_FLOOR_PP = 0.5
META_FLOOR_MIN_MODELS = 3


def _period_subbucket(period: str, hour_business: int) -> str:
    """把 9_16 进一步拆成 9_12 / 13_16 子区间；其他时段保持原样。

    仅在 ``use_sub916_split`` 开启时由 v3 主流程调用。
    """
    if str(period) == "9_16":
        if int(hour_business) < SUB_916_BREAK_HOUR:
            return "9_12"
        return "13_16"
    return str(period)


@dataclass
class SegmentMetaModel:
    task: str
    period: str
    model_names: list[str]
    imputer: SimpleImputer
    estimator: Ridge
    best_single_model: str = ""
    best_single_smape: float = 0.0
    fit_smape: float = 0.0
    cv_smape: float = 0.0
    use_learner: bool = True
    extra_feature_names: list[str] = field(default_factory=list)
    # 9-16 子区间切分：v3.2 新增
    sub_period: str = ""
    # 触发原因：strict / meta_floor / fallback（用于排查）
    enable_reason: str = ""


def _build_features(
    group: pd.DataFrame,
    model_cols: list[str],
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    cols = list(model_cols)
    if extra_cols:
        cols = cols + [c for c in extra_cols if c in group.columns]
    return group[cols].copy()


def _feature_names(model_cols: list[str], extra_cols: list[str] | None) -> list[str]:
    base = list(model_cols)
    if extra_cols:
        base = base + list(extra_cols)
    return base


def _add_periodic_and_spike_features(wide: pd.DataFrame) -> pd.DataFrame:
    """在 wide frame 上补充 hour_sin/hour_cos + period_onehot + is_high/low_spike。

    这些特征由 v3.2 自动注入，使用门槛:
    - hour_sin/cos: 总是注入
    - period_onehot (1_8/9_16/17_24 三个 0/1): 总是注入
    - is_high_spike / is_low_spike: 仅在 spike_prob 存在时注入
    """
    out = wide.copy()
    if "ds" in out.columns and "hour_sin" not in out.columns:
        ds = pd.to_datetime(out["ds"], errors="coerce")
        hour = ds.dt.hour.replace({0: 24}).astype(int)
        rad = 2.0 * np.pi * hour.astype(float) / 24.0
        out["hour_sin"] = np.sin(rad)
        out["hour_cos"] = np.cos(rad)
    for p in ("1_8", "9_16", "17_24"):
        col = f"period_{p}"
        if col not in out.columns and "period" in out.columns:
            out[col] = (out["period"].astype(str) == p).astype(float)
    if "spike_prob" in out.columns:
        sp = pd.to_numeric(out["spike_prob"], errors="coerce")
        if "is_high_spike" not in out.columns:
            out["is_high_spike"] = (sp >= HIGH_SPIKE_FLOOR).astype(float)
        if "is_low_spike" not in out.columns:
            out["is_low_spike"] = (sp <= LOW_SPIKE_FLOOR).astype(float)
    return out


def _find_best_single_model(clean_group: pd.DataFrame, model_cols: list[str]) -> tuple[str, float]:
    """SMAPE of the best single model on the (full) training set.

    The training set is what the Ridge sees during fitting, so this
    is the only fair baseline for :func:`_find_best_single_model`. For
    a held-out comparison, see :func:`_cv_single_model_smapes`.
    """
    y_true = clean_group["y_true"].to_numpy(dtype=float)
    best_name, best_smape = "", float("inf")
    for col in model_cols:
        if col not in clean_group.columns:
            continue
        preds = clean_group[col].to_numpy(dtype=float)
        valid = ~(np.isnan(y_true) | np.isnan(preds))
        if valid.sum() == 0:
            continue
        s = smape_floor50(y_true[valid], preds[valid])
        if s < best_smape:
            best_smape = s
            best_name = col
    return best_name, best_smape


def _cv_single_model_smapes(
    clean_group: pd.DataFrame,
    model_cols: list[str],
    *,
    cv_folds: int,
) -> tuple[str, float]:
    """Cross-validated SMAPE of every single model.

    The same time-series folds used for the Ridge are used here so the
    numbers are directly comparable. Returns the name and CV SMAPE of
    the best single model.
    """
    if len(clean_group) < cv_folds * 2:
        return _find_best_single_model(clean_group, model_cols)
    y_true = clean_group["y_true"].to_numpy(dtype=float)
    tscv = TimeSeriesSplit(n_splits=cv_folds)
    best_name, best_smape = "", float("inf")
    for col in model_cols:
        if col not in clean_group.columns:
            continue
        preds = clean_group[col].to_numpy(dtype=float)
        per_fold = []
        for _, val_idx in tscv.split(clean_group):
            y_v = y_true[val_idx]
            p_v = preds[val_idx]
            valid = ~(np.isnan(y_v) | np.isnan(p_v))
            if valid.sum() == 0:
                continue
            per_fold.append(smape_floor50(y_v[valid], p_v[valid]))
        if not per_fold:
            continue
        s = float(np.mean(per_fold))
        if s < best_smape:
            best_smape = s
            best_name = col
    if not best_name:
        return _find_best_single_model(clean_group, model_cols)
    return best_name, best_smape


def augment_long_table_with_extras(
    long_df: pd.DataFrame,
    extras_df: pd.DataFrame | None,
    *,
    join_keys: tuple[str, ...] = ("ds",),
) -> pd.DataFrame:
    """Augment a contract long table with extra features by joining on `ds`.

    The extras_df is expected to be a frame of auxiliary signals (e.g.
    ``spike_prob``, ``is_spike``, ``is_peak``) keyed by ``ds``. Each row
    in ``long_df`` inherits the corresponding row from ``extras_df``. If
    ``extras_df`` is None or empty, the original long_df is returned
    unchanged (backward compatible).
    """
    if extras_df is None or extras_df.empty:
        return long_df
    extras = extras_df.copy()
    if "ds" in extras.columns:
        extras["ds"] = pd.to_datetime(extras["ds"], errors="coerce")
    # Drop columns that would either collide with id columns / y_true in
    # ``long_df`` or pollute the wide frame (e.g. raw ``时刻`` strings).
    reserved = {
        "task", "model_name", "target_day", "ds", "period",
        "hour_business", "y_true", "y_pred", "时刻",
    }
    keep = [c for c in extras.columns if c not in reserved]
    if not keep:
        return long_df
    extras = extras[list(join_keys) + [c for c in keep if c not in join_keys]]
    out = long_df.copy()
    if "ds" in out.columns:
        out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.merge(extras, on=list(join_keys), how="left")
    return out


def _resolve_target_day(work: pd.DataFrame) -> pd.Series:
    """Compute target_day series from ds (with NaT → '' fallback).

    Returns a Series of strings in ``%Y-%m-%d`` format. Falls back to
    using ``ds.dt.normalize()`` if the column is missing/NaT — that
    matches the multi-day evaluation scripts in the project.
    """
    if "target_day" not in work.columns:
        if "ds" in work.columns:
            return pd.to_datetime(work["ds"], errors="coerce").dt.normalize().dt.strftime("%Y-%m-%d")
        return pd.Series([""] * len(work), index=work.index)
    series = pd.to_datetime(work["target_day"], errors="coerce")
    if series.isna().any() and "ds" in work.columns:
        ds_norm = pd.to_datetime(work["ds"], errors="coerce").dt.normalize()
        series = series.fillna(ds_norm)
    return series.dt.strftime("%Y-%m-%d")


def fit_meta_learners_from_long_table(
    df: pd.DataFrame,
    *,
    alpha: float = 1.0,
    cv_folds: int = 3,
    extra_feature_cols: list[str] | None = None,
    min_use_learner_improvement: float = 0.0,
    use_spike_features: bool = True,
    enable_meta_floor: bool = True,
    use_sub916_split: bool = True,
) -> tuple[dict[tuple[str, str], SegmentMetaModel], pd.DataFrame]:
    """Fit one Ridge per (task, period).

    The frame may contain additional feature columns beyond the model
    predictions. Anything listed in ``extra_feature_cols`` is appended to
    the regression feature matrix; if the columns are missing the
    function silently drops them and the learner degenerates to the
    legacy v3 behaviour (model predictions only).

    When ``use_spike_features=True`` (default), ``spike_prob`` and
    ``is_spike`` columns are auto-included if present in the wide
    frame — this is the integration point with the spike_detector.

    When ``enable_meta_floor=True`` (v3.2) we *also* turn the learner
    on if ``cv_smape - best_cv_smape < META_FLOOR_PP`` *and* the
    candidate pool has ``>= META_FLOOR_MIN_MODELS`` models. This stops
    the strict-gate problem where a 1-pp-loss-on-CV still locks us to
    a single model even though the ridge has access to useful aux
    features (hour_sin/cos, is_high_spike, …).

    When ``use_sub916_split=True`` (v3.2) the 9_16 segment is split
    into 9_12 and 13_16 and each sub-interval gets its own ridge.
    """
    wide = _add_periodic_and_spike_features(build_wide_frame(df))
    base_model_cols = [
        column for column in wide.columns if column not in BASE_ID_COLS
    ]
    # Identify which extra columns are actually present.
    extras_present: list[str] = []
    if extra_feature_cols:
        extras_present = [c for c in extra_feature_cols if c in wide.columns]
    if use_spike_features:
        # 自动注入 spike_detector 的特征；列缺失时静默跳过。
        for c in SPIKE_FEATURE_COLS:
            if c in wide.columns and c not in extras_present:
                extras_present.append(c)
    # 显式追加 v3.2 周期性 / 区间 onehot / spike 阈值特征
    for c in (
        "hour_sin", "hour_cos",
        "period_1_8", "period_9_16", "period_17_24",
        "is_high_spike", "is_low_spike",
    ):
        if c in wide.columns and c not in extras_present:
            extras_present.append(c)

    if not base_model_cols:
        raise ValueError("No model columns found after pivoting prediction table")

    models: dict[tuple[str, str], SegmentMetaModel] = {}
    report_rows: list[dict[str, object]] = []

    # 9-16 子区间切分：先 groupby 一次，把每个 (task, 9_16) 组拆成 9_12 / 13_16
    if use_sub916_split and "hour_business" in wide.columns:
        wide["__sub_period__"] = wide.apply(
            lambda r: _period_subbucket(str(r.get("period", "")), int(r.get("hour_business", 0))),
            axis=1,
        )
    else:
        wide["__sub_period__"] = wide.get("period", "").astype(str)

    for (task, sub_period), group in wide.groupby(["task", "__sub_period__"], sort=True):
        # Empty sub_period (period column not set) — skip
        if str(sub_period) in ("", "nan"):
            continue
        period = sub_period if "_" in str(sub_period) else str(group["period"].iloc[0])
        active_model_cols = [
            column for column in base_model_cols
            if column in group.columns and group[column].notna().any()
        ]
        clean_group = group.dropna(subset=["y_true"]).copy()
        if not active_model_cols or clean_group.empty:
            continue

        best_name, best_smape = _find_best_single_model(clean_group, active_model_cols)

        # Per-segment extras (re-check presence, since some segments may
        # be missing columns that exist elsewhere).
        seg_extras = [c for c in extras_present if c in clean_group.columns]
        feature_frame = _build_features(clean_group, active_model_cols, seg_extras)
        fnames = _feature_names(active_model_cols, seg_extras)
        imputer = SimpleImputer(strategy="median")
        x_all = imputer.fit_transform(feature_frame[fnames])
        y_all = clean_group["y_true"].to_numpy(dtype=float)

        if len(clean_group) >= cv_folds * 2:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_smapes = []
            for train_idx, val_idx in tscv.split(x_all):
                x_tr, x_vl = x_all[train_idx], x_all[val_idx]
                y_tr, y_vl = y_all[train_idx], y_all[val_idx]
                est = Ridge(alpha=alpha, random_state=42)
                est.fit(x_tr, y_tr)
                y_hat = est.predict(x_vl)
                cv_smapes.append(smape_floor50(y_vl, y_hat))
            cv_smape = float(np.mean(cv_smapes))
            # Compare against the best single model's CV SMAPE so the
            # gate is comparing apples to apples.
            best_cv_name, best_cv_smape = _cv_single_model_smapes(
                clean_group, active_model_cols, cv_folds=cv_folds,
            )
            best_smape_cv = best_cv_smape
        else:
            cv_smape = float("inf")
            best_cv_name, best_cv_smape = "", float("inf")
            best_smape_cv = float("inf")

        estimator = Ridge(alpha=alpha, random_state=42)
        estimator.fit(x_all, y_all)
        y_fit = estimator.predict(x_all)
        fit_smape = float(smape_floor50(y_all, y_fit))

        # ── use_learner gate (v3.2: strict + meta_floor fallback) ──
        strict = (cv_smape + min_use_learner_improvement) < best_smape_cv
        enable_reason = "strict"
        if not strict and enable_meta_floor and np.isfinite(cv_smape) and np.isfinite(best_smape_cv):
            gap_pp = cv_smape - best_smape_cv
            if gap_pp < META_FLOOR_PP and len(active_model_cols) >= META_FLOOR_MIN_MODELS:
                strict = True
                enable_reason = "meta_floor"
        use_learner = strict

        segment_model = SegmentMetaModel(
            task=str(task),
            period=str(period),
            model_names=active_model_cols,
            imputer=imputer,
            estimator=estimator,
            best_single_model=best_name,
            best_single_smape=best_smape,
            fit_smape=fit_smape,
            cv_smape=cv_smape if cv_smape != float("inf") else fit_smape,
            use_learner=use_learner,
            extra_feature_names=seg_extras,
            sub_period=str(sub_period),
            enable_reason=enable_reason,
        )
        models[(str(task), str(period))] = segment_model

        coef_map = dict(zip(fnames, estimator.coef_.tolist()))
        report_rows.append(
            {
                "task": task,
                "period": period,
                "sub_period": str(sub_period),
                "sample_count": int(len(clean_group)),
                "cv_smape": cv_smape,
                "fit_smape": fit_smape,
                "best_single_model": best_name,
                "best_single_smape": best_smape,
                "use_learner": use_learner,
                "enable_reason": enable_reason,
                "extra_features": ",".join(seg_extras),
                **{f"coef_{name}": float(coef_map.get(name, 0.0)) for name in fnames},
            }
        )

    return models, pd.DataFrame(report_rows)


def apply_meta_learners(
    df: pd.DataFrame,
    models: dict[tuple[str, str], SegmentMetaModel],
    *,
    task: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """Apply per-(task, period) meta-learners to a forecast frame.

    The frame is expected to be either a long table (``model_name`` +
    ``y_pred``) or a wide frame. The function pivots to wide form,
    resolves the (task, period) bucket, and applies the fitted Ridge
    (or the best single model when the segment gate is off).

    v3.2 hardens:
    - target_day is recomputed from ``ds.dt.normalize()`` if missing/NaT
    - empty frames are returned as-is (no RuntimeError); this is the
      correct behaviour for forecast-day frames where y_true is NaN.
    - 9_16 sub-interval lookup falls back to 9_16 if the 9_12/13_16
      variant was never trained (e.g. too few samples).
    """
    work = df.copy()
    if work.empty:
        return work
    work["ds"] = pd.to_datetime(work["ds"], errors="coerce")

    if "model_name" in work.columns and "y_pred" in work.columns:
        work["hour_business"] = work["ds"].dt.hour.replace({0: 24}).astype(int)
        work["period"] = work["hour_business"].map(_infer_period)
        work["target_day"] = work["ds"].dt.normalize().where(
            work["ds"].dt.hour != 0, work["ds"].dt.normalize() - pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        work["task"] = task

        id_cols = ["task", "target_day", "ds", "period", "hour_business"]
        if "y_true" in work.columns:
            id_cols_with_truth = id_cols + ["y_true"]
        else:
            id_cols_with_truth = id_cols

        truth_df = work[id_cols_with_truth].drop_duplicates(subset=id_cols)
        wide_pred = work.pivot_table(
            index=id_cols, columns="model_name", values="y_pred", aggfunc="last"
        ).reset_index()
        wide_pred.columns.name = None
        work = truth_df.merge(wide_pred, on=id_cols, how="inner")

    # ── v3.2 fallback: derive target_day from ds if missing/NaT ──
    work["target_day"] = _resolve_target_day(work)
    if "period" not in work.columns and "hour_business" in work.columns:
        work["period"] = work["hour_business"].map(_infer_period)
    if "period" not in work.columns and "ds" in work.columns:
        work["period"] = pd.to_datetime(work["ds"]).dt.hour.replace({0: 24}).map(_infer_period)
    if "task" not in work.columns:
        work["task"] = task

    task_days = pd.to_datetime(work["target_day"], errors="coerce")
    if task_days.isna().all():
        # Forecast-day frame with no resolvable target_day — return as-is
        return work
    valid_dates = (task_days >= pd.Timestamp(test_start)) & (task_days <= pd.Timestamp(test_end))
    if not valid_dates.any():
        # No rows in the test window — this is the case when the
        # long table only has training-day rows. Return an empty
        # frame rather than raising (preserves the v3.0 contract for
        # training-only use-cases while still emitting a warning).
        import logging
        logging.getLogger(__name__).warning(
            "apply_meta_learners: no test rows for task=%s in [%s, %s]; returning empty frame",
            task, test_start, test_end,
        )
        return work.iloc[0:0].copy()
    task_df = work[valid_dates].copy()
    if task_df.empty:
        return task_df

    # Inject the same aux features we trained on so the Ridge can
    # consume them (hour_sin/cos / period onehots / spike thresholds).
    task_df = _add_periodic_and_spike_features(task_df)

    non_model_cols = {
        "task", "target_day", "ds", "period", "hour_business",
        "y_true", "y_pred", "y_fused",
    }
    all_feature_cols = [c for c in task_df.columns if c not in non_model_cols]

    fused_parts: list[pd.DataFrame] = []
    for period, group in task_df.groupby("period", sort=True):
        # Lookup key: prefer the 9_12 / 13_16 sub-bucket if available
        hour_b = int(group["hour_business"].iloc[0]) if "hour_business" in group.columns else 0
        sub_key = _period_subbucket(str(period), hour_b)
        key = (task, sub_key)
        meta_model = models.get(key) or models.get((task, str(period)))
        if meta_model is None:
            if all_feature_cols:
                out = group.copy()
                out["y_fused"] = group[all_feature_cols].mean(axis=1)
                fused_parts.append(out)
            continue

        available_models = [m for m in meta_model.model_names if m in group.columns and group[m].notna().any()]
        available_extras = [c for c in meta_model.extra_feature_names if c in group.columns]
        available_features = available_models + available_extras

        if not available_models:
            out = group.copy()
            out["y_fused"] = np.nan
            fused_parts.append(out)
            continue

        best_name = meta_model.best_single_model
        if best_name not in available_models:
            best_name = available_models[0]
        best_preds = group[best_name].to_numpy(dtype=float) if best_name in group.columns else None

        if not meta_model.use_learner:
            y_pred = best_preds if best_preds is not None else np.full(len(group), np.nan)
            out = group.copy()
            out["y_fused"] = y_pred
            fused_parts.append(out)
            continue

        coverage_ok = len(available_models) >= max(1, len(meta_model.model_names) * 0.5)
        if meta_model.cv_smape < meta_model.best_single_smape and coverage_ok and available_features:
            feature_frame = _build_features(group, available_models, available_extras)
            fnames = _feature_names(available_models, available_extras)
            try:
                # Some sklearn versions enforce strict column-name matching on transform;
                # we add any missing extras (NaN-filled) so imputer/transform accept the frame.
                missing_extras = [c for c in meta_model.extra_feature_names if c not in feature_frame.columns]
                if missing_extras:
                    for c in missing_extras:
                        feature_frame[c] = np.nan
                    fnames = _feature_names(available_models, list(meta_model.extra_feature_names))
                x_test = meta_model.imputer.transform(feature_frame[fnames])
                ridge_preds = meta_model.estimator.predict(x_test)
            except Exception:
                # 特征空间与训练时不完全一致 → 退到 best single
                ridge_preds = None
        else:
            ridge_preds = None

        if ridge_preds is not None:
            y_pred = ridge_preds
        elif best_preds is not None:
            y_pred = best_preds
        else:
            y_pred = np.full(len(group), np.nan)

        out = group.copy()
        out["y_fused"] = y_pred
        fused_parts.append(out)

    return pd.concat(fused_parts, ignore_index=True).sort_values(["target_day", "hour_business"]).reset_index(drop=True)
