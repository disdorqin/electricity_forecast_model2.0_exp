"""
Period-aware dynamic weight router.

For every (task, period) bucket the router fits a *constrained* weight
vector that — given the latest per-model predictions and a few
auxiliary signals (e.g. ``spike_prob``, ``hour``) — is the best convex
combination to minimise :func:`smape_floor50`.

Pipeline
========
1. Pivot the long table to a wide frame with one column per model.
2. Optionally merge in extra signals (spike probability, hour, etc.).
3. For every (task, period):
   a. Build a feature matrix ``X`` of model predictions + extras.
   b. Try a small grid of Ridge alphas; for each alpha do a
      ``TimeSeriesSplit`` cross-validation. Each fold's weights are
      the clipped/projected Ridge coefficients, normalised so that
      ``sum(w) == 1`` and ``lower_bound <= w_i <= upper_bound``.
      The CV score is the resulting SMAPE on the held-out fold.
   c. Pick the alpha with the lowest mean CV SMAPE.
   d. Refit on all data with the chosen alpha, project the
      coefficients into valid weights and store the result.
4. Return a weights frame (one row per model per (task, period)) plus
   a report frame (CV scores, chosen alpha, weight sums, …).

The output of :func:`fit_dynamic_router_from_long_table` is fully
compatible with :func:`_apply_fixed_weights` (the projection step
already enforces sum-to-one) so the downstream fuse stage does not
have to be aware of which learner produced the weights.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from .contracts import build_wide_frame
from .metrics import smape_floor50
from .weights import _project_with_bounds


logger = logging.getLogger(__name__)


BASE_ID_COLS = ["task", "target_day", "ds", "period", "hour_business", "y_true"]


@dataclass
class DynamicRouterConfig:
    lower_bound: float = -0.5
    upper_bound: float = 1.2
    alpha_grid: tuple[float, ...] = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0)
    cv_folds: int = 3
    realtime_extra_cols: tuple[str, ...] = ("spike_prob", "hour")
    ridge_floor_smape: float = 0.0
    """If the best CV SMAPE from the Ridge is below the best single
    model's SMAPE minus this margin, the router emits the projected
    Ridge weights; otherwise it emits the simple "1/N" baseline and
    the consumer is expected to fall back to a best-single-model
    gate (mirrors ``use_learner`` semantics in the v3 meta-learner)."""


def _initial_weights(preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Inverse-MAE weights, then projected into the feasible region.

    Used both as a CV prior for the Ridge fold and as the fallback
    when the Ridge under-performs the best single model.
    """
    errors = np.mean(np.abs(preds - y_true[:, None]), axis=0)
    scores = 1.0 / np.maximum(errors, 1e-6)
    weights = scores / scores.sum()
    return _project_with_bounds(weights, -0.5, 1.2)


def _find_best_single_model(clean_group: pd.DataFrame, model_cols: list[str]) -> tuple[str, float]:
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


def _select_extras(
    wide: pd.DataFrame,
    *,
    task: str,
    candidates: tuple[str, ...],
) -> list[str]:
    """Return the subset of `candidates` actually present in the wide frame.

    For the realtime task we also require at least 50% non-null
    coverage of the column — otherwise the extra feature would be
    uninformative and could destabilise the Ridge fit.
    """
    extras: list[str] = []
    for col in candidates:
        if col not in wide.columns:
            continue
        coverage = float(wide[col].notna().mean())
        if coverage < 0.5:
            continue
        extras.append(col)
    return extras


def _build_segment_X(
    group: pd.DataFrame,
    model_cols: list[str],
    extra_cols: list[str],
) -> tuple[np.ndarray, list[str]]:
    cols = list(model_cols) + list(extra_cols)
    X = group[cols].to_numpy(dtype=float)
    return X, cols


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """SMAPE that ignores rows where either side is NaN.

    The router runs on validation predictions which may have missing
    model entries for a small number of hours (e.g. RT916 only covers
    realtime hours). Skipping these rows keeps the CV score
    comparable across folds and across alphas.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("inf")
    return smape_floor50(y_true[mask], y_pred[mask])


def _fit_ridge_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    alpha: float,
    n_models: int,
    lower_bound: float,
    upper_bound: float,
) -> np.ndarray:
    """Fit a Ridge, take the first n_models coefficients, project to feasible region."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train)
    est = Ridge(alpha=alpha, random_state=42)
    est.fit(X_imp, y_train)
    coefs = est.coef_[:n_models]
    return _project_with_bounds(coefs, lower_bound, upper_bound)


def _cv_alpha_search(
    X: np.ndarray,
    y: np.ndarray,
    n_models: int,
    *,
    alpha_grid: tuple[float, ...],
    cv_folds: int,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, list[float], list[np.ndarray], list[float]]:
    """For each alpha, run a time-series CV and return mean SMAPE per fold.

    Returns the alpha with the lowest mean SMAPE, the per-alpha mean
    SMAPE list (aligned with ``alpha_grid``), the per-fold weight
    matrices so that callers can inspect stability, and the per-fold
    held-out SMAPE values for the best alpha (so the caller can reuse
    the *true* cross-validated score instead of recomputing it).
    """
    n = len(X)
    best_alpha = float(alpha_grid[0])
    best_score = float("inf")
    alpha_scores: list[float] = []
    fold_weights: dict[float, list[np.ndarray]] = {a: [] for a in alpha_grid}
    fold_scores: dict[float, list[float]] = {a: [] for a in alpha_grid}

    if n < max(2, cv_folds * 2):
        # Not enough data for CV: use the first alpha.
        return (
            best_alpha,
            [float("inf")] * len(alpha_grid),
            [],
            [],
        )

    tscv = TimeSeriesSplit(n_splits=cv_folds)
    for alpha in alpha_grid:
        per_fold = []
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_vl = X[train_idx], X[val_idx]
            y_tr, y_vl = y[train_idx], y[val_idx]
            weights = _fit_ridge_weights(
                X_tr, y_tr,
                alpha=alpha, n_models=n_models,
                lower_bound=lower_bound, upper_bound=upper_bound,
            )
            preds_vl = X_vl[:, :n_models] @ weights
            per_fold.append(_safe_smape(y_vl, preds_vl))
            fold_weights[alpha].append(weights)
            fold_scores[alpha].append(per_fold[-1])
        mean_score = float(np.mean(per_fold))
        alpha_scores.append(mean_score)
        if mean_score < best_score:
            best_score = mean_score
            best_alpha = float(alpha)

    fold_weights_best = fold_weights[best_alpha]
    fold_scores_best = fold_scores[best_alpha]
    return best_alpha, alpha_scores, fold_weights_best, fold_scores_best


def fit_dynamic_router_from_long_table(
    df: pd.DataFrame,
    *,
    spike_df: pd.DataFrame | None = None,
    config: DynamicRouterConfig | None = None,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    alpha_grid: tuple[float, ...] | None = None,
    cv_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one constrained Ridge per (task, period).

    The function is fully backward compatible: ``lower_bound``,
    ``upper_bound``, ``alpha_grid`` and ``cv_folds`` keyword arguments
    override the corresponding field in ``config`` (the latter is kept
    around for callers that want to set everything in one place).
    """
    cfg = config or DynamicRouterConfig()
    if lower_bound is not None:
        cfg.lower_bound = float(lower_bound)
    if upper_bound is not None:
        cfg.upper_bound = float(upper_bound)
    if alpha_grid is not None:
        cfg.alpha_grid = tuple(alpha_grid)
    if cv_folds is not None:
        cfg.cv_folds = int(cv_folds)

    work = df.copy()
    if spike_df is not None and not spike_df.empty:
        spike = spike_df.copy()
        if "ds" in spike.columns:
            spike["ds"] = pd.to_datetime(spike["ds"], errors="coerce")
        if "ds" in work.columns:
            work["ds"] = pd.to_datetime(work["ds"], errors="coerce")
        # Avoid double-merging if caller already attached extras.
        if "spike_prob" not in work.columns:
            join_keys = ["ds"] if "ds" in spike.columns and "ds" in work.columns else None
            if join_keys is not None:
                work = work.merge(spike, on=join_keys, how="left")
            else:
                logger.warning("spike_df has no `ds` column; skipping merge.")

    wide = build_wide_frame(work)
    # `build_wide_frame` may produce a duplicate `y_true` column
    # when the input is not strictly long-format. Drop duplicates.
    if wide.columns.duplicated().any():
        wide = wide.loc[:, ~wide.columns.duplicated()].copy()
    # Re-attach any extras the caller may have included alongside
    # the long table. `build_wide_frame` pivots on the prediction
    # column, so non-prediction columns are dropped — we re-merge
    # them here keyed by `ds` so the per-(task, period) regression
    # can use them as extra features.
    extra_cols_to_reattach: list[str] = []
    if spike_df is not None and not spike_df.empty:
        for col in ("spike_prob", "is_spike"):
            if col in spike_df.columns:
                if col not in wide.columns:
                    spike_keyed = spike_df[["ds", col]].drop_duplicates(subset="ds")
                    spike_keyed["ds"] = pd.to_datetime(spike_keyed["ds"], errors="coerce")
                    spike_keyed = spike_keyed.rename(columns={col: f"__extra_{col}"})
                    wide = wide.merge(spike_keyed, on="ds", how="left")
                    wide = wide.rename(columns={f"__extra_{col}": col})
                # Whether we just added it or it was already there,
                # make sure it is excluded from the model column list
                # and counted as an extra feature.
                if col not in extra_cols_to_reattach:
                    extra_cols_to_reattach.append(col)
    # `hour` can always be recovered from `ds` deterministically.
    if "ds" in wide.columns:
        if "hour" not in wide.columns:
            wide["hour"] = pd.to_datetime(wide["ds"]).dt.hour.replace({0: 24}).astype(int)
        if "hour" not in extra_cols_to_reattach:
            extra_cols_to_reattach.append("hour")

    base_model_cols = [
        c for c in wide.columns
        if c not in BASE_ID_COLS and c not in extra_cols_to_reattach
    ]
    if not base_model_cols:
        raise ValueError("No model columns found after pivoting prediction table")

    weights_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []

    for (task, period), group in wide.groupby(["task", "period"], sort=True):
        active_model_cols = [
            c for c in base_model_cols
            if c in group.columns and group[c].notna().any()
        ]
        clean_group = group.dropna(subset=["y_true"]).copy()
        if not active_model_cols or clean_group.empty:
            continue

        # ── Decide which extras to use for this (task, period) ──
        if str(task) == "realtime":
            extras = _select_extras(clean_group, task="realtime", candidates=cfg.realtime_extra_cols)
        else:
            extras = []

        n_models = len(active_model_cols)
        X, fnames = _build_segment_X(clean_group, active_model_cols, extras)
        y = clean_group["y_true"].to_numpy(dtype=float)

        # ── Cross-validate over alpha_grid ──
        best_alpha, alpha_scores, fold_weights, fold_scores_best = _cv_alpha_search(
            X, y, n_models,
            alpha_grid=cfg.alpha_grid,
            cv_folds=cfg.cv_folds,
            lower_bound=cfg.lower_bound,
            upper_bound=cfg.upper_bound,
        )

        # ── Final fit on all data with the best alpha ──
        final_weights = _fit_ridge_weights(
            X, y, alpha=best_alpha, n_models=n_models,
            lower_bound=cfg.lower_bound, upper_bound=cfg.upper_bound,
        )
        ensemble_pred = X[:, :n_models] @ final_weights
        fit_smape = _safe_smape(y, ensemble_pred)

        # ── Compare against best single model on a fair basis ──
        # cv_pred: mean held-out SMAPE from the per-fold Ridge fits.
        # best_cv_smape: lowest mean held-out SMAPE across the per-model
        # time-series CVs (same fold structure, single-model predictions
        # on the validation fold). This is the "use_router" gate.
        best_name, best_smape = _find_best_single_model(clean_group, active_model_cols)
        if fold_scores_best:
            cv_pred = float(np.mean(fold_scores_best))
            best_cv_smapes: list[float] = []
            tscv = TimeSeriesSplit(n_splits=cfg.cv_folds)
            for col in active_model_cols:
                preds_col = clean_group[col].to_numpy(dtype=float)
                per_fold: list[float] = []
                for _, val_idx in tscv.split(X):
                    y_v = y[val_idx]
                    p_v = preds_col[val_idx]
                    valid = ~(np.isnan(y_v) | np.isnan(p_v))
                    if valid.sum() == 0:
                        continue
                    per_fold.append(smape_floor50(y_v[valid], p_v[valid]))
                if per_fold:
                    best_cv_smapes.append(float(np.mean(per_fold)))
            best_cv_smape = min(best_cv_smapes) if best_cv_smapes else float("inf")
        else:
            cv_pred = float("inf")
            best_cv_smape = float("inf")
        use_router = (cv_pred + cfg.ridge_floor_smape) < best_cv_smape

        if not use_router:
            # Emit best-single gate: all weight on best_name, rest 0.
            gated = np.zeros(n_models, dtype=float)
            if best_name in active_model_cols:
                gated[active_model_cols.index(best_name)] = 1.0
            final_weights = _project_with_bounds(gated, cfg.lower_bound, cfg.upper_bound)

        for model_name, w in zip(active_model_cols, final_weights):
            weights_rows.append(
                {
                    "task": task,
                    "period": period,
                    "model_name": model_name,
                    "weight": float(w),
                    "sample_count": int(len(clean_group)),
                    "weight_lower_bound": float(cfg.lower_bound),
                    "weight_upper_bound": float(cfg.upper_bound),
                    "use_router": bool(use_router),
                    "best_alpha": float(best_alpha),
                    "extra_features": ",".join(extras),
                }
            )

        row: dict[str, object] = {
            "task": task,
            "period": period,
            "sample_count": int(len(clean_group)),
            "smape_fit": fit_smape,
            "smape_cv_ridge": float(cv_pred),
            "smape_cv_best_single": best_cv_smape,
            "best_single_model": best_name,
            "best_single_smape": float(best_smape),
            "use_router": bool(use_router),
            "best_alpha": float(best_alpha),
            "weight_sum": float(final_weights.sum()),
            "extra_features": ",".join(extras),
        }
        for a, s in zip(cfg.alpha_grid, alpha_scores):
            row[f"smape_cv_alpha_{a}"] = float(s)
        for model_name, w in zip(active_model_cols, final_weights):
            row[f"weight_{model_name}"] = float(w)
        report_rows.append(row)

    weights_df = pd.DataFrame(weights_rows)
    report_df = pd.DataFrame(report_rows)
    if not weights_df.empty:
        weights_df = weights_df.sort_values(["task", "period", "model_name"]).reset_index(drop=True)
    if not report_df.empty:
        report_df = report_df.sort_values(["task", "period"]).reset_index(drop=True)
    return weights_df, report_df


def apply_dynamic_router(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    task: str,
    test_start: str,
    test_end: str,
    renormalize_missing: bool = True,
) -> pd.DataFrame:
    """Apply the dynamic-router weights to a wide-format forecast frame.

    This is a thin wrapper around the original ``_apply_fixed_weights``
    renormalization rule: if a particular model has a NaN prediction
    for some hours, the remaining active models' weights are rescaled
    to sum to one. The wrapper lives next to the router so that the
    fuse stage has a single, well-documented entry point.
    """
    task_df = df[df["task"] == task].copy() if "task" in df.columns else df.copy()
    if "target_day" in task_df.columns:
        task_days = pd.to_datetime(task_df["target_day"])
        task_df = task_df[
            (task_days >= pd.Timestamp(test_start)) & (task_days <= pd.Timestamp(test_end))
        ].copy()
    if task_df.empty:
        raise RuntimeError(f"No rows found for task={task} in [{test_start}, {test_end}].")

    non_feature_cols = {
        "task", "target_day", "ds", "period", "hour_business",
        "y_true", "y_pred", "y_fused",
    }
    feature_cols = [c for c in task_df.columns if c not in non_feature_cols]
    if not feature_cols:
        raise RuntimeError("No model prediction columns available for dynamic router.")

    task_weights = weights_df[weights_df["task"] == task].copy()
    fused_values: list[float] = []
    renorm_count = 0
    for _, row in task_df.iterrows():
        period_weights = task_weights[task_weights["period"] == row["period"]]
        value = 0.0
        used_weight_sum = 0.0
        for _, weight_row in period_weights.iterrows():
            model_name = weight_row["model_name"]
            if model_name not in task_df.columns or pd.isna(row[model_name]):
                continue
            w = float(weight_row["weight"])
            value += w * float(row[model_name])
            used_weight_sum += w
        if renormalize_missing and used_weight_sum > 1e-9 and abs(used_weight_sum - 1.0) > 0.01:
            value /= used_weight_sum
            renorm_count += 1
        fused_values.append(value)
    if renorm_count:
        logger.info(
            "apply_dynamic_router renormalized %d/%d rows (some models missing) in task=%s",
            renorm_count, len(task_df), task,
        )
    task_df = task_df.copy()
    task_df["y_fused"] = fused_values
    return task_df.sort_values(["target_day", "hour_business"]).reset_index(drop=True)
