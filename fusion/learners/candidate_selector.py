"""Candidate selector: compare multiple fusion strategies and pick the best.

Compares:
- equal_weight
- static_convex
- bgew
- each single model (one-hot)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics, smape_floor50, mae
from .static_convex import fit_static_convex
from .bgew import fit_bgew

logger = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    candidate_name: str
    selected_mode: str  # "equal_weight", "static_convex", "bgew", "single_model"
    selected_model: str | None  # For single_model mode
    weights: dict[str, float]
    metric_value: float
    metric_name: str
    full_metrics: dict = field(default_factory=dict)


def _evaluate_candidate(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    weights: dict[str, float],
    *,
    metric_name: str = "sMAPE_floor50",
) -> tuple[float, dict]:
    """Evaluate a candidate weight vector on OOF data.

    Returns
    -------
    metric_value : float
        The optimization metric value.
    full_metrics : dict
        Complete metrics suite including intersection_retention_rate.
    """
    sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)].copy()

    # Track retention
    original_rows = len(sub.drop_duplicates(subset=["target_day", "ds", "hour_business"]))

    # Pivot to wide
    wide = sub.pivot_table(
        index=["target_day", "ds", "hour_business"],
        columns="model_name",
        values="y_pred",
        aggfunc="first",
    )

    available = [m for m in weights.keys() if m in wide.columns and weights[m] > 0]
    if not available:
        return float("inf"), {}

    intersection_rows = len(wide.dropna())
    wide = wide.dropna()

    truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
        ["target_day", "ds", "hour_business", "y_true"]
    ].set_index(["target_day", "ds", "hour_business"])

    wide = wide.join(truth, how="inner")

    if wide.empty:
        return float("inf"), {}

    y_true = wide["y_true"].values.astype(float)
    y_pred = np.zeros(len(y_true))
    for m, w in weights.items():
        if m in wide.columns and w > 0:
            y_pred += w * wide[m].values.astype(float)

    metric_fn = smape_floor50 if metric_name == "sMAPE_floor50" else mae
    metric_value = metric_fn(y_true, y_pred)

    full_metrics = compute_all_metrics(y_true, y_pred)
    full_metrics["intersection_retention_rate"] = (
        intersection_rows / original_rows if original_rows > 0 else 0.0
    )
    full_metrics["original_rows"] = original_rows
    full_metrics["intersection_rows"] = intersection_rows

    return metric_value, full_metrics


def fit_all_candidates(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    eligible_models: list[str],
    *,
    metric_name: str = "sMAPE_floor50",
    tau: float = 30.0,
    eta: float = 0.5,
) -> list[tuple[str, str, str | None, dict[str, float], float]]:
    """Fit all candidate strategies on a dataset and return their weights.

    Returns
    -------
    list of (candidate_name, selected_mode, selected_model, weights, fit_metric)
    """
    fitted = []

    # 1. Equal weight
    if len(eligible_models) > 0:
        w_equal = {m: 1.0 / len(eligible_models) for m in eligible_models}
        score, _ = _evaluate_candidate(oof_df, task, period, w_equal, metric_name=metric_name)
        fitted.append(("equal_weight", "equal_weight", None, w_equal, score))

    # 2. Static convex
    if len(eligible_models) > 1:
        sc_result = fit_static_convex(oof_df, task, period, eligible_models, metric_name=metric_name)
        fitted.append(("static_convex", "static_convex", None, sc_result.weights, sc_result.metric_value))

    # 3. BGEW
    if len(eligible_models) > 1:
        bgew_result = fit_bgew(oof_df, task, period, eligible_models, metric_name=metric_name, tau=tau, eta=eta)
        fitted.append(("bgew", "bgew", None, bgew_result.weights, bgew_result.metric_value))

    # 4. Single models (one-hot)
    for model in eligible_models:
        w_single = {m: 1.0 if m == model else 0.0 for m in eligible_models}
        score, _ = _evaluate_candidate(oof_df, task, period, w_single, metric_name=metric_name)
        fitted.append((f"single_{model}", "single_model", model, w_single, score))

    return fitted


def select_best_candidate(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    eligible_models: list[str],
    *,
    metric_name: str = "sMAPE_floor50",
    tau: float = 30.0,
    eta: float = 0.5,
) -> tuple[CandidateResult, pd.DataFrame]:
    """Select best candidate strategy for one (task, period).

    Parameters
    ----------
    oof_df : pd.DataFrame
        Normalized OOF long-table.
    task : str
        "dayahead" or "realtime"
    period : str
        "1_8", "9_16", or "17_24"
    eligible_models : list[str]
        Models that passed coverage threshold.
    metric_name : str
        Metric to optimize.
    tau : float
        BGEW tau parameter.
    eta : float
        BGEW eta parameter.

    Returns
    -------
    CandidateResult
        Best candidate.
    pd.DataFrame
        Candidate metrics table with full metrics.
    """
    fitted = fit_all_candidates(
        oof_df, task, period, eligible_models,
        metric_name=metric_name, tau=tau, eta=eta,
    )

    if not fitted:
        raise ValueError(f"No candidates available for task={task}, period={period}")

    # Build CandidateResult objects
    candidates = []
    for cand_name, sel_mode, sel_model, weights, fit_metric in fitted:
        _, full_m = _evaluate_candidate(oof_df, task, period, weights, metric_name=metric_name)
        candidates.append(CandidateResult(
            candidate_name=cand_name,
            selected_mode=sel_mode,
            selected_model=sel_model,
            weights=weights,
            metric_value=fit_metric,
            metric_name=metric_name,
            full_metrics=full_m,
        ))

    # Select best (lowest metric, with simplicity tiebreaker)
    # Simplicity preference: when scores are within 0.1% relative tolerance,
    # prefer simpler strategies.  Rank: single_model=0, equal_weight=1, static_convex=2, bgew=3
    _complexity = {"single_model": 0, "equal_weight": 1, "static_convex": 2, "bgew": 3}

    best_score = min(c.metric_value for c in candidates)
    tol = max(abs(best_score) * 1e-3, 1e-9)  # 0.1% relative tolerance

    def _sort_key(c):
        return (0 if c.metric_value <= best_score + tol else 1, _complexity.get(c.selected_mode, 9))

    best = min(candidates, key=_sort_key)

    # Build candidate metrics table with full metrics
    rows = []
    for c in candidates:
        fm = c.full_metrics
        rows.append({
            "task": task,
            "period": period,
            "candidate_name": c.candidate_name,
            "selected_mode": c.selected_mode,
            "model_name_or_fusion": c.selected_model if c.selected_model else "fusion",
            "n": fm.get("n", 0),
            "MAE": fm.get("MAE", float("nan")),
            "RMSE": fm.get("RMSE", float("nan")),
            "sMAPE_floor50": fm.get("sMAPE_floor50", float("nan")),
            "bias_mean": fm.get("bias_mean", float("nan")),
            "bias_median": fm.get("bias_median", float("nan")),
            "q90_high_price_MAE": fm.get("q90_high_price_MAE", float("nan")),
            "q95_high_price_MAE": fm.get("q95_high_price_MAE", float("nan")),
            "intersection_retention_rate": fm.get("intersection_retention_rate", float("nan")),
            "original_rows": fm.get("original_rows", 0),
            "intersection_rows": fm.get("intersection_rows", 0),
        })

    candidate_metrics_df = pd.DataFrame(rows)

    logger.info(
        "Best candidate for task=%s, period=%s: %s (score=%.4f)",
        task, period, best.candidate_name, best.metric_value,
    )

    return best, candidate_metrics_df
