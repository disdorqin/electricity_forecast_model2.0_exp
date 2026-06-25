"""Candidate selector: compare multiple fusion strategies and pick the best.

Compares:
- equal_weight
- static_convex
- bgew
- each single model (one-hot)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import compute_all_metrics
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


def _evaluate_candidate(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    weights: dict[str, float],
    *,
    metric_name: str = "sMAPE_floor50",
) -> float:
    """Evaluate a candidate weight vector on OOF data."""
    sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)].copy()

    # Pivot to wide
    wide = sub.pivot_table(
        index=["target_day", "ds", "hour_business"],
        columns="model_name",
        values="y_pred",
        aggfunc="first",
    )

    available = [m for m in weights.keys() if m in wide.columns and weights[m] > 0]
    if not available:
        return float("inf")

    wide = wide[available].dropna()

    truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
        ["target_day", "ds", "hour_business", "y_true"]
    ].set_index(["target_day", "ds", "hour_business"])

    wide = wide.join(truth, how="inner")

    if wide.empty:
        return float("inf")

    y_true = wide["y_true"].values.astype(float)
    y_pred = np.zeros(len(y_true))
    for m, w in weights.items():
        if m in wide.columns and w > 0:
            y_pred += w * wide[m].values.astype(float)

    from .metrics import smape_floor50, mae
    metric_fn = smape_floor50 if metric_name == "sMAPE_floor50" else mae
    return metric_fn(y_true, y_pred)


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
        Candidate metrics table.
    """
    candidates = []

    # 1. Equal weight
    if len(eligible_models) > 0:
        w_equal = {m: 1.0 / len(eligible_models) for m in eligible_models}
        score_equal = _evaluate_candidate(oof_df, task, period, w_equal, metric_name=metric_name)
        candidates.append(CandidateResult(
            candidate_name="equal_weight",
            selected_mode="equal_weight",
            selected_model=None,
            weights=w_equal,
            metric_value=score_equal,
            metric_name=metric_name,
        ))

    # 2. Static convex
    if len(eligible_models) > 1:
        sc_result = fit_static_convex(oof_df, task, period, eligible_models, metric_name=metric_name)
        candidates.append(CandidateResult(
            candidate_name="static_convex",
            selected_mode="static_convex",
            selected_model=None,
            weights=sc_result.weights,
            metric_value=sc_result.metric_value,
            metric_name=metric_name,
        ))

    # 3. BGEW
    if len(eligible_models) > 1:
        bgew_result = fit_bgew(oof_df, task, period, eligible_models, metric_name=metric_name, tau=tau, eta=eta)
        candidates.append(CandidateResult(
            candidate_name="bgew",
            selected_mode="bgew",
            selected_model=None,
            weights=bgew_result.weights,
            metric_value=bgew_result.metric_value,
            metric_name=metric_name,
        ))

    # 4. Single models (one-hot)
    for model in eligible_models:
        w_single = {m: 1.0 if m == model else 0.0 for m in eligible_models}
        score_single = _evaluate_candidate(oof_df, task, period, w_single, metric_name=metric_name)
        candidates.append(CandidateResult(
            candidate_name=f"single_{model}",
            selected_mode="single_model",
            selected_model=model,
            weights=w_single,
            metric_value=score_single,
            metric_name=metric_name,
        ))

    # Select best (lowest metric)
    if not candidates:
        raise ValueError(f"No candidates available for task={task}, period={period}")

    best = min(candidates, key=lambda c: c.metric_value)

    # Build candidate metrics table
    rows = []
    for c in candidates:
        rows.append({
            "task": task,
            "period": period,
            "candidate_name": c.candidate_name,
            "selected_mode": c.selected_mode,
            "model_name_or_fusion": c.selected_model if c.selected_model else "fusion",
            "n": len(oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)]),
            "MAE": c.metric_value if c.metric_name == "MAE" else float("nan"),
            "RMSE": float("nan"),
            "sMAPE_floor50": c.metric_value if c.metric_name == "sMAPE_floor50" else float("nan"),
            "bias_mean": float("nan"),
            "bias_median": float("nan"),
            "q90_high_price_MAE": float("nan"),
            "q95_high_price_MAE": float("nan"),
        })

    candidate_metrics_df = pd.DataFrame(rows)

    logger.info(
        "Best candidate for task=%s, period=%s: %s (score=%.4f)",
        task, period, best.candidate_name, best.metric_value,
    )

    return best, candidate_metrics_df
