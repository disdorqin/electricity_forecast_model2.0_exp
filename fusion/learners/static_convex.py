"""Static convex (non-negative simplex) weight optimization.

Fits weights w >= 0, sum(w) = 1 to minimize a chosen metric (default sMAPE_floor50).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .metrics import smape_floor50, mae

logger = logging.getLogger(__name__)


@dataclass
class StaticConvexResult:
    task: str
    period: str
    weights: dict[str, float]
    metric_value: float
    metric_name: str
    n_samples: int


def _pivot_for_optimization(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    eligible_models: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Pivot OOF data into (n_samples, n_models) arrays for optimization.

    Returns
    -------
    y_true_mat : np.ndarray, shape (n_samples,)
    y_pred_mat : np.ndarray, shape (n_samples, n_models)
    model_names : list[str]
        Models in column order.
    """
    sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)].copy()

    # Pivot to wide format
    wide = sub.pivot_table(
        index=["target_day", "ds", "hour_business"],
        columns="model_name",
        values="y_pred",
        aggfunc="first",
    )

    # Keep only eligible models that are present
    available = [m for m in eligible_models if m in wide.columns]
    if not available:
        raise ValueError(f"No eligible models available for task={task}, period={period}")

    wide = wide[available].dropna()

    # Get y_true (should be consistent across models for same time point)
    truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
        ["target_day", "ds", "hour_business", "y_true"]
    ].set_index(["target_day", "ds", "hour_business"])

    wide = wide.join(truth, how="inner")

    y_true = wide["y_true"].values.astype(float)
    y_pred = wide[available].values.astype(float)

    return y_true, y_pred, available


def _objective(
    w: np.ndarray,
    y_true: np.ndarray,
    y_pred_mat: np.ndarray,
    metric_fn: callable,
) -> float:
    """Objective function for optimization."""
    y_pred_fused = y_pred_mat @ w
    return metric_fn(y_true, y_pred_fused)


def fit_static_convex(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    eligible_models: list[str],
    *,
    metric_name: str = "sMAPE_floor50",
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> StaticConvexResult:
    """Fit non-negative simplex weights for one (task, period).

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
        "sMAPE_floor50" or "MAE"
    lower_bound : float
        Minimum weight (default 0.0 for non-negative)
    upper_bound : float
        Maximum weight (default 1.0)

    Returns
    -------
    StaticConvexResult
    """
    metric_fn = smape_floor50 if metric_name == "sMAPE_floor50" else mae

    try:
        y_true, y_pred_mat, models = _pivot_for_optimization(oof_df, task, period, eligible_models)
    except ValueError as e:
        logger.warning("static_convex: %s", e)
        return StaticConvexResult(
            task=task,
            period=period,
            weights={m: 0.0 for m in eligible_models},
            metric_value=float("nan"),
            metric_name=metric_name,
            n_samples=0,
        )

    n_models = len(models)
    if n_models == 1:
        # Single model: weight = 1.0
        return StaticConvexResult(
            task=task,
            period=period,
            weights={models[0]: 1.0},
            metric_value=metric_fn(y_true, y_pred_mat[:, 0]),
            metric_name=metric_name,
            n_samples=len(y_true),
        )

    # Initial guess: equal weights
    w0 = np.ones(n_models) / n_models

    # Constraints: sum(w) = 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: [lower_bound, upper_bound] for each weight
    bounds = [(lower_bound, upper_bound)] * n_models

    # Optimize
    result = minimize(
        _objective,
        w0,
        args=(y_true, y_pred_mat, metric_fn),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success:
        logger.warning(
            "static_convex: optimization failed for task=%s, period=%s: %s",
            task, period, result.message,
        )
        # Fallback to equal weights
        weights = {m: 1.0 / n_models for m in models}
        y_pred_fused = y_pred_mat @ np.array(list(weights.values()))
        metric_val = metric_fn(y_true, y_pred_fused)
    else:
        weights = {m: float(w) for m, w in zip(models, result.x)}
        metric_val = float(result.fun)

    return StaticConvexResult(
        task=task,
        period=period,
        weights=weights,
        metric_value=metric_val,
        metric_name=metric_name,
        n_samples=len(y_true),
    )
