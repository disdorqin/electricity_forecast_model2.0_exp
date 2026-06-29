"""BGEW: Backward-Gated Expert Weighting.

Time-decayed multiplicative weights update with recency bias.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import smape_floor50, mae

logger = logging.getLogger(__name__)


@dataclass
class BGEWResult:
    task: str
    period: str
    weights: dict[str, float]
    metric_value: float
    metric_name: str
    n_samples: int
    trace: list[dict] = field(default_factory=list)


def fit_bgew(
    oof_df: pd.DataFrame,
    task: str,
    period: str,
    eligible_models: list[str],
    *,
    metric_name: str = "sMAPE_floor50",
    tau: float = 30.0,
    eta: float = 0.5,
    loss_clip: float = 5.0,
) -> BGEWResult:
    """Fit BGEW weights for one (task, period).

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
    tau : float
        Time constant for gate decay (days). Default 30.
    eta : float
        Learning rate for weight update. Default 0.5.
    loss_clip : float
        Clip normalized loss to avoid outlier explosion. Default 5.0.

    Returns
    -------
    BGEWResult
    """
    metric_fn = smape_floor50 if metric_name == "sMAPE_floor50" else mae

    sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)].copy()

    # Pivot to wide format
    wide = sub.pivot_table(
        index=["target_day", "ds", "hour_business"],
        columns="model_name",
        values="y_pred",
        aggfunc="first",
    )

    available = [m for m in eligible_models if m in wide.columns]
    if not available:
        logger.warning("bgew: no eligible models for task=%s, period=%s", task, period)
        return BGEWResult(
            task=task,
            period=period,
            weights={m: 0.0 for m in eligible_models},
            metric_value=float("nan"),
            metric_name=metric_name,
            n_samples=0,
        )

    wide = wide[available].dropna()

    # Get y_true
    truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
        ["target_day", "ds", "hour_business", "y_true"]
    ].set_index(["target_day", "ds", "hour_business"])

    wide = wide.join(truth, how="inner")

    if wide.empty:
        logger.warning("bgew: no data after join for task=%s, period=%s", task, period)
        return BGEWResult(
            task=task,
            period=period,
            weights={m: 1.0 / len(available) for m in available},
            metric_value=float("nan"),
            metric_name=metric_name,
            n_samples=0,
        )

    # Sort by target_day (most recent last)
    wide = wide.sort_index(level=0)

    y_true_all = wide["y_true"].values.astype(float)
    y_pred_all = wide[available].values.astype(float)

    # Map each row to its target_day
    row_target_days = wide.index.get_level_values(0)
    unique_days = sorted(row_target_days.unique())

    # Convert unique_days to numeric offsets
    day_offsets_unique = pd.to_datetime(unique_days)
    day_offsets_unique = (day_offsets_unique - day_offsets_unique.min()).days.values
    latest_day = day_offsets_unique.max()

    # Create mapping from day string to offset
    day_to_offset = {day: offset for day, offset in zip(unique_days, day_offsets_unique)}

    # Map each row to its day offset
    row_day_offsets = np.array([day_to_offset[day] for day in row_target_days])

    # Initialize weights: equal
    n_models = len(available)
    weights = np.ones(n_models) / n_models

    trace = []
    step_index = 0

    # Process each unique day
    for day_offset in sorted(set(day_offsets_unique)):
        day_mask = row_day_offsets == day_offset
        y_true_day = y_true_all[day_mask]
        y_pred_day = y_pred_all[day_mask]

        # Compute loss per model
        losses = []
        for m_idx in range(n_models):
            loss = metric_fn(y_true_day, y_pred_day[:, m_idx])
            losses.append(loss)
        losses = np.array(losses)

        # Normalize loss by median
        median_loss = np.median(losses)
        if median_loss < 1e-9:
            median_loss = 1.0
        normalized_losses = losses / median_loss

        # Clip
        normalized_losses = np.clip(normalized_losses, 0.0, loss_clip)

        # Compute gate
        age_days = latest_day - day_offset
        gate = np.exp(-age_days / tau)

        # Update weights
        weights_before = weights.copy()
        weights = weights * np.exp(-eta * gate * normalized_losses)
        weights = weights / weights.sum()

        # Record trace
        target_day_str = unique_days[list(day_offsets_unique).index(day_offset)]
        for m_idx, model_name in enumerate(available):
            trace.append({
                "task": task,
                "period": period,
                "target_day": target_day_str,
                "step_index": step_index,
                "model_name": model_name,
                "weight_before": float(weights_before[m_idx]),
                "loss": float(losses[m_idx]),
                "normalized_loss": float(normalized_losses[m_idx]),
                "gate_time": float(gate),
                "weight_after": float(weights[m_idx]),
            })

        step_index += 1

    # Final weights
    weights_dict = {m: float(w) for m, w in zip(available, weights)}

    # Compute final metric
    y_pred_fused = y_pred_all @ weights
    metric_val = metric_fn(y_true_all, y_pred_fused)

    return BGEWResult(
        task=task,
        period=period,
        weights=weights_dict,
        metric_value=metric_val,
        metric_name=metric_name,
        n_samples=len(y_true_all),
        trace=trace,
    )
