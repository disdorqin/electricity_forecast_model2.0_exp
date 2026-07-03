# -*- coding: utf-8 -*-
"""Metrics for evaluating the residual stack.

Provides:
    compute_stack_metrics  — All required metrics for one configuration.
    compare_configs        — Build a comparison table across configurations.
    format_metrics_table   — Pretty-print for CLI output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ── Thresholds ─────────────────────────────────────────────────────────

HIGH_SPIKE_THRESHOLD: float = 150.0
"""y_true > this is considered a high-spike event."""

LOW_VALLEY_ABSOLUTE: float = 50.0
"""y_true <= this is a low-valley event."""

OVERESTIMATE_THRESHOLD: float = 30.0
"""y_pred - y_true >= this is an overestimate."""


def compute_stack_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    base_pred_col: str = "base_pred",
    final_pred_col: str = "final_pred",
) -> dict[str, Any]:
    """Compute all metrics required for residual stack evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Corrected DataFrame (must contain y_true, base_pred, final_pred,
        hour_business, high_spike_applied, negative_applied).
    base_pred_col : str
        Column with predictions *before* stack corrections.
        Usually ``base_pred`` (which is ``base_fused_pred``).
    final_pred_col : str
        Column with predictions *after* full stack corrections.
        Usually ``final_pred``.

    Returns
    -------
    dict[str, Any]
        Metric dictionary (see docstring for full list).
    """
    metrics: dict[str, Any] = {}

    y_true = df[y_true_col].values.astype(float)
    before = df[base_pred_col].values.astype(float)
    after = df[final_pred_col].values.astype(float)

    is_9_16 = df["hour_business"].between(9, 16).values
    is_normal = ~is_9_16

    # ── Counts ─────────────────────────────────────────────────────
    neg_mask = y_true < 0
    lv_mask = y_true <= LOW_VALLEY_ABSOLUTE
    spike_mask = y_true > HIGH_SPIKE_THRESHOLD
    severe_mask = (y_true - before) > 100  # severe underestimate

    metrics["negative_count"] = int(neg_mask.sum())
    metrics["low_valley_count"] = int(lv_mask.sum())
    metrics["high_spike_count"] = int(spike_mask.sum())
    metrics["severe_underestimate"] = int(severe_mask.sum())

    # ── Negative price MAE ─────────────────────────────────────────
    if metrics["negative_count"] > 0:
        metrics["negative_MAE_before"] = float(
            np.mean(np.abs(y_true[neg_mask] - before[neg_mask]))
        )
        metrics["negative_MAE_after"] = float(
            np.mean(np.abs(y_true[neg_mask] - after[neg_mask]))
        )
    else:
        metrics["negative_MAE_before"] = 0.0
        metrics["negative_MAE_after"] = 0.0

    if metrics.get("negative_MAE_before", 0) > 0:
        metrics["negative_MAE_improvement"] = round(
            (metrics["negative_MAE_before"] - metrics["negative_MAE_after"])
            / metrics["negative_MAE_before"] * 100, 2
        )
    else:
        metrics["negative_MAE_improvement"] = 0.0

    # ── Low valley MAE ─────────────────────────────────────────────
    if metrics["low_valley_count"] > 0:
        metrics["low_valley_MAE_before"] = float(
            np.mean(np.abs(y_true[lv_mask] - before[lv_mask]))
        )
        metrics["low_valley_MAE_after"] = float(
            np.mean(np.abs(y_true[lv_mask] - after[lv_mask]))
        )
    else:
        metrics["low_valley_MAE_before"] = 0.0
        metrics["low_valley_MAE_after"] = 0.0

    if metrics.get("low_valley_MAE_before", 0) > 0:
        metrics["low_valley_MAE_improvement"] = round(
            (metrics["low_valley_MAE_before"] - metrics["low_valley_MAE_after"])
            / metrics["low_valley_MAE_before"] * 100, 2
        )
    else:
        metrics["low_valley_MAE_improvement"] = 0.0

    # ── Negative miss (y_true < 0 but y_pred >= 0) ─────────────────
    metrics["negative_miss_before"] = int(np.sum((y_true < 0) & (before >= 0)))
    metrics["negative_miss_after"] = int(np.sum((y_true < 0) & (after >= 0)))

    # ── Low valley overestimate ────────────────────────────────────
    if metrics["low_valley_count"] > 0:
        metrics["low_valley_overestimate_before"] = int(
            np.sum((before[lv_mask] - y_true[lv_mask]) >= OVERESTIMATE_THRESHOLD)
        )
        metrics["low_valley_overestimate_after"] = int(
            np.sum((after[lv_mask] - y_true[lv_mask]) >= OVERESTIMATE_THRESHOLD)
        )
    else:
        metrics["low_valley_overestimate_before"] = 0
        metrics["low_valley_overestimate_after"] = 0

    # ── Overall sMAPE (floor 50) ───────────────────────────────────
    def _smape(actual: np.ndarray, pred: np.ndarray) -> float:
        denom = (np.abs(actual) + np.abs(pred)) / 2.0
        denom = np.clip(denom, 50.0, None)
        return float(np.mean(np.abs(actual - pred) / denom * 100))

    smape_before = _smape(y_true, before)
    smape_after = _smape(y_true, after)
    metrics["overall_sMAPE_before"] = round(smape_before, 4)
    metrics["overall_sMAPE_after"] = round(smape_after, 4)
    metrics["overall_sMAPE_improvement"] = round(smape_before - smape_after, 4)

    # ── High spike MAE ─────────────────────────────────────────────
    if metrics["high_spike_count"] > 0:
        metrics["high_spike_MAE_before"] = float(
            np.mean(np.abs(y_true[spike_mask] - before[spike_mask]))
        )
        metrics["high_spike_MAE_after"] = float(
            np.mean(np.abs(y_true[spike_mask] - after[spike_mask]))
        )
    else:
        metrics["high_spike_MAE_before"] = 0.0
        metrics["high_spike_MAE_after"] = 0.0

    if metrics.get("high_spike_MAE_before", 0) > 0:
        metrics["high_spike_MAE_improvement"] = round(
            (metrics["high_spike_MAE_before"] - metrics["high_spike_MAE_after"])
            / metrics["high_spike_MAE_before"] * 100, 2
        )
    else:
        metrics["high_spike_MAE_improvement"] = 0.0

    # ── False lift rate ────────────────────────────────────────────
    if "high_spike_applied" in df.columns:
        hs_applied = df["high_spike_applied"].values.astype(bool)
        if hs_applied.sum() > 0:
            correct_lift = (
                df.loc[hs_applied, final_pred_col].values
                > df.loc[hs_applied, y_true_col].values
            )
            metrics["false_lift_rate"] = round(
                1.0 - float(np.mean(correct_lift)), 4
            )
        else:
            metrics["false_lift_rate"] = 0.0
    else:
        metrics["false_lift_rate"] = None

    # ── Normal degradation ─────────────────────────────────────────
    if is_normal.sum() > 0:
        yt_n = y_true[is_normal]
        bf_n = before[is_normal]
        af_n = after[is_normal]

        def _smape_vec(act: np.ndarray, pred: np.ndarray) -> float:
            denom = (np.abs(act) + np.abs(pred)) / 2.0
            denom = np.clip(denom, 50.0, None)
            return float(np.mean(np.abs(act - pred) / denom * 100))

        smape_n_before = _smape_vec(yt_n, bf_n)
        smape_n_after = _smape_vec(yt_n, af_n)
        metrics["normal_sMAPE_before"] = round(smape_n_before, 4)
        metrics["normal_sMAPE_after"] = round(smape_n_after, 4)
        metrics["normal_degradation"] = round(smape_n_after - smape_n_before, 4)
    else:
        metrics["normal_degradation"] = 0.0

    # ── Data-limited flag ──────────────────────────────────────────
    metrics["data_limited"] = metrics["negative_count"] < 5

    return metrics


# ── Comparison ─────────────────────────────────────────────────────────


def compare_configs(configs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a comparison DataFrame across multiple config results.

    Parameters
    ----------
    configs : dict[str, dict[str, Any]]
        Mapping of config label → metrics dict (as returned by
        :func:`compute_stack_metrics`).

    Returns
    -------
    pd.DataFrame
        Index = metric, columns = config labels.
    """
    rows: list[str] = []
    for label, metrics in configs.items():
        if not rows:
            rows = list(metrics.keys())
    data: dict[str, list[Any]] = {label: [] for label in configs}
    for metric in rows:
        for label, metrics in configs.items():
            data[label].append(metrics.get(metric, "N/A"))
    return pd.DataFrame(data, index=rows)


def format_metrics_table(metrics: dict[str, Any]) -> str:
    """Format metrics dict as a human-readable table string."""
    lines: list[str] = ["Residual Stack Metrics", "=" * 40]
    key_order = [
        "negative_count", "low_valley_count", "high_spike_count",
        "severe_underestimate",
        "negative_MAE_before", "negative_MAE_after", "negative_MAE_improvement",
        "low_valley_MAE_before", "low_valley_MAE_after", "low_valley_MAE_improvement",
        "negative_miss_before", "negative_miss_after",
        "low_valley_overestimate_before", "low_valley_overestimate_after",
        "overall_sMAPE_before", "overall_sMAPE_after", "overall_sMAPE_improvement",
        "high_spike_MAE_before", "high_spike_MAE_after", "high_spike_MAE_improvement",
        "false_lift_rate",
        "normal_sMAPE_before", "normal_sMAPE_after", "normal_degradation",
        "data_limited",
    ]
    for key in key_order:
        if key in metrics:
            val = metrics[key]
            lines.append(f"  {key}: {val}")
    for key in sorted(set(metrics) - set(key_order)):
        lines.append(f"  {key}: {metrics[key]}")
    return "\n".join(lines)
