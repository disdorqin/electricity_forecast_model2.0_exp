"""Coverage report: check each (task, period, model) has sufficient data."""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_coverage_report(
    oof_df: pd.DataFrame,
    *,
    coverage_threshold: float = 0.95,
) -> pd.DataFrame:
    """Compute coverage report for all (task, period, model) combinations.

    Parameters
    ----------
    oof_df : pd.DataFrame
        Normalized OOF long-table.
    coverage_threshold : float
        Minimum coverage ratio for eligibility (default 0.95).

    Returns
    -------
    pd.DataFrame
        Coverage report with columns:
        task, period, model_name, expected_rows, available_rows,
        coverage, nan_y_pred_count, eligible, reason
    """
    # Determine expected rows per (task, period):
    # For each (task, period), the expected count is the max number of
    # unique (target_day, ds, hour_business) points any model has.
    groups = []
    for (task, period), grp in oof_df.groupby(["task", "period"]):
        # Expected = total unique time points in this (task, period)
        time_points = grp.drop_duplicates(subset=["target_day", "ds", "hour_business"])
        expected = len(time_points)

        for model_name, model_grp in grp.groupby("model_name"):
            available = len(model_grp.drop_duplicates(subset=["target_day", "ds", "hour_business"]))
            nan_count = int(model_grp["y_pred"].isna().sum())
            coverage = available / expected if expected > 0 else 0.0
            eligible = coverage >= coverage_threshold
            reason = ""
            if not eligible:
                reason = f"coverage {coverage:.2%} < threshold {coverage_threshold:.2%}"

            groups.append({
                "task": task,
                "period": period,
                "model_name": model_name,
                "expected_rows": expected,
                "available_rows": available,
                "coverage": round(coverage, 4),
                "nan_y_pred_count": nan_count,
                "eligible": eligible,
                "reason": reason,
            })

    report = pd.DataFrame(groups)
    n_ineligible = (~report["eligible"]).sum()
    if n_ineligible > 0:
        logger.warning(
            "Coverage report: %d model(s) below threshold %.0f%%",
            n_ineligible,
            coverage_threshold * 100,
        )
    else:
        logger.info("Coverage report: all models eligible (threshold=%.0f%%)", coverage_threshold * 100)

    return report


def get_eligible_models(
    coverage_report: pd.DataFrame,
    *,
    task: str,
    period: str,
) -> list[str]:
    """Return list of eligible model names for a (task, period)."""
    sub = coverage_report[
        (coverage_report["task"] == task)
        & (coverage_report["period"] == period)
        & coverage_report["eligible"]
    ]
    return sub["model_name"].tolist()
