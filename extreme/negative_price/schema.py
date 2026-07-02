# -*- coding: utf-8 -*-
"""
schema.py — Label definitions, constants, and column names for negative price module.

Defines:
    - Label column names (negative_price, low_valley, overestimate_low)
    - Threshold constants
    - Feature column families
    - Excluded columns (prediction-time forbidden)
"""

from __future__ import annotations

from typing import Final

# ── Label constants ───────────────────────────────────────────────────

NEGATIVE_PRICE_COL: Final[str] = "label_negative_price"
"""y_true < 0 binary label."""

LOW_VALLEY_COL: Final[str] = "label_low_valley"
"""y_true <= p10 OR y_true <= 50 (configurable) binary label."""

OVERESTIMATE_LOW_COL: Final[str] = "label_overestimate_low"
"""y_pred - y_true >= threshold binary label."""

# ── Threshold defaults ────────────────────────────────────────────────

NEGATIVE_PRICE_THRESHOLD: Final[float] = 0.0
"""y_true < 0 = negative price."""

LOW_VALLEY_ABSOLUTE_THRESHOLD: Final[float] = 50.0
"""y_true <= 50 is considered low valley."""

LOW_VALLEY_PERCENTILE: Final[float] = 0.10
"""y_true <= p10 is considered low valley."""

OVERESTIMATE_LOW_THRESHOLD: Final[float] = 30.0
"""y_pred - y_true >= 30 is considered overestimate_low."""

# ── Prediction-time forbidden columns ─────────────────────────────────

TARGET_LEAKAGE_COLS: Final[list[str]] = [
    # Actual price / future values
    "y_true",
    "实时电价",
    "realtime_price",
    # Error / residual columns
    "residual",
    "abs_error",
    "smape",
    "smape_floor50",
    "smape_floor50_neg",
    # Label columns (training target only)
    NEGATIVE_PRICE_COL,
    LOW_VALLEY_COL,
    OVERESTIMATE_LOW_COL,
    # Spike / high-price columns
    "high_spike",
    "high_spike_flag",
    "label_high_spike",
    "severe_underestimate",
    "severe_underestimate_flag",
    # Correction columns
    "negative_correction_applied",
    "downward_lift",
    "negative_correction_reason",
]
"""
Columns that must NEVER appear as prediction-time features.
"""

# ── Safe feature families ─────────────────────────────────────────────

SAFE_FEATURE_FAMILIES: Final[dict[str, list[str]]] = {
    "time_features": [
        "hour_business", "hour", "period", "weekday",
        "day_of_week", "month", "day_of_month", "is_weekend",
        "season_bucket",
    ],
    "prediction_signals": [
        "base_fused_pred",
        "final_pred",
        "sgdfnet_pred",
        "timemixer_pred",
        "rt916_pred",
        "timesfm_pred",
        "dayahead_proxy",
        "prediction_spread",
        "pred_std",
        "pred_range",
    ],
    "forecast_exogenous": [
        "地方电厂总加预测值",
        "联络线受电负荷预测值",
        "风电总加预测值",
        "光伏总加预测值",
        "核电总加预测值",
        "自备机组总加预测值",
        "试验机组总加预测值",
        "直调负荷预测值",
        "竞价空间预测值",
        "新能源总加预测值",
        "风电总加预测值_rolling_avg_24h",
        "光伏总加预测值_rolling_avg_24h",
        "直调负荷预测值_rolling_avg_24h",
        "竞价空间预测值_rolling_avg_24h",
    ],
    "negative_risk_signals": [
        "negative_price_rate_hour",
        "negative_price_rate_period",
        "low_valley_rate_hour",
        "low_valley_rate_period",
        "min_pred_last_24h",
        "renewable_ratio",
    ],
}
"""
Feature families that are safe at prediction time.
Enforced via exclude lists — these are documented categories.
"""
