# -*- coding: utf-8 -*-
"""schema.py — Label definitions, constants, and column names for negative price module."""
from __future__ import annotations
from typing import Final

NEGATIVE_PRICE_COL: Final[str] = "label_negative_price"
LOW_VALLEY_COL: Final[str] = "label_low_valley"
OVERESTIMATE_LOW_COL: Final[str] = "label_overestimate_low"

NEGATIVE_PRICE_THRESHOLD: Final[float] = 0.0
LOW_VALLEY_ABSOLUTE_THRESHOLD: Final[float] = 50.0
LOW_VALLEY_PERCENTILE: Final[float] = 0.10
OVERESTIMATE_LOW_THRESHOLD: Final[float] = 30.0

TARGET_LEAKAGE_COLS: Final[list[str]] = [
    "y_true", "实时电价", "realtime_price",
    "residual", "abs_error", "smape", "smape_floor50", "smape_floor50_neg",
    NEGATIVE_PRICE_COL, LOW_VALLEY_COL, OVERESTIMATE_LOW_COL,
    "high_spike", "high_spike_flag", "label_high_spike",
    "severe_underestimate", "severe_underestimate_flag",
    "negative_correction_applied", "downward_lift", "negative_correction_reason",
]

SAFE_FEATURE_FAMILIES: Final[dict[str, list[str]]] = {
    "time_features": [
        "hour_business", "hour", "period", "weekday",
        "day_of_week", "month", "day_of_month", "is_weekend", "season_bucket",
    ],
    "prediction_signals": [
        "base_fused_pred", "final_pred", "final_pred_before_negative",
        "sgdfnet_pred", "timemixer_pred", "rt916_pred", "timesfm_pred",
        "dayahead_proxy", "prediction_spread", "model_disagreement",
        "pred_std", "pred_range",
    ],
    "forecast_exogenous": [
        "地方电厂总加预测值", "联络线受电负荷预测值",
        "风电总加预测值", "光伏总加预测值", "核电总加预测值",
        "自备机组总加预测值", "试验机组总加预测值",
        "直调负荷预测值", "竞价空间预测值", "新能源总加预测值",
        "风电总加预测值_rolling_avg_24h", "光伏总加预测值_rolling_avg_24h",
        "直调负荷预测值_rolling_avg_24h", "竞价空间预测值_rolling_avg_24h",
    ],
    "negative_risk_signals": [
        "recent_negative_rate_by_hour", "recent_negative_rate_by_period",
        "recent_low_price_rate_by_hour", "recent_low_price_rate_by_period",
        "recent_mean_low_residual_by_hour",
        "min_pred_last_24h", "renewable_ratio",
    ],
}
