# -*- coding: utf-8 -*-
"""
schema.py — Leakage-safe feature schema for P0/P3 spike risk pipeline.

Defines constants used across build/train/predict scripts to enforce
that prediction-time features never include actual-value or target-leakage
columns at inference time.

Reference:
    ACTUAL_COLS is imported from SGDFNet.data_contract for DRY alignment
    with the main prediction pipeline.  This file adds leakage-specific
    groupings on top.

Constants:
    ACTUAL_VALUE_EXCLUDE_COLS
        Chinese-named actual exogenous columns that must NEVER appear as
        prediction-time features at inference.  They are allowed only as
        historical lagged features (shift(24) or shift(168)) or as labels.
    TARGET_LEAKAGE_COLS
        Columns that directly leak the prediction target (y_true, residuals,
        sMAPE, spike flags).  Must be excluded from feature selection in
        both training and inference.
    LABEL_COLS
        Columns that are the training target / label.  Allowed in the dataset
        only as y, never as X.
    PREDICTION_TIME_ALLOWED
        Feature families that are safe at prediction time.  This list is
        descriptive — the enforcement uses the exclude lists above.
"""

from __future__ import annotations

from typing import Final

# ── Import the canonical actual-value column list ────────────────────────
try:
    from SGDFNet.src.sgdfnet.data_contract import ACTUAL_COLS as _ACTUAL_COLS
except ImportError:
    # Fallback copy if the module cannot be imported (e.g. standalone test)
    _ACTUAL_COLS: list[str] = [
        "地方电厂总加实际值",
        "联络线受电负荷实际值",
        "风电总加实际值",
        "光伏总加实际值",
        "核电总加实际值",
        "自备机组总加实际值",
        "试验机组总加实际值",
        "直调负荷实际值",
        "竞价空间实际值",
        "新能源总加实际值",
    ]

# ── Exogenous actual-value columns (prediction-time forbidden) ──────────
ACTUAL_VALUE_EXCLUDE_COLS: Final[list[str]] = _ACTUAL_COLS
"""
Chinese exogenous actual-value columns.
These must NOT be used as prediction-time features.

Allowed uses:
  - Historical lag with .shift(24) / .shift(168) in the main prediction pipeline
  - Training label (e.g. 实时电价 as y_true)
  - Offline evaluation
"""

# ── Target-leakage columns (prediction-time forbidden) ──────────────────
TARGET_LEAKAGE_COLS: Final[list[str]] = [
    # Actual price columns (available only after the fact)
    "y_true",
    "实时电价",
    "日前电价",
    "realtime_price",
    "dayahead_price",
    # Error / residual columns (computed from y_true)
    "residual",
    "abs_error",
    "smape",
    "smape_floor50",
    # Spike / severe flags (derived from y_true)
    "high_spike",
    "high_spike_flag",
    "high_spike_label",
    "label_high_spike",
    "severe_underestimate",
    "severe_underestimate_flag",
    "spike_label",
    "spike_risk_flag",
]
"""
Columns that leak the prediction target.
Computed from y_true; must NEVER appear in the feature matrix at
training or inference time.
"""

# ── Label columns (allowed only as y) ───────────────────────────────────
LABEL_COLS: Final[list[str]] = [
    "spike_label",
    "label_high_spike",
    "high_spike_flag",
    "severe_underestimate_flag",
]
"""
Training target columns.  Dropped from features before training.
"""

# ── All excluded columns combined (for feature selection) ───────────────
ALL_EXCLUDED_COLS: Final[set[str]] = set(ACTUAL_VALUE_EXCLUDE_COLS + TARGET_LEAKAGE_COLS + LABEL_COLS)
"""
Union of all columns that must not appear as prediction-time features.
"""

# ── Safe feature families (descriptive, for documentation) ──────────────
PREDICTION_TIME_ALLOWED: Final[dict[str, list[str]]] = {
    "calendar": [
        "hour_business", "hour", "weekday", "day_of_week",
        "month", "day_of_month", "is_weekend", "period", "season_bucket",
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
    ],
    "model_predictions": [
        "y_pred",
        "base_fused_pred",
        "final_pred",
        "sgdfnet_pred",
        "timemixer_pred",
        "rt916_pred",
        "timesfm_pred",
    ],
    "model_disagreement": [
        "pred_std",
        "pred_range",
        "pred_max",
        "pred_min",
    ],
    "risk_scores": [
        "spike_risk_score",
        "high_spike_prob",
    ],
    "time_keys": [
        "ds",
        "timestamp",
        "business_day",
        "target_hour",
    ],
}
"""
Feature families that are safe at prediction time (descriptive only).
The enforcement is done via ACTUAL_VALUE_EXCLUDE_COLS + TARGET_LEAKAGE_COLS + LABEL_COLS.
"""
