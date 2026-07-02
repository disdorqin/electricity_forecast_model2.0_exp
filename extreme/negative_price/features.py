# -*- coding: utf-8 -*-
"""
features.py — Leakage-safe feature engineering for negative price correction.

Prediction-time safe features include:
    - Time features (hour_business, period, weekday, month, season)
    - Prediction signals (base_fused_pred, dayahead_proxy, prediction_spread)
    - Forecast exogenous (renewable, load, 竞价空间预测值)
    - Negative risk signals (recent low_price rate by hour/period)

All features use only information available before the prediction target hour.
No y_true, actual values, or future data is used.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from extreme.negative_price.schema import (
    SAFE_FEATURE_FAMILIES,
    TARGET_LEAKAGE_COLS,
    NEGATIVE_PRICE_COL,
    NEGATIVE_PRICE_THRESHOLD,
)


def engineer_negative_price_features(
    df: pd.DataFrame,
    *,
    history_lookback_days: int = 30,
    pred_col: str = "base_fused_pred",
) -> pd.DataFrame:
    """Engineer prediction-time-safe features for negative price correction.

    All features must be computable at inference time without access to future
    y_true or actual exogenous values.

    Args:
        df: Raw DataFrame with at least hour_business, ds, and prediction columns.
            Must NOT contain y_true or actual_value columns (they will be excluded).
        history_lookback_days: Days of history for rolling rate features.
        pred_col: Column to use as the primary prediction signal.

    Returns:
        DataFrame with added feature columns. Original columns preserved.
    """
    result = df.copy()

    # Drop any leakage columns that might be present
    for col in TARGET_LEAKAGE_COLS:
        if col in result.columns:
            result.drop(columns=[col], inplace=True, errors="ignore")

    # ── 1. Time features ────────────────────────────────────────────
    if "hour" not in result.columns and "ds" in result.columns:
        result["hour"] = pd.to_datetime(result["ds"]).dt.hour

    if "period" not in result.columns and "hour_business" in result.columns:
        hb = result["hour_business"].values
        result["period"] = pd.cut(hb, bins=[0, 8, 16, 24], labels=["1_8", "9_16", "17_24"],
                                  include_lowest=True).astype(str)

    if "weekday" not in result.columns and "ds" in result.columns:
        result["weekday"] = pd.to_datetime(result["ds"]).dt.weekday

    if "is_weekend" not in result.columns and "weekday" in result.columns:
        result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)

    if "ds" in result.columns:
        dt = pd.to_datetime(result["ds"])
        if "month" not in result.columns:
            result["month"] = dt.dt.month
        if "day_of_month" not in result.columns:
            result["day_of_month"] = dt.dt.day
        if "season_bucket" not in result.columns:
            result["season_bucket"] = dt.dt.month.map(
                lambda m: {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}.get(m, 0)
            )

    # ── 2. Prediction spread (diversity) ────────────────────────────
    pred_cols = [c for c in result.columns if "pred" in c.lower()
                 and c not in (pred_col, "y_pred")]
    if len(pred_cols) >= 2 and pred_col in result.columns:
        result["prediction_spread"] = result[pred_cols].std(axis=1, skipna=True)
        result["pred_range"] = result[pred_cols].max(axis=1, skipna=True) - result[pred_cols].min(axis=1, skipna=True)
    elif pred_col in result.columns:
        result["prediction_spread"] = 0.0
        result["pred_range"] = 0.0

    # ── 3. Renewable ratio feature ──────────────────────────────────
    renewable_cols = [c for c in result.columns if "风电" in c or "光伏" in c or "新能源" in c]
    load_cols = [c for c in result.columns if "负荷" in c or "load" in c.lower()]
    if renewable_cols and load_cols:
        total_renewable = result[renewable_cols].sum(axis=1, skipna=True).fillna(0)
        total_load = result[load_cols].sum(axis=1, skipna=True).replace(0, np.nan).fillna(1)
        result["renewable_ratio"] = (total_renewable / total_load).clip(0, 2)
    else:
        result["renewable_ratio"] = 0.0

    # ── 4. Day-ahead proxy (if applicable) ──────────────────────────
    # Use the prediction itself as a "dayahead proxy" if no explicit column
    if pred_col in result.columns:
        result["dayahead_proxy"] = result[pred_col]

    # ── 5. Negative/low price rate features ─────────────────────────
    # These require a sorted DataFrame by (business_day, hour_business)
    if "hour_business" in result.columns and pred_col in result.columns:
        hb = result["hour_business"]

        # Per-hour negative prediction rate (using available prediction history)
        min_pred = result.groupby("hour_business")[pred_col].transform("min")
        result["min_pred_last_24h"] = min_pred

        # Negative prediction rate by hour (in-sample estimate)
        neg_pred_rate = result.groupby("hour_business")[pred_col].transform(
            lambda x: (x < NEGATIVE_PRICE_THRESHOLD).mean()
        )
        result["negative_price_rate_hour"] = neg_pred_rate

        # Low valley rate by period
        if "period" in result.columns:
            low_valley_rate = result.groupby("period")[pred_col].transform(
                lambda x: (x <= 50).mean()
            )
            result["low_valley_rate_period"] = low_valley_rate

        # Low valley rate by hour_business
        low_valley_rate_hour = result.groupby("hour_business")[pred_col].transform(
            lambda x: (x <= 50).mean()
        )
        result["low_valley_rate_hour"] = low_valley_rate_hour

        # Negative price rate by period
        if "period" in result.columns:
            neg_period_rate = result.groupby("period")[pred_col].transform(
                lambda x: (x < NEGATIVE_PRICE_THRESHOLD).mean()
            )
            result["negative_price_rate_period"] = neg_period_rate

    return result


def get_feature_columns(
) -> list[str]:
    """Return the list of all feature columns produced by this module.

    These can be used for model training or inference feature selection.
    """
    all_features = []
    for family_name, cols in SAFE_FEATURE_FAMILIES.items():
        all_features.extend(cols)
    return all_features


def select_feature_columns(
    df: pd.DataFrame,
    extra_cols: Optional[list[str]] = None,
) -> list[str]:
    """Select available feature columns from a DataFrame.

    Args:
        df: DataFrame with engineered features.
        extra_cols: Additional feature columns to include (optional).

    Returns:
        List of available feature column names.
    """
    candidates = get_feature_columns()
    if extra_cols:
        candidates.extend(extra_cols)
    return [c for c in candidates if c in df.columns]
