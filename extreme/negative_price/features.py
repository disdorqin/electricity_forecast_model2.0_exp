# -*- coding: utf-8 -*-
"""features.py — Leakage-safe feature engineering for negative price correction."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from extreme.negative_price.schema import (
    SAFE_FEATURE_FAMILIES, TARGET_LEAKAGE_COLS, NEGATIVE_PRICE_THRESHOLD,
)


def _infer_period(hour_business: pd.Series) -> pd.Series:
    return pd.cut(hour_business, bins=[0, 8, 16, 24],
                  labels=["1_8", "9_16", "17_24"], include_lowest=True).astype(str)


def _compute_low_valley_rate(pred_series: pd.Series, threshold: float = 50.0) -> float:
    if len(pred_series) == 0:
        return 0.0
    return float((pred_series <= threshold).mean())


def _compute_negative_rate(pred_series: pd.Series) -> float:
    if len(pred_series) == 0:
        return 0.0
    return float((pred_series < NEGATIVE_PRICE_THRESHOLD).mean())


def engineer_negative_price_features(
    df: pd.DataFrame, *,
    history_df: Optional[pd.DataFrame] = None,
    history_lookback_days: int = 30,
    pred_col: str = "base_fused_pred",
) -> pd.DataFrame:
    result = df.copy()
    drop_cols = [c for c in TARGET_LEAKAGE_COLS if c in result.columns]
    if drop_cols:
        result.drop(columns=drop_cols, inplace=True, errors="ignore")

    if "hour" not in result.columns and "ds" in result.columns:
        result["hour"] = pd.to_datetime(result["ds"]).dt.hour
    if "period" not in result.columns and "hour_business" in result.columns:
        result["period"] = _infer_period(result["hour_business"])
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
                lambda m: {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
                           6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}.get(m, 0))

    individual_pred_cols = [c for c in result.columns
                            if c.endswith("_pred") and c not in (pred_col, "y_pred")]
    if len(individual_pred_cols) >= 2 and pred_col in result.columns:
        pred_vals = result[individual_pred_cols]
        result["prediction_spread"] = pred_vals.std(axis=1, skipna=True)
        result["model_disagreement"] = pred_vals.std(axis=1, skipna=True)
        result["pred_range"] = pred_vals.max(axis=1, skipna=True) - pred_vals.min(axis=1, skipna=True)
    elif pred_col in result.columns:
        result["prediction_spread"] = 0.0
        result["model_disagreement"] = 0.0
        result["pred_range"] = 0.0

    if pred_col in result.columns and "final_pred_before_negative" not in result.columns:
        result["final_pred_before_negative"] = result[pred_col]
    if pred_col in result.columns:
        result["dayahead_proxy"] = result[pred_col]

    renewable_keywords = ["风电总加预测值", "光伏总加预测值", "新能源总加预测值"]
    load_keywords = ["直调负荷预测值"]
    renewable_cols = [c for c in result.columns if any(k in c for k in renewable_keywords)]
    load_cols = [c for c in result.columns if any(k in c for k in load_keywords)]
    if renewable_cols and load_cols:
        total_renewable = result[renewable_cols].sum(axis=1, skipna=True).fillna(0)
        total_load = result[load_cols].sum(axis=1, skipna=True).replace(0, np.nan).fillna(1)
        result["renewable_ratio"] = (total_renewable / total_load).clip(0, 2)
    else:
        result["renewable_ratio"] = 0.0

    if history_df is not None and pred_col in history_df.columns:
        hb_lookup = history_df["hour_business"] if "hour_business" in history_df.columns else None
        period_lookup = (_infer_period(history_df["hour_business"])
                         if "hour_business" in history_df.columns else None)
        hist_pred = history_df[pred_col]
    elif pred_col in result.columns:
        hb_lookup = result["hour_business"] if "hour_business" in result.columns else None
        period_lookup = result["period"] if "period" in result.columns else None
        hist_pred = result[pred_col]
    else:
        hb_lookup = period_lookup = hist_pred = None

    if hb_lookup is not None and hist_pred is not None:
        rate_df = pd.DataFrame({"pred": hist_pred, "hb": hb_lookup})
        if period_lookup is not None:
            rate_df["period"] = period_lookup

        # Add residual info from history_df when available
        if history_df is not None and "y_true" in history_df.columns:
            rate_df["residual"] = (history_df["y_true"] - history_df[pred_col]).fillna(0.0)
            # Mean residual in low-price regime per hour
            low_mask = history_df["y_true"] <= 50
            low_residual = rate_df.loc[low_mask.values, "residual"] if low_mask.any() else pd.Series([0.0])
            hr_low_res = (
                history_df.loc[low_mask.values, "hour_business"] if low_mask.any() else pd.Series(dtype=float)
            )
            if low_mask.any():
                low_res_df = pd.DataFrame({"res": low_residual.values, "hb": hr_low_res.values})
                hr_mean_low_res = low_res_df.groupby("hb")["res"].mean()
                result["recent_mean_low_residual_by_hour"] = (
                    result["hour_business"].map(hr_mean_low_res).fillna(0.0)
                )
            else:
                result["recent_mean_low_residual_by_hour"] = 0.0
        else:
            result["recent_mean_low_residual_by_hour"] = 0.0

        hr_neg = rate_df.groupby("hb")["pred"].apply(_compute_negative_rate)
        result["recent_negative_rate_by_hour"] = result["hour_business"].map(hr_neg).fillna(0.0)
        if "period" in rate_df.columns:
            per_neg = rate_df.groupby("period")["pred"].apply(_compute_negative_rate)
            result["recent_negative_rate_by_period"] = result["period"].map(per_neg).fillna(0.0)
        hr_low = rate_df.groupby("hb")["pred"].apply(lambda x: _compute_low_valley_rate(x, 50.0))
        result["recent_low_price_rate_by_hour"] = result["hour_business"].map(hr_low).fillna(0.0)
        if "period" in rate_df.columns:
            per_low = rate_df.groupby("period")["pred"].apply(lambda x: _compute_low_valley_rate(x, 50.0))
            result["recent_low_price_rate_by_period"] = result["period"].map(per_low).fillna(0.0)
        hr_min = rate_df.groupby("hb")["pred"].min()
        result["min_pred_last_24h"] = result["hour_business"].map(hr_min).fillna(0.0)
    else:
        for c in ["recent_negative_rate_by_hour", "recent_negative_rate_by_period",
                   "recent_low_price_rate_by_hour", "recent_low_price_rate_by_period",
                   "recent_mean_low_residual_by_hour",
                   "min_pred_last_24h"]:
            result[c] = 0.0

    return result


def get_feature_columns() -> list[str]:
    all_features = []
    for _family_name, cols in SAFE_FEATURE_FAMILIES.items():
        all_features.extend(cols)
    return all_features


def select_feature_columns(df: pd.DataFrame, extra_cols: Optional[list[str]] = None) -> list[str]:
    candidates = get_feature_columns()
    if extra_cols:
        candidates.extend(extra_cols)
    return [c for c in candidates if c in df.columns]
