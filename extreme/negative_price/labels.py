# -*- coding: utf-8 -*-
"""labels.py — Label generation for negative price and low-valley regimes."""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
from extreme.negative_price.schema import (
    LOW_VALLEY_COL, NEGATIVE_PRICE_COL, NEGATIVE_PRICE_THRESHOLD,
    LOW_VALLEY_ABSOLUTE_THRESHOLD, LOW_VALLEY_PERCENTILE,
    OVERESTIMATE_LOW_COL, OVERESTIMATE_LOW_THRESHOLD,
)


def generate_negative_price_labels(df: pd.DataFrame, y_true_col: str = "y_true") -> pd.Series:
    return (df[y_true_col] < NEGATIVE_PRICE_THRESHOLD).astype(int)


def compute_low_valley_percentile(
    df: pd.DataFrame, y_true_col: str = "y_true",
    percentile: float = LOW_VALLEY_PERCENTILE,
) -> float:
    vals = df[y_true_col].dropna().values
    if len(vals) == 0:
        return LOW_VALLEY_ABSOLUTE_THRESHOLD
    return float(np.percentile(vals, percentile * 100))


def generate_low_valley_labels(
    df: pd.DataFrame, y_true_col: str = "y_true",
    percentile_threshold: Optional[float] = None,
    absolute_threshold: float = LOW_VALLEY_ABSOLUTE_THRESHOLD,
) -> pd.Series:
    """label = 1 if y_true <= max(percentile_threshold, absolute_threshold), else 0."""
    effective = absolute_threshold
    if percentile_threshold is not None:
        effective = max(effective, percentile_threshold)
    return (df[y_true_col] <= effective).astype(int)


def generate_overestimate_low_labels(
    df: pd.DataFrame, y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    threshold: float = OVERESTIMATE_LOW_THRESHOLD,
) -> pd.Series:
    mask = df[y_true_col].notna() & df[y_pred_col].notna()
    result = pd.Series(0, index=df.index, dtype=float)
    result.loc[mask] = ((df.loc[mask, y_pred_col] - df.loc[mask, y_true_col]) >= threshold).astype(float)
    result.loc[~mask] = np.nan
    return result


def add_all_labels(
    df: pd.DataFrame, y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df[NEGATIVE_PRICE_COL] = generate_negative_price_labels(df, y_true_col)
    percentile_val = None
    if history_df is not None:
        percentile_val = compute_low_valley_percentile(history_df, y_true_col)
    df[LOW_VALLEY_COL] = generate_low_valley_labels(df, y_true_col, percentile_threshold=percentile_val)
    df[OVERESTIMATE_LOW_COL] = generate_overestimate_low_labels(df, y_true_col, y_pred_col)
    return df
