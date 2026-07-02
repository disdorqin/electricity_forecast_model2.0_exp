# -*- coding: utf-8 -*-
"""
labels.py — Label generation for negative price and low-valley regimes.

Provides:
    - generate_negative_price_labels: y_true < 0
    - generate_low_valley_labels: y_true <= p10 or y_true <= threshold
    - generate_overestimate_low_labels: y_pred - y_true >= threshold
    - compute_low_valley_percentile: determine p10 threshold from history

All label functions are leakage-safe (use only current y_true/y_pred).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from extreme.negative_price.schema import (
    LOW_VALLEY_COL,
    NEGATIVE_PRICE_COL,
    NEGATIVE_PRICE_THRESHOLD,
    LOW_VALLEY_ABSOLUTE_THRESHOLD,
    LOW_VALLEY_PERCENTILE,
    OVERESTIMATE_LOW_COL,
    OVERESTIMATE_LOW_THRESHOLD,
)


def generate_negative_price_labels(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
) -> pd.Series:
    """Generate binary labels for negative price events.

    label = 1 if y_true < threshold (default 0.0), else 0.

    Args:
        df: DataFrame with y_true column.
        y_true_col: Name of the y_true column.

    Returns:
        pd.Series with 0/1 labels.
    """
    return (df[y_true_col] < NEGATIVE_PRICE_THRESHOLD).astype(int)


def compute_low_valley_percentile(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    percentile: float = LOW_VALLEY_PERCENTILE,
) -> float:
    """Compute the percentile threshold for low-valley labeling from history.

    Args:
        df: Historical DataFrame with y_true column.
        y_true_col: Name of the y_true column.
        percentile: Percentile to use (default 0.10).

    Returns:
        Threshold value at the given percentile.
    """
    vals = df[y_true_col].dropna().values
    if len(vals) == 0:
        return LOW_VALLEY_ABSOLUTE_THRESHOLD
    return float(np.percentile(vals, percentile * 100))


def generate_low_valley_labels(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    percentile_threshold: Optional[float] = None,
    absolute_threshold: float = LOW_VALLEY_ABSOLUTE_THRESHOLD,
) -> pd.Series:
    """Generate binary labels for low-valley events.

    label = 1 if y_true <= min(percentile_threshold, absolute_threshold), else 0.

    The effective threshold is the MORE CONSERVATIVE (lower) of the two,
    ensuring we flag the most extreme low-price events.

    Args:
        df: DataFrame with y_true column.
        y_true_col: Name of the y_true column.
        percentile_threshold: Value at the configured percentile.
                             If None, only uses absolute_threshold.
        absolute_threshold: Absolute price threshold (default 50).

    Returns:
        pd.Series with 0/1 labels.
    """
    effective = absolute_threshold
    if percentile_threshold is not None:
        effective = min(effective, percentile_threshold)

    return (df[y_true_col] <= effective).astype(int)


def generate_overestimate_low_labels(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    threshold: float = OVERESTIMATE_LOW_THRESHOLD,
) -> pd.Series:
    """Generate binary labels for overestimate_low events.

    label = 1 if y_pred - y_true >= threshold, else 0.
    Only computed where both y_true and y_pred are non-null.

    Note: This is a DIAGNOSTIC label, NOT a training label.
    It is used only for evaluation metrics, never as a prediction-time feature.

    Args:
        df: DataFrame with y_true and y_pred columns.
        y_true_col: Name of the y_true column.
        y_pred_col: Name of the y_pred column.
        threshold: Minimum overestimate amount.

    Returns:
        pd.Series with 0/1 labels (NaN where y_true or y_pred is missing).
    """
    mask = df[y_true_col].notna() & df[y_pred_col].notna()
    result = pd.Series(0, index=df.index, dtype=float)
    result.loc[mask] = ((df.loc[mask, y_pred_col] - df.loc[mask, y_true_col]) >= threshold).astype(float)
    result.loc[~mask] = np.nan
    return result


def add_all_labels(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Add all three label columns to a DataFrame in place.

    Args:
        df: DataFrame to label (modified in place).
        y_true_col: y_true column name.
        y_pred_col: y_pred column name.
        history_df: Optional historical data for percentile computation.

    Returns:
        DataFrame with added label columns.
    """
    df[NEGATIVE_PRICE_COL] = generate_negative_price_labels(df, y_true_col)

    percentile_val = None
    if history_df is not None:
        percentile_val = compute_low_valley_percentile(history_df, y_true_col)
    df[LOW_VALLEY_COL] = generate_low_valley_labels(
        df, y_true_col, percentile_threshold=percentile_val,
    )

    df[OVERESTIMATE_LOW_COL] = generate_overestimate_low_labels(
        df, y_true_col, y_pred_col,
    )

    return df
