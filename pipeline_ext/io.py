# -*- coding: utf-8 -*-
"""I/O layer for the plugin interface.

Loads external prediction CSVs, validates against the unified schema,
and enforces business-rule constraints.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_ext.schema import (
    check_leakage_safe,
    check_timestamp_uniqueness,
    check_uniqueness,
    check_y_pred_present,
    validate_schema,
)


def load_prediction_csv(
    path: str | Path,
    allow_long_format: bool = False,
) -> pd.DataFrame:
    """Load a prediction CSV and run all schema + business-rule checks.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    allow_long_format : bool
        If True, skip the (timestamp, model_name) uniqueness check,
        allowing multiple rows per timestamp (e.g. long-format ensemble
        prediction dumps).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with schema validated.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If schema validation or business-rule checks fail.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction CSV not found: {path}")

    df = pd.read_csv(path)

    # 1. Schema validation — required fields
    validate_schema(df)

    # 2. Leakage-safe check
    check_leakage_safe(df)

    # 3. y_pred completeness
    check_y_pred_present(df)

    # 4. Uniqueness of (business_day, hour_business)
    check_uniqueness(df)

    # 5. Timestamp uniqueness (skipped in long-format mode)
    if not allow_long_format:
        check_timestamp_uniqueness(df)

    return df


def validate_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an in-memory DataFrame against the prediction schema.

    Useful when the caller already has a DataFrame (e.g. from a
    PredictionProvider) and wants to run the same validation pipeline
    without going through CSV I/O.

    Returns the DataFrame unchanged on success.
    """
    validate_schema(df)
    check_leakage_safe(df)
    check_y_pred_present(df)
    check_uniqueness(df)
    return df


def load_and_validate(
    path: str | Path,
    allow_long_format: bool = False,
) -> pd.DataFrame:
    """Convenience alias for :func:`load_prediction_csv`."""
    return load_prediction_csv(path=path, allow_long_format=allow_long_format)
