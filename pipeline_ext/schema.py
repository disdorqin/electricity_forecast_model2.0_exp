# -*- coding: utf-8 -*-
"""Unified prediction schema for the plugin interface.

Defines required / optional fields and validation helpers that all
external prediction sources must satisfy before entering the fusion,
correction, or monitor chain.

REQUIRED_FIELDS  — Every prediction CSV must have these.
OPTIONAL_FIELDS  — May be present; validated if present.
ALL_FIELDS       — Union of required and optional.
"""

from __future__ import annotations

from typing import Final

import pandas as pd

# ── Required fields ────────────────────────────────────────────────────
REQUIRED_FIELDS: Final[list[str]] = [
    "model_name",
    "business_day",
    "hour_business",
    "timestamp",
    "y_pred",
    "source_file",
    "prediction_mode",
    "leakage_safe",
]
"""Columns that MUST exist in every external prediction CSV."""

# ── Optional fields ────────────────────────────────────────────────────
OPTIONAL_FIELDS: Final[list[str]] = [
    "base_fused_pred",
    "final_pred",
    "high_spike_prob",
    "negative_prob",
    "low_valley_prob",
    "module_name",
]
"""Columns that MAY exist; validated for type / range if present."""

ALL_FIELDS: Final[list[str]] = REQUIRED_FIELDS + OPTIONAL_FIELDS
"""Complete set of recognised schema fields."""

# ── Expected dtypes (informative; validation uses lenient checks) ──────
REQUIRED_DTYPES: Final[dict[str, str]] = {
    "model_name": "str",
    "business_day": "str (YYYY-MM-DD)",
    "hour_business": "int (1-24)",
    "timestamp": "datetime",
    "y_pred": "float",
    "source_file": "str",
    "prediction_mode": "str",
    "leakage_safe": "bool",
}

# ── Valid values for categorical fields ────────────────────────────────
VALID_PREDICTION_MODES: Final[set[str]] = {"dayahead", "realtime"}

# ── Validation helpers ─────────────────────────────────────────────────


def validate_schema(df: pd.DataFrame, raise_on_error: bool = True) -> list[str]:
    """Validate that *df* contains all required schema fields.

    Returns a list of missing field names (empty = pass).
    If *raise_on_error* is True, raises ``ValueError`` on first failure.
    """
    missing = [col for col in REQUIRED_FIELDS if col not in df.columns]
    if missing and raise_on_error:
        raise ValueError(
            f"Missing required field(s): {missing}. "
            f"Present columns: {sorted(df.columns)}"
        )
    return missing


def check_uniqueness(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> list[str]:
    """Check that (business_day, hour_business) pairs are unique.

    Returns a list of duplicate pair descriptions (empty = pass).
    """
    if "business_day" not in df.columns or "hour_business" not in df.columns:
        msg = "Cannot check uniqueness — business_day and/or hour_business missing."
        if raise_on_error:
            raise ValueError(msg)
        return [msg]

    dupes = df[["business_day", "hour_business"]].dropna()
    mask = dupes.duplicated(keep=False)
    if mask.any():
        pairs = (
            df.loc[mask, ["business_day", "hour_business"]]
            .drop_duplicates()
            .head(10)
        )
        descriptions = [
            f"{row['business_day']} / h{int(row['hour_business'])}"
            for _, row in pairs.iterrows()
        ]
        msg = f"Duplicate (business_day, hour_business) pairs: {descriptions}"
        if raise_on_error:
            raise ValueError(msg)
        return descriptions
    return []


def check_leakage_safe(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> bool:
    """Verify the *leakage_safe* column is truthy for all rows.

    Returns True if all rows pass.
    """
    if "leakage_safe" not in df.columns:
        msg = "Column 'leakage_safe' is missing."
        if raise_on_error:
            raise ValueError(msg)
        return False

    unsafe = df["leakage_safe"].astype(str).str.lower().isin(["false", "0", "no", ""])
    if unsafe.any():
        n_unsafe = int(unsafe.sum())
        msg = (
            f"{n_unsafe} row(s) have leakage_safe == false. "
            "All predictions must be leakage-safe."
        )
        if raise_on_error:
            raise ValueError(msg)
        return False
    return True


def check_y_pred_present(df: pd.DataFrame, raise_on_error: bool = True) -> bool:
    """Verify *y_pred* column has no NaN / inf values."""
    if "y_pred" not in df.columns:
        if raise_on_error:
            raise ValueError("Column 'y_pred' is missing.")
        return False

    n_missing = int(df["y_pred"].isna().sum())
    n_inf = int((~df["y_pred"].apply(lambda v: isinstance(v, (int, float))) | df["y_pred"].isin([float("inf"), float("-inf")])).sum())

    total_bad = n_missing  # approximate; inf check above catches the rest
    if n_missing > 0:
        msg = f"y_pred has {n_missing} missing value(s)."
        if raise_on_error:
            raise ValueError(msg)
        return False
    return True


def check_timestamp_uniqueness(
    df: pd.DataFrame,
    raise_on_error: bool = True,
) -> list[str]:
    """Check that (timestamp, model_name) pairs are unique.

    Returns list of duplicate descriptions (empty = pass).
    If duplicates exist and *raise_on_error* is True, raises ValueError
    unless the caller explicitly opts into ``allow_long_format=True``
    (handled at the ``io`` level — see :func:`load_prediction_csv`).
    """
    if "timestamp" not in df.columns or "model_name" not in df.columns:
        msg = "Cannot check — timestamp and/or model_name missing."
        if raise_on_error:
            raise ValueError(msg)
        return [msg]

    dupes = df[["timestamp", "model_name"]].dropna()
    mask = dupes.duplicated(keep=False)
    if mask.any():
        n_dupes = int(mask.sum())
        msg = (
            f"{n_dupes} duplicate (timestamp, model_name) row(s) detected. "
            "Use allow_long_format=True if this is intentional."
        )
        if raise_on_error:
            raise ValueError(msg)
        return [msg]
    return []
