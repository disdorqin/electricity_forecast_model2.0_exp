# -*- coding: utf-8 -*-
"""
schema.py — Unified prediction schema and standardisation function.

Extends the concepts in ``fusion/contracts.py`` without importing or modifying it.
All external-model predictions must be mappable to this schema so that downstream
pipeline stages (correction, monitoring, fusion) can consume them without
knowing the originating model name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ── Canonical column names ──────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "task",          # "dayahead" | "realtime"
    "model_name",    # arbitrary string, NOT a hardcoded list
    "target_day",    # date string / Timestamp
    "ds",            # full datetime
    "period",        # "1_8" | "9_16" | "17_24"
    "hour_business", # int 1-24
    "y_true",        # actual price
    "y_pred",        # predicted price
)

VALID_TASKS: frozenset[str] = frozenset({"dayahead", "realtime"})
VALID_PERIODS: frozenset[str] = frozenset({"1_8", "9_16", "17_24"})


@dataclass(frozen=True)
class PredictionTableSpec:
    """Describes the canonical schema for a prediction table.

    Every external-model CSV, after column mapping and normalisation, must
    conform to this spec before it enters the pipeline.
    """

    required_columns: tuple[str, ...] = field(default=REQUIRED_COLUMNS)
    valid_tasks: frozenset[str] = field(default=VALID_TASKS)
    valid_periods: frozenset[str] = field(default=VALID_PERIODS)

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that *df* conforms to this spec (in-place check, no copy)."""
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        bad_tasks = sorted(set(df["task"].unique()) - self.valid_tasks)
        if bad_tasks:
            raise ValueError(f"Unsupported task values: {bad_tasks}")

        bad_periods = sorted(set(df["period"].unique()) - self.valid_periods)
        if bad_periods:
            raise ValueError(f"Unsupported period values: {bad_periods}")


# ── Default column-name map ─────────────────────────────────────────────

DEFAULT_COLUMN_MAP: dict[str, str] = {
    "task": "task",
    "model_name": "model_name",
    "target_day": "target_day",
    "ds": "ds",
    "period": "period",
    "hour_business": "hour_business",
    "y_true": "y_true",
    "y_pred": "y_pred",
}
"""Default identity mapping — assumes CSV already uses canonical column names."""


# ── Standardisation ─────────────────────────────────────────────────────


def _normalize_task(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "dayahead": "dayahead",
        "da": "dayahead",
        "realtime": "realtime",
        "rt": "realtime",
    }
    if text not in mapping:
        raise ValueError(f"Unsupported task value: {value!r}")
    return mapping[text]


def _infer_period(hour_business: int) -> str:
    hour = int(hour_business)
    if 1 <= hour <= 8:
        return "1_8"
    if 9 <= hour <= 16:
        return "9_16"
    if 17 <= hour <= 24:
        return "17_24"
    raise ValueError(f"hour_business out of range: {hour_business}")


def standardize_predictions(
    df: pd.DataFrame,
    spec: Optional[PredictionTableSpec] = None,
) -> pd.DataFrame:
    """Normalise a prediction DataFrame to the canonical schema.

    Steps
    -----
    1. Verify all required columns are present.
    2. Normalise ``task`` (dayahead / realtime).
    3. Parse ``target_day`` and ``ds`` as datetimes.
    4. Cast ``hour_business`` to int.
    5. Infer ``period`` from ``hour_business`` when missing / invalid.
    6. Coerce ``y_true`` / ``y_pred`` to numeric.
    7. Validate task and period values against the spec.

    Parameters
    ----------
    df : pd.DataFrame
        Raw prediction DataFrame (column names must match canonical names
        *after* external mapping has been applied).
    spec : PredictionTableSpec or None
        Schema to validate against.  Defaults to ``PredictionTableSpec()``.

    Returns
    -------
    pd.DataFrame
        Normalised copy of the input.

    Raises
    ------
    ValueError
        If required columns are missing or values are out of range.
    """
    spec = spec or PredictionTableSpec()
    out = df.copy()

    # 1. Required columns
    missing = [c for c in spec.required_columns if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 2. Task normalisation
    out["task"] = out["task"].map(_normalize_task)

    # 3. Datetime parsing
    out["target_day"] = pd.to_datetime(out["target_day"]).dt.strftime("%Y-%m-%d")
    out["ds"] = pd.to_datetime(out["ds"])

    # 4. Business hour
    out["hour_business"] = out["hour_business"].astype(int)

    # 5. Period
    out["period"] = out["period"].astype(str).str.strip()
    out["period"] = out.apply(
        lambda row: _infer_period(row["hour_business"])
        if not row["period"] or row["period"] in ("nan", "")
        else row["period"],
        axis=1,
    )

    # 6. Numeric coercion
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    # 7. Validate
    bad_tasks = sorted(set(out["task"].unique()) - spec.valid_tasks)
    if bad_tasks:
        raise ValueError(f"Unsupported task labels after normalisation: {bad_tasks}")

    bad_periods = sorted(set(out["period"].unique()) - spec.valid_periods)
    if bad_periods:
        raise ValueError(f"Unsupported period labels after normalisation: {bad_periods}")

    if out[["y_true", "y_pred"]].isna().any().any():
        raise ValueError("Predictions contain NaN in y_true or y_pred")

    return out
