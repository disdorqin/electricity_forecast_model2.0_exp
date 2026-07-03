# -*- coding: utf-8 -*-
"""
schema.py — P5 external-model prediction schema contract.

Required columns (prediction-time safe):
    model_name, business_day, hour_business, timestamp,
    y_pred, source_file, prediction_mode, leakage_safe

Optional / evaluation-only:
    y_true, base_fused_pred, final_pred, high_spike_prob,
    negative_prob, low_valley_prob, module_name, task, period

Key rules:
    - ``leakage_safe`` must be the lowercase string ``"true"`` or it fails.
    - ``y_true`` is never required — external CSVs may lack it.
    - ``timestamp`` = ``{business_day}T{hour_business}:00:00`` can be inferred.
    - hour 24 maps to ``business_day - 1`` (e.g. ``2026-01-02 00:00`` → ``business_day=2026-01-01``).
    - Old aliases ``target_day`` → ``business_day`` and ``ds`` → ``timestamp`` are accepted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

# ── Canonical column names ──────────────────────────────────────────────

REQUIRED_COLUMNS: tuple[str, ...] = (
    "model_name",        # arbitrary string, NOT a hardcoded list
    "business_day",      # date string YYYY-MM-DD
    "hour_business",     # int 1-24
    "timestamp",         # full datetime
    "y_pred",            # predicted price
    "source_file",       # origin CSV / model run identifier
    "prediction_mode",   # "dayahead" | "realtime" | "external"
    "leakage_safe",      # must be lowercase "true"
)

OPTIONAL_COLUMNS: tuple[str, ...] = (
    "y_true",            # actual price (evaluation only, never required)
    "base_fused_pred",   # fused prediction before correction
    "final_pred",        # final prediction after all corrections
    "high_spike_prob",   # spike probability from high_spike module
    "negative_prob",     # negative price probability
    "low_valley_prob",   # low valley probability
    "module_name",       # correction / monitor module identifier
    "task",              # "dayahead" | "realtime" (legacy)
    "period",            # "1_8" | "9_16" | "17_24" (inferred if absent)
)

VALID_PREDICTION_MODES: frozenset[str] = frozenset({"dayahead", "realtime", "external"})

# ── Alias map for backward compatibility ────────────────────────────────

COLUMN_ALIASES: dict[str, str] = {
    "target_day": "business_day",
    "ds": "timestamp",
}
"""Old column names → new canonical names."""


@dataclass(frozen=True)
class PredictionTableSpec:
    """Canonical schema for external-model prediction tables.

    Parameters
    ----------
    required_columns : tuple[str, ...]
        Columns that MUST be present after normalisation.
    optional_columns : tuple[str, ...]
        Columns that MAY be present (evaluation extras).
    valid_prediction_modes : frozenset[str]
        Accepted values for ``prediction_mode``.
    allow_long_format : bool
        If False (default), each ``(model_name, business_day, hour_business)``
        tuple must be unique.  Set to True when the same model may have
        multiple rows per hour (e.g. ensemble members).
    """

    required_columns: tuple[str, ...] = field(default=REQUIRED_COLUMNS)
    optional_columns: tuple[str, ...] = field(default=OPTIONAL_COLUMNS)
    valid_prediction_modes: frozenset[str] = field(default=VALID_PREDICTION_MODES)
    allow_long_format: bool = False

    def validate(self, df: pd.DataFrame) -> None:
        """Validate that *df* conforms to this spec (in-place, no copy).

        Checks:
            1. All required columns present.
            2. ``leakage_safe`` is the string ``"true"``.
            3. ``prediction_mode`` values are valid.
            4. ``hour_business`` in 1-24.
            5. ``(model_name, business_day, hour_business)`` uniqueness.
        """
        # 1. Required columns
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"DataFrame missing required columns: {missing}"
            )

        # 2. leakage_safe must be true
        safe_vals = df["leakage_safe"].astype(str).str.strip().str.lower()
        bad_safe = safe_vals[~(safe_vals == "true")]
        if not bad_safe.empty:
            raise ValueError(
                f"leakage_safe must be 'true'; found {bad_safe.unique().tolist()!r}"
            )

        # 3. prediction_mode validation
        bad_modes = sorted(
            set(df["prediction_mode"].unique()) - self.valid_prediction_modes
        )
        if bad_modes:
            raise ValueError(
                f"Unsupported prediction_mode values: {bad_modes}"
            )

        # 4. hour_business range
        hb = pd.to_numeric(df["hour_business"], errors="coerce")
        if hb.isna().any() or ((hb < 1) | (hb > 24)).any():
            raise ValueError("hour_business must be int in 1-24")

        # 5. Uniqueness
        if not self.allow_long_format:
            dup_mask = df.duplicated(
                subset=["model_name", "business_day", "hour_business"],
                keep=False,
            )
            if dup_mask.any():
                dups = df.loc[dup_mask, ["model_name", "business_day", "hour_business"]].drop_duplicates()
                raise ValueError(
                    f"Duplicate (model_name, business_day, hour_business) rows "
                    f"found ({len(dups)} groups). Use allow_long_format=True to permit."
                )


# ── Column-name aliasing ────────────────────────────────────────────────

DEFAULT_COLUMN_MAP: dict[str, str] = {
    "model_name": "model_name",
    "business_day": "business_day",
    "hour_business": "hour_business",
    "timestamp": "timestamp",
    "y_pred": "y_pred",
    "source_file": "source_file",
    "prediction_mode": "prediction_mode",
    "leakage_safe": "leakage_safe",
}
"""Default identity mapping — assumes CSV already uses canonical names."""


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old alias columns (target_day, ds) to canonical names.

    Only renames columns that actually exist and that don't already have
    the canonical name.
    """
    rename = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename[old] = new
    if rename:
        df = df.rename(columns=rename)
    return df


# ── Timestamp ↔ (business_day, hour_business) helpers ───────────────────


def _construct_timestamp(
    business_day: str,
    hour_business: int,
) -> datetime:
    """Build a ``datetime`` from a business day and hour.

    Hour 24 is a special case — it represents 00:00 of the *next* calendar
    day but belongs to *business_day* D.
    """
    base = datetime.strptime(business_day, "%Y-%m-%d")
    if hour_business == 24:
        return base + timedelta(days=1)  # 00:00 next day
    return base.replace(hour=hour_business)


def _parse_timestamp(ts: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (business_day, hour_business) parsed from a timestamp Series.

    business_day = calendar date D
    hour_business = hour 1-24
    If the time is 00:00:00, it maps to business_day D-1, hour_business=24.
    """
    dt = pd.to_datetime(ts)

    # Detect 00:00 → belongs to previous business day, hour=24
    midnight_mask = (dt.dt.hour == 0) & (dt.dt.minute == 0) & (dt.dt.second == 0)
    business_day = dt.dt.strftime("%Y-%m-%d")
    hour_business = dt.dt.hour.astype(int)
    hour_business = hour_business.clip(lower=1)  # hour 0 → 1 (safety)

    # Re-map midnight rows: business_day -= 1 day, hour = 24
    if midnight_mask.any():
        prev_day = (dt[midnight_mask] - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
        business_day = business_day.astype(object)
        business_day.iloc[midnight_mask.values] = prev_day
        hour_business = hour_business.astype(object)
        hour_business.iloc[midnight_mask.values] = 24

    return business_day, hour_business.astype(int)


# ── Standardisation ─────────────────────────────────────────────────────


def _normalize_prediction_mode(value: str) -> str:
    text = str(value).strip().lower()
    mapping = {
        "dayahead": "dayahead",
        "da": "dayahead",
        "realtime": "realtime",
        "rt": "realtime",
        "external": "external",
    }
    if text not in mapping:
        raise ValueError(f"Unsupported prediction_mode value: {value!r}")
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
    *,
    infer_timestamp: bool = True,
    infer_period: bool = True,
    fill_source_file: Optional[str] = None,
    fill_prediction_mode: Optional[str] = None,
) -> pd.DataFrame:
    """Normalise a prediction DataFrame to the canonical schema.

    Steps
    -----
    1. Apply column aliases (``target_day`` → ``business_day``, ``ds`` → ``timestamp``).
    2. Verify all required columns are present.
    3. Infer ``timestamp`` from ``business_day`` + ``hour_business`` if missing.
    4. Infer ``business_day`` + ``hour_business`` from ``timestamp`` if missing.
    5. Normalise ``prediction_mode``.
    6. Normalise ``leakage_safe`` to lowercase ``"true"``.
    7. Cast ``y_pred`` (and ``y_true`` if present) to numeric.
    8. Infer ``period`` from ``hour_business`` if missing.
    9. Run ``spec.validate()``.

    Parameters
    ----------
    df : pd.DataFrame
        Raw prediction DataFrame (column names may use old aliases).
    spec : PredictionTableSpec or None
        Schema to validate against.  Defaults to ``PredictionTableSpec()``.
    infer_timestamp : bool
        If True and ``timestamp`` is missing, build from ``business_day`` +
        ``hour_business``.
    infer_period : bool
        If True and ``period`` is missing, infer from ``hour_business``.
    fill_source_file : str or None
        If set, fill ``source_file`` column with this value when missing.
    fill_prediction_mode : str or None
        If set, fill ``prediction_mode`` column with this value when missing.

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

    # 1. Apply column aliases
    out = apply_column_aliases(out)

    # 2. Fill optional fields
    if fill_source_file is not None and "source_file" not in out.columns:
        out["source_file"] = fill_source_file
    if fill_prediction_mode is not None and "prediction_mode" not in out.columns:
        out["prediction_mode"] = fill_prediction_mode

    # 3. Infer timestamp from (business_day, hour_business)
    if infer_timestamp and "timestamp" not in out.columns:
        if "business_day" in out.columns and "hour_business" in out.columns:
            out["hour_business"] = pd.to_numeric(out["hour_business"], errors="coerce").fillna(1).astype(int)
            out["timestamp"] = out.apply(
                lambda r: _construct_timestamp(
                    str(r["business_day"]), int(r["hour_business"])
                ),
                axis=1,
            )
            out["timestamp"] = pd.to_datetime(out["timestamp"])

    # 4. Infer (business_day, hour_business) from timestamp
    if "timestamp" in out.columns and ("business_day" not in out.columns or "hour_business" not in out.columns):
        bd, hb = _parse_timestamp(out["timestamp"])
        out["business_day"] = bd
        out["hour_business"] = hb

    # 4b. Normalize hour_business == 0 from timestamp (midnight convention)
    if "hour_business" in out.columns and "business_day" in out.columns and "timestamp" in out.columns:
        zero_mask = pd.to_numeric(out["hour_business"], errors="coerce") == 0
        if zero_mask.any():
            ts_at_zero = pd.to_datetime(out.loc[zero_mask, "timestamp"])
            bd, hb = _parse_timestamp(ts_at_zero)
            out.loc[zero_mask, "business_day"] = bd
            out.loc[zero_mask, "hour_business"] = hb

    # 5. Normalise prediction_mode
    if "prediction_mode" in out.columns:
        out["prediction_mode"] = out["prediction_mode"].map(_normalize_prediction_mode)
    else:
        # Default to "external" for external-model CSVs without this column
        out["prediction_mode"] = "external"

    # 6. Normalise leakage_safe
    if "leakage_safe" in out.columns:
        out["leakage_safe"] = (
            out["leakage_safe"].astype(str).str.strip().str.lower()
        )
    else:
        out["leakage_safe"] = "true"

    # 7. Numeric coercion (only if column exists — validation catches missing later)
    if "hour_business" in out.columns:
        out["hour_business"] = pd.to_numeric(
            out["hour_business"], errors="coerce"
        ).fillna(1).astype(int)
    if "y_pred" in out.columns:
        out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    # 8. Infer period
    if infer_period and "period" not in out.columns:
        if "hour_business" in out.columns:
            out["period"] = out["hour_business"].apply(_infer_period)
    elif "period" in out.columns:
        out["period"] = out["period"].astype(str).str.strip()
        if "hour_business" in out.columns:
            out["period"] = out.apply(
                lambda r: _infer_period(r["hour_business"])
                if not r["period"] or r["period"] in ("nan", "")
                else r["period"],
                axis=1,
            )

    # 9. Validate
    spec.validate(out)

    return out
