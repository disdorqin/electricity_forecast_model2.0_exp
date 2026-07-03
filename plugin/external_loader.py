# -*- coding: utf-8 -*-
"""
external_loader.py — Load external-model prediction CSVs and normalise to the
canonical ``PredictionTableSpec``.

No model names are hardcoded anywhere in this module.  The caller provides
the column mapping and model-name metadata, making it fully generic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from plugin.schema import (
    DEFAULT_COLUMN_MAP,
    PredictionTableSpec,
    standardize_predictions,
)


@dataclass
class ExternalPredictionSource:
    """Describes a single external-model prediction source.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    column_mapping : dict[str, str] | None
        Mapping from CSV column names → canonical column names.
        ``None`` means the CSV already uses canonical names.
    model_name_override : str | None
        If set, overwrites the ``model_name`` column in the CSV with this
        value.  Useful when the CSV does not contain a model_name column
        or you want to rename the model.
    task_override : str | None
        If set, overwrites the ``task`` column (e.g. ``"dayahead"``).
    """
    path: str | Path
    column_mapping: Optional[dict[str, str]] = None
    model_name_override: Optional[str] = None
    task_override: Optional[str] = None


def load_external_predictions(
    source: ExternalPredictionSource,
    spec: Optional[PredictionTableSpec] = None,
    *,
    infer_missing_period: bool = True,
) -> pd.DataFrame:
    """Read an external-model prediction CSV and return a standardised DataFrame.

    Steps
    -----
    1. Read CSV from *source.path*.
    2. Rename columns according to *source.column_mapping* (or identity).
    3. Apply *source.model_name_override* / *source.task_override* if given.
    4. Fill missing ``period`` from ``hour_business`` if requested.
    5. Call ``standardize_predictions()`` for final normalisation.

    Parameters
    ----------
    source : ExternalPredictionSource
        Description of the CSV file and any overrides.
    spec : PredictionTableSpec or None
        Optional schema to validate against.
    infer_missing_period : bool
        If True (default), infer ``period`` from ``hour_business`` when the
        column is absent or empty.

    Returns
    -------
    pd.DataFrame
        Standardised prediction table conforming to ``PredictionTableSpec``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV cannot be standardised (missing columns, bad values, …).
    """
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"External prediction CSV not found: {path}")

    df = pd.read_csv(path)

    # ── Column mapping ──────────────────────────────────────────────
    mapping = source.column_mapping or dict(DEFAULT_COLUMN_MAP)
    # Only rename columns that actually exist in the CSV
    rename = {k: v for k, v in mapping.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)

    # ── Overrides ───────────────────────────────────────────────────
    if source.model_name_override is not None:
        df["model_name"] = source.model_name_override
    if source.task_override is not None:
        df["task"] = source.task_override

    # ── Fill defaults for optional columns ──────────────────────────
    if "task" not in df.columns:
        df["task"] = "dayahead"

    if "ds" not in df.columns and "target_day" in df.columns:
        df["ds"] = pd.to_datetime(df["target_day"])

    if "period" not in df.columns and infer_missing_period and "hour_business" in df.columns:
        from plugin.schema import _infer_period
        df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype(int)
        df["period"] = df["hour_business"].apply(_infer_period)

    # ── Standardise ─────────────────────────────────────────────────
    return standardize_predictions(df, spec=spec)
