# -*- coding: utf-8 -*-
"""
external_loader.py — Load external-model prediction CSVs and normalise to the
canonical ``PredictionTableSpec`` (P5 contract).

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
        If set, overwrites ``model_name`` in the CSV with this value.
        Useful when the CSV has no model_name column or you want to rename.
    source_file_tag : str | None
        If set, fills the ``source_file`` column with this value.
        Defaults to the CSV filename stem.
    prediction_mode_override : str | None
        If set, fills ``prediction_mode`` (``"dayahead"`` | ``"realtime"``).
    """
    path: str | Path
    column_mapping: Optional[dict[str, str]] = None
    model_name_override: Optional[str] = None
    source_file_tag: Optional[str] = None
    prediction_mode_override: Optional[str] = None


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
    3. Apply overrides (*model_name*, *source_file*, *prediction_mode*).
    4. Call ``standardize_predictions()`` for final normalisation.

    Parameters
    ----------
    source : ExternalPredictionSource
        Description of the CSV file and any overrides.
    spec : PredictionTableSpec or None
        Optional schema to validate against.
    infer_missing_period : bool
        If True (default), infer ``period`` from ``hour_business`` when absent.

    Returns
    -------
    pd.DataFrame
        Standardised prediction table conforming to ``PredictionTableSpec``.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV cannot be standardised.
    """
    path = Path(source.path)
    if not path.exists():
        raise FileNotFoundError(f"External prediction CSV not found: {path}")

    df = pd.read_csv(path)

    # ── Column mapping ──────────────────────────────────────────────
    mapping = source.column_mapping or dict(DEFAULT_COLUMN_MAP)
    rename = {k: v for k, v in mapping.items() if k in df.columns and k != v}
    if rename:
        df = df.rename(columns=rename)

    # ── Overrides ───────────────────────────────────────────────────
    if source.model_name_override is not None:
        df["model_name"] = source.model_name_override

    if source.source_file_tag is not None:
        df["source_file"] = source.source_file_tag
    elif "source_file" not in df.columns:
        df["source_file"] = path.stem

    if source.prediction_mode_override is not None:
        df["prediction_mode"] = source.prediction_mode_override

    # ── Default leakage_safe if missing ─────────────────────────────
    if "leakage_safe" not in df.columns:
        df["leakage_safe"] = "true"

    # ── Standardise ─────────────────────────────────────────────────
    return standardize_predictions(
        df,
        spec=spec,
        infer_period=infer_missing_period,
    )
