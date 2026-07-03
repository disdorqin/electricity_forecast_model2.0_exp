"""Adapter for loading, normalizing, and validating intraday correction packs.

Reads the pack exported by the deep branch (Phase 10 handoff contract)
and converts it to the mainline standardized schema.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from corrections.intraday_tracker.schema import (
    INTRADAY_PACK_EVAL_FIELDS,
    INTRADAY_PACK_INPUT_FIELDS,
    MAINLINE_OUTPUT_FIELDS,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def load_intraday_pack(path: str) -> pd.DataFrame:
    """Load intraday correction pack from CSV.

    Returns empty DataFrame with correct columns if file not found.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Intraday pack not found: %s", path)
        return pd.DataFrame(columns=INTRADAY_PACK_INPUT_FIELDS)

    try:
        df = pd.read_csv(str(p), encoding="utf-8-sig")
    except Exception as e:
        logger.error("Failed to read intraday pack: %s", e)
        return pd.DataFrame(columns=INTRADAY_PACK_INPUT_FIELDS)

    logger.info("Loaded intraday pack: %d rows from %s", len(df), path)
    return df


def normalize_intraday_pack(df: pd.DataFrame, source_pack_path: str = "") -> pd.DataFrame:
    """Normalize intraday pack to mainline output schema.

    Ensures all required fields exist, fills missing with safe defaults.
    """
    if len(df) == 0:
        out = pd.DataFrame(columns=MAINLINE_OUTPUT_FIELDS)
        out["source_pack_path"] = source_pack_path
        return out

    out = pd.DataFrame()

    # Direct mapping
    for col in ["business_day", "cutoff_hour", "target_hour", "ds",
                "base_model_name", "base_pred", "intraday_corrected_pred",
                "intraday_final_correction", "intraday_confidence",
                "policy_decision", "fusion_weight", "shadow_only_flag",
                "guardrail_reason", "mode"]:
        if col in df.columns:
            out[col] = df[col]
        else:
            out[col] = _default_for(col)

    # hour_business = target_hour (same thing in 9_16 segment)
    out["hour_business"] = df.get("target_hour", df.get("hour_business", 0))

    # source_pack_path
    out["source_pack_path"] = source_pack_path

    # Ensure types
    out["business_day"] = pd.to_datetime(out["business_day"], errors="coerce")
    out["cutoff_hour"] = pd.to_numeric(out["cutoff_hour"], errors="coerce").fillna(0).astype(int)
    out["target_hour"] = pd.to_numeric(out["target_hour"], errors="coerce").fillna(0).astype(int)
    out["hour_business"] = pd.to_numeric(out["hour_business"], errors="coerce").fillna(0).astype(int)
    out["intraday_confidence"] = pd.to_numeric(out["intraday_confidence"], errors="coerce").fillna(0.0)
    out["fusion_weight"] = pd.to_numeric(out["fusion_weight"], errors="coerce").fillna(0.0)
    out["shadow_only_flag"] = out["shadow_only_flag"].fillna(False)

    return out


def validate_intraday_pack(
    df: pd.DataFrame,
    mode: str = "online",
) -> ValidationResult:
    """Validate intraday pack against mainline rules.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized intraday pack.
    mode : str
        "online" or "eval". Online mode must NOT contain y_true.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult(valid=True)

    if len(df) == 0:
        result.add_warning("Empty intraday pack")
        return result

    # Rule 1: mode must be INTRADAY
    if "mode" in df.columns:
        non_intraday = df[df["mode"] != "INTRADAY"]
        if len(non_intraday) > 0:
            result.add_error(
                f"Found {len(non_intraday)} rows with mode != INTRADAY"
            )

    # Rule 2: target_hour must > cutoff_hour
    if "target_hour" in df.columns and "cutoff_hour" in df.columns:
        bad = df[df["target_hour"] <= df["cutoff_hour"]]
        if len(bad) > 0:
            result.add_error(
                f"Found {len(bad)} rows where target_hour <= cutoff_hour"
            )

    # Rule 3: cutoff_hour >= 10 to allow reading
    if "cutoff_hour" in df.columns:
        low_cutoff = df[df["cutoff_hour"] < 10]
        if len(low_cutoff) > 0:
            result.add_warning(
                f"Found {len(low_cutoff)} rows with cutoff_hour < 10 (should be shadow/disabled)"
            )

    # Rule 4: cutoff_hour < 12 → shadow_only or disabled
    if "cutoff_hour" in df.columns:
        mid_cutoff = df[(df["cutoff_hour"] >= 10) & (df["cutoff_hour"] < 12)]
        if len(mid_cutoff) > 0:
            non_shadow = mid_cutoff[
                (mid_cutoff.get("policy_decision", pd.Series(dtype=str)) != "SHADOW_ONLY")
                & (mid_cutoff.get("policy_decision", pd.Series(dtype=str)) != "DISABLED")
            ]
            if len(non_shadow) > 0:
                result.add_warning(
                    f"Found {len(non_shadow)} rows with cutoff 10-11 not shadow/disabled"
                )

    # Rule 5: fusion_weight in [0, 0.3]
    if "fusion_weight" in df.columns:
        fw = pd.to_numeric(df["fusion_weight"], errors="coerce").fillna(0)
        bad_fw = fw[(fw < 0) | (fw > 0.3)]
        if len(bad_fw) > 0:
            result.add_error(
                f"Found {len(bad_fw)} rows with fusion_weight outside [0, 0.3]"
            )

    # Rule 6: intraday_confidence in [0, 1]
    if "intraday_confidence" in df.columns:
        conf = pd.to_numeric(df["intraday_confidence"], errors="coerce").fillna(0)
        bad_conf = conf[(conf < 0) | (conf > 1)]
        if len(bad_conf) > 0:
            result.add_error(
                f"Found {len(bad_conf)} rows with intraday_confidence outside [0, 1]"
            )

    # Rule 7: online mode must NOT contain y_true
    if mode == "online" and "y_true" in df.columns:
        has_ytrue = df["y_true"].notna().sum()
        if has_ytrue > 0:
            result.add_error(
                f"Online pack contains y_true ({has_ytrue} non-null rows). "
                "y_true must be stripped for online mode."
            )

    # Rule 8: business_day + target_hour must not duplicate
    if "business_day" in df.columns and "target_hour" in df.columns:
        dupes = df.duplicated(subset=["business_day", "target_hour"], keep=False)
        if dupes.any():
            n_dupes = dupes.sum()
            result.add_error(
                f"Found {n_dupes} duplicate (business_day, target_hour) rows"
            )

    return result


def _default_for(col: str):
    """Return safe default for a missing column."""
    defaults = {
        "business_day": pd.NaT,
        "cutoff_hour": 0,
        "target_hour": 0,
        "hour_business": 0,
        "ds": pd.NaT,
        "mode": "INTRADAY",
        "base_model_name": "sgdfnet",
        "base_pred": 0.0,
        "intraday_corrected_pred": 0.0,
        "intraday_final_correction": 0.0,
        "intraday_confidence": 0.0,
        "policy_decision": "DISABLED",
        "fusion_weight": 0.0,
        "shadow_only_flag": False,
        "guardrail_reason": "pack_missing",
        "source_pack_path": "",
    }
    return defaults.get(col, None)
