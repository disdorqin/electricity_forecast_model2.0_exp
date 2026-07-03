"""Apply intraday tracker correction to main pipeline forecast — Phase 12.

Phase 12 fix: prediction column propagation.
When correction is applied, ALL existing prediction columns (rt_pred, y_fused, y_pred)
must be synchronously updated so that downstream steps (classifier, final output)
see the corrected values.

Implements the fusion logic:
- shadow mode: record shadow prediction but don't change any prediction column
- low_weight / high_weight mode: blend base prediction with corrected prediction,
  and update ALL existing prediction columns
- off / disabled: no change
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from corrections.intraday_tracker.policy import (
    IntradayTrackerMainlineConfig,
    apply_mainline_intraday_policy,
)

logger = logging.getLogger(__name__)

# All prediction columns that may need synchronous update
PREDICTION_COLUMNS = ["rt_pred", "y_fused", "y_pred"]


def apply_intraday_tracker_correction(
    base_forecast_df: pd.DataFrame,
    intraday_pack_df: pd.DataFrame,
    mode: str = "shadow",
    config: Optional[IntradayTrackerMainlineConfig] = None,
    prediction_mode: str = "INTRADAY",
) -> Tuple[pd.DataFrame, dict]:
    """Apply intraday tracker correction to base forecast.

    Phase 12: When correction is applied, ALL existing prediction columns
    (rt_pred, y_fused, y_pred) are synchronously updated.

    Parameters
    ----------
    base_forecast_df : pd.DataFrame
        Main pipeline forecast. Must have at least one of:
        rt_pred, y_fused, y_pred.
    intraday_pack_df : pd.DataFrame
        Normalized + validated intraday pack.
    mode : str
        "shadow" | "low_weight" | "high_weight" | "off"
    config : IntradayTrackerMainlineConfig
        Policy configuration.
    prediction_mode : str
        "INTRADAY" or "FULL_DAY".

    Returns
    -------
    tuple of (updated_df, stats_dict)
    """
    if config is None:
        config = IntradayTrackerMainlineConfig()

    stats = {
        "intraday_enabled": mode != "off",
        "intraday_mode": mode,
        "prediction_mode": prediction_mode,
        "pack_rows": len(intraday_pack_df),
        "matched_rows": 0,
        "applied_rows": 0,
        "shadow_rows": 0,
        "disabled_rows": 0,
        "avg_fusion_weight": 0.0,
        "avg_confidence": 0.0,
        "policy_counts": {},
        "guardrail_counts": {},
        "fallback_reason": None,
        "safe_fallback": True,
    }

    result = base_forecast_df.copy()

    # Add intraday audit columns
    result["intraday_available"] = False
    result["intraday_applied"] = False
    result["intraday_policy_decision"] = "DISABLED"
    result["intraday_fusion_weight"] = 0.0
    result["intraday_confidence"] = 0.0
    result["intraday_guardrail_reason"] = ""
    result["intraday_shadow_pred"] = np.nan
    result["intraday_shadow_delta"] = np.nan
    result["intraday_prediction_column_updated"] = False
    result["intraday_updated_columns"] = ""

    # Detect which prediction columns exist
    existing_pred_cols = [c for c in PREDICTION_COLUMNS if c in result.columns]

    if not existing_pred_cols:
        stats["fallback_reason"] = "no_base_pred_column"
        logger.warning("Cannot find any prediction column (rt_pred/y_fused/y_pred) in forecast df")
        result["rt_pred_before_intraday"] = np.nan
        result["rt_pred_after_intraday"] = np.nan
        result["y_fused_before_intraday"] = np.nan if "y_fused" not in result.columns else result["y_fused"]
        result["y_fused_after_intraday"] = np.nan if "y_fused" not in result.columns else result["y_fused"]
        return result, stats

    # Use the first existing prediction column as the "base" for reading
    base_pred_col = existing_pred_cols[0]

    # Record before-values for ALL prediction columns
    result["rt_pred_before_intraday"] = result[base_pred_col].copy()
    if "y_fused" in result.columns:
        result["y_fused_before_intraday"] = result["y_fused"].copy()
    if "y_pred" in result.columns:
        result["y_pred_before_intraday"] = result["y_pred"].copy()

    # Initialize after-values as copies of before
    result["rt_pred_after_intraday"] = result["rt_pred_before_intraday"].copy()
    if "y_fused" in result.columns:
        result["y_fused_after_intraday"] = result["y_fused_before_intraday"].copy()
    if "y_pred" in result.columns:
        result["y_pred_after_intraday"] = result["y_pred_before_intraday"].copy()

    # If mode is off or prediction_mode is FULL_DAY, return early
    if mode == "off" or prediction_mode != "INTRADAY":
        if prediction_mode != "INTRADAY":
            stats["fallback_reason"] = f"prediction_mode={prediction_mode}"
            logger.info("Intraday tracker disabled: prediction_mode=%s", prediction_mode)
        return result, stats

    if len(intraday_pack_df) == 0:
        stats["fallback_reason"] = "empty_pack"
        logger.info("Intraday tracker: empty pack, safe fallback")
        return result, stats

    # Apply mainline policy (second layer of defense)
    pack = apply_mainline_intraday_policy(intraday_pack_df, config, prediction_mode)

    if "policy_decision" in pack.columns:
        stats["policy_counts"] = pack["policy_decision"].value_counts().to_dict()

    # Prepare pack for merge
    merge_cols = ["business_day", "target_hour", "intraday_corrected_pred",
                  "intraday_final_correction", "intraday_confidence",
                  "policy_decision", "fusion_weight", "shadow_only_flag",
                  "guardrail_reason", "cutoff_hour"]
    available_cols = [c for c in merge_cols if c in pack.columns]
    pack_merge = pack[available_cols].copy()
    pack_merge["business_day"] = pd.to_datetime(pack_merge["business_day"], errors="coerce")

    # Detect hour column in result
    result_hour_col = "hour_business" if "hour_business" in result.columns else "target_hour"
    result["business_day"] = pd.to_datetime(
        result.get("business_day", result.get("target_day", "")), errors="coerce"
    )

    # Merge
    merged = result.merge(
        pack_merge,
        left_on=["business_day", result_hour_col],
        right_on=["business_day", "target_hour"],
        how="left",
        suffixes=("", "_intraday"),
    )

    has_intraday = merged["intraday_corrected_pred"].notna()
    merged["intraday_available"] = has_intraday
    stats["matched_rows"] = int(has_intraday.sum())

    # Apply corrections row by row
    for idx in merged.index:
        if not has_intraday.loc[idx]:
            merged.loc[idx, "intraday_applied"] = False
            merged.loc[idx, "intraday_policy_decision"] = "DISABLED"
            stats["disabled_rows"] += 1
            continue

        policy = str(merged.loc[idx, "policy_decision"])
        fw = float(merged.loc[idx, "fusion_weight"]) if pd.notna(merged.loc[idx, "fusion_weight"]) else 0.0
        conf = float(merged.loc[idx, "intraday_confidence"]) if pd.notna(merged.loc[idx, "intraday_confidence"]) else 0.0
        corrected_pred = float(merged.loc[idx, "intraday_corrected_pred"])
        base_pred = float(merged.loc[idx, base_pred_col])
        shadow_flag = bool(merged.loc[idx, "shadow_only_flag"]) if "shadow_only_flag" in merged.columns else False

        merged.loc[idx, "intraday_policy_decision"] = policy
        merged.loc[idx, "intraday_fusion_weight"] = fw
        merged.loc[idx, "intraday_confidence"] = conf
        merged.loc[idx, "intraday_shadow_pred"] = corrected_pred
        merged.loc[idx, "intraday_shadow_delta"] = corrected_pred - base_pred

        if mode == "shadow":
            # Shadow mode: DON'T update any prediction column
            merged.loc[idx, "intraday_applied"] = False
            merged.loc[idx, "rt_pred_after_intraday"] = base_pred
            if "y_fused" in existing_pred_cols:
                merged.loc[idx, "y_fused_after_intraday"] = merged.loc[idx, "y_fused_before_intraday"]
            if "y_pred" in existing_pred_cols:
                merged.loc[idx, "y_pred_after_intraday"] = merged.loc[idx, "y_pred_before_intraday"]
            merged.loc[idx, "intraday_prediction_column_updated"] = False
            merged.loc[idx, "intraday_updated_columns"] = ""
            stats["shadow_rows"] += 1

        elif mode in ("low_weight", "high_weight"):
            if policy in ("DISABLED", "SHADOW_ONLY") or shadow_flag:
                # Policy says don't apply — no column update
                merged.loc[idx, "intraday_applied"] = False
                merged.loc[idx, "rt_pred_after_intraday"] = base_pred
                if "y_fused" in existing_pred_cols:
                    merged.loc[idx, "y_fused_after_intraday"] = merged.loc[idx, "y_fused_before_intraday"]
                if "y_pred" in existing_pred_cols:
                    merged.loc[idx, "y_pred_after_intraday"] = merged.loc[idx, "y_pred_before_intraday"]
                merged.loc[idx, "intraday_prediction_column_updated"] = False
                merged.loc[idx, "intraday_updated_columns"] = ""
                if policy == "DISABLED":
                    stats["disabled_rows"] += 1
                else:
                    stats["shadow_rows"] += 1
            else:
                # Apply fusion: final = (1-w)*base + w*corrected
                final_pred = (1.0 - fw) * base_pred + fw * corrected_pred

                # Phase 12: Update ALL existing prediction columns
                updated_cols = []
                for col in existing_pred_cols:
                    merged.loc[idx, col] = final_pred
                    updated_cols.append(col)

                # Update after-values
                merged.loc[idx, "rt_pred_after_intraday"] = final_pred
                if "y_fused" in existing_pred_cols:
                    merged.loc[idx, "y_fused_after_intraday"] = final_pred
                if "y_pred" in existing_pred_cols:
                    merged.loc[idx, "y_pred_after_intraday"] = final_pred

                merged.loc[idx, "intraday_prediction_column_updated"] = True
                merged.loc[idx, "intraday_updated_columns"] = ",".join(updated_cols)
                merged.loc[idx, "intraday_applied"] = True
                stats["applied_rows"] += 1
        else:
            # Unknown mode, treat as shadow
            merged.loc[idx, "intraday_applied"] = False
            merged.loc[idx, "rt_pred_after_intraday"] = base_pred
            merged.loc[idx, "intraday_prediction_column_updated"] = False
            merged.loc[idx, "intraday_updated_columns"] = ""
            stats["shadow_rows"] += 1

    # Compute averages
    if stats["matched_rows"] > 0:
        matched = merged[has_intraday]
        stats["avg_fusion_weight"] = float(matched["intraday_fusion_weight"].mean())
        stats["avg_confidence"] = float(matched["intraday_confidence"].mean())

    # Guardrail counts
    if "guardrail_reason" in merged.columns:
        gr = merged.loc[has_intraday, "guardrail_reason"]
        if gr.notna().any():
            stats["guardrail_counts"] = gr[gr.notna() & (gr != "")].value_counts().to_dict()

    return merged, stats
