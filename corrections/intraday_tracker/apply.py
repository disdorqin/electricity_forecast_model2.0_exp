"""Apply intraday tracker correction to main pipeline forecast — Phase 11.

Implements the fusion logic:
- shadow mode: record shadow prediction but don't change final
- low_weight / high_weight mode: blend base prediction with corrected prediction
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


def apply_intraday_tracker_correction(
    base_forecast_df: pd.DataFrame,
    intraday_pack_df: pd.DataFrame,
    mode: str = "shadow",
    config: Optional[IntradayTrackerMainlineConfig] = None,
    prediction_mode: str = "INTRADAY",
) -> Tuple[pd.DataFrame, dict]:
    """Apply intraday tracker correction to base forecast.

    Parameters
    ----------
    base_forecast_df : pd.DataFrame
        Main pipeline forecast with columns:
        business_day, hour_business, ds, rt_pred, model_name
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

    # Detect base prediction column
    base_pred_col = None
    for col in ["rt_pred", "y_fused", "rt_pred_final"]:
        if col in result.columns:
            base_pred_col = col
            break

    if base_pred_col is None:
        stats["fallback_reason"] = "no_base_pred_column"
        logger.warning("Cannot find base prediction column in forecast df")
        result["rt_pred_before_intraday"] = np.nan
        result["rt_pred_after_intraday"] = np.nan
        return result, stats

    result["rt_pred_before_intraday"] = result[base_pred_col].copy()
    result["rt_pred_after_intraday"] = result["rt_pred_before_intraday"].copy()

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
            # Shadow mode: don't change final prediction
            merged.loc[idx, "intraday_applied"] = False
            merged.loc[idx, "rt_pred_after_intraday"] = base_pred
            stats["shadow_rows"] += 1
        elif mode in ("low_weight", "high_weight"):
            if policy in ("DISABLED", "SHADOW_ONLY") or shadow_flag:
                # Policy says don't apply
                merged.loc[idx, "intraday_applied"] = False
                merged.loc[idx, "rt_pred_after_intraday"] = base_pred
                if policy == "DISABLED":
                    stats["disabled_rows"] += 1
                else:
                    stats["shadow_rows"] += 1
            else:
                # Apply fusion: rt_pred_final = (1-w)*base + w*corrected
                final_pred = (1.0 - fw) * base_pred + fw * corrected_pred
                merged.loc[idx, base_pred_col] = final_pred
                merged.loc[idx, "rt_pred_after_intraday"] = final_pred
                merged.loc[idx, "intraday_applied"] = True
                stats["applied_rows"] += 1
        else:
            # Unknown mode, treat as shadow
            merged.loc[idx, "intraday_applied"] = False
            merged.loc[idx, "rt_pred_after_intraday"] = base_pred
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
