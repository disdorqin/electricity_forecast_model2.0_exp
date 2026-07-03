"""Mainline policy gating for Intraday Tracker — Phase 11.

Second layer of defense: the main repo must NOT blindly trust the
external pack's policy_decision. This module re-evaluates and enforces
mainline-specific rules.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradayTrackerMainlineConfig:
    """Configuration for mainline intraday tracker policy."""
    enabled: bool = True
    default_mode: str = "shadow"  # shadow | low_weight | high_weight | off
    min_cutoff_hour: int = 12
    high_weight_cutoff_hour: int = 14
    min_confidence_low: float = 0.35
    min_confidence_high: float = 0.55
    min_observed_hours: int = 3
    max_residual_std: float = 180.0
    low_weight: float = 0.12
    high_weight: float = 0.22
    max_fusion_weight: float = 0.25
    disable_full_day: bool = True
    disable_day_ahead: bool = True
    negative_guardrail: bool = True

    @classmethod
    def from_yaml(cls, path: str) -> "IntradayTrackerMainlineConfig":
        """Load config from YAML file."""
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, using defaults")
            return cls()

        p = Path(path)
        if not p.exists():
            logger.warning("Config file not found: %s, using defaults", path)
            return cls()

        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def apply_mainline_intraday_policy(
    df: pd.DataFrame,
    config: Optional[IntradayTrackerMainlineConfig] = None,
    prediction_mode: str = "INTRADAY",
) -> pd.DataFrame:
    """Apply mainline policy gating to intraday pack.

    This is the second layer of defense. Even if the deep branch pack
    says HIGH_WEIGHT, the mainline policy can override it.

    Parameters
    ----------
    df : pd.DataFrame
        Normalized intraday pack with columns:
        mode, cutoff_hour, intraday_confidence, n_observed,
        residual_std_today, policy_decision, fusion_weight
    config : IntradayTrackerMainlineConfig
        Policy configuration.
    prediction_mode : str
        "INTRADAY" or "FULL_DAY". If FULL_DAY, all corrections disabled.

    Returns
    -------
    pd.DataFrame with updated policy_decision and fusion_weight columns.
    """
    if config is None:
        config = IntradayTrackerMainlineConfig()

    if len(df) == 0:
        return df

    df = df.copy()

    # Ensure numeric types
    df["cutoff_hour"] = pd.to_numeric(df.get("cutoff_hour", 0), errors="coerce").fillna(0).astype(int)
    df["intraday_confidence"] = pd.to_numeric(df.get("intraday_confidence", 0), errors="coerce").fillna(0.0)
    df["fusion_weight"] = pd.to_numeric(df.get("fusion_weight", 0), errors="coerce").fillna(0.0)

    # n_observed may not be in normalized pack; get from original if available
    if "n_observed" not in df.columns:
        df["n_observed"] = 0
    df["n_observed"] = pd.to_numeric(df["n_observed"], errors="coerce").fillna(0).astype(int)

    if "residual_std_today" not in df.columns:
        df["residual_std_today"] = 0.0
    df["residual_std_today"] = pd.to_numeric(df["residual_std_today"], errors="coerce").fillna(0.0)

    # Rule 0: If config disabled, disable everything
    if not config.enabled:
        df["policy_decision"] = "DISABLED"
        df["fusion_weight"] = 0.0
        return df

    # Rule 1: prediction_mode must be INTRADAY
    if config.disable_full_day and prediction_mode != "INTRADAY":
        df["policy_decision"] = "DISABLED"
        df["fusion_weight"] = 0.0
        logger.info("Mainline policy: FULL_DAY mode → all DISABLED")
        return df

    # Rule 2: mode must be INTRADAY
    if "mode" in df.columns:
        non_intraday_mask = df["mode"] != "INTRADAY"
        if non_intraday_mask.any():
            df.loc[non_intraday_mask, "policy_decision"] = "DISABLED"
            df.loc[non_intraday_mask, "fusion_weight"] = 0.0

    # Rule 3: n_observed < min → DISABLED
    low_obs_mask = df["n_observed"] < config.min_observed_hours
    if low_obs_mask.any():
        df.loc[low_obs_mask, "policy_decision"] = "DISABLED"
        df.loc[low_obs_mask, "fusion_weight"] = 0.0

    # Rule 4: cutoff < 12 → SHADOW_ONLY
    low_cutoff_mask = df["cutoff_hour"] < config.min_cutoff_hour
    # Only apply to rows not already DISABLED
    active_mask = df["policy_decision"] != "DISABLED"
    shadow_cutoff_mask = low_cutoff_mask & active_mask
    if shadow_cutoff_mask.any():
        df.loc[shadow_cutoff_mask, "policy_decision"] = "SHADOW_ONLY"
        df.loc[shadow_cutoff_mask, "fusion_weight"] = 0.0

    # Rule 5: confidence < min_confidence_low → SHADOW_ONLY
    active_mask = df["policy_decision"].isin(["LOW_WEIGHT", "HIGH_WEIGHT"])
    low_conf_mask = (df["intraday_confidence"] < config.min_confidence_low) & active_mask
    if low_conf_mask.any():
        df.loc[low_conf_mask, "policy_decision"] = "SHADOW_ONLY"
        df.loc[low_conf_mask, "fusion_weight"] = 0.0

    # Rule 6: residual_std > max → SHADOW_ONLY
    active_mask = df["policy_decision"].isin(["LOW_WEIGHT", "HIGH_WEIGHT"])
    high_std_mask = (df["residual_std_today"] > config.max_residual_std) & active_mask
    if high_std_mask.any():
        df.loc[high_std_mask, "policy_decision"] = "SHADOW_ONLY"
        df.loc[high_std_mask, "fusion_weight"] = 0.0

    # Rule 7: LOW_WEIGHT cap
    low_mask = df["policy_decision"] == "LOW_WEIGHT"
    if low_mask.any():
        df.loc[low_mask, "fusion_weight"] = df.loc[low_mask, "fusion_weight"].clip(upper=config.low_weight)

    # Rule 8: HIGH_WEIGHT requires cutoff >= high_weight_cutoff AND confidence >= min_confidence_high
    high_mask = df["policy_decision"] == "HIGH_WEIGHT"
    if high_mask.any():
        invalid_high = high_mask & (
            (df["cutoff_hour"] < config.high_weight_cutoff_hour)
            | (df["intraday_confidence"] < config.min_confidence_high)
        )
        if invalid_high.any():
            df.loc[invalid_high, "policy_decision"] = "LOW_WEIGHT"
            df.loc[invalid_high, "fusion_weight"] = df.loc[invalid_high, "fusion_weight"].clip(upper=config.low_weight)

    # Rule 9: fusion_weight must never exceed max_fusion_weight
    df["fusion_weight"] = df["fusion_weight"].clip(upper=config.max_fusion_weight)

    # Rule 10: fusion_weight must be non-negative
    df["fusion_weight"] = df["fusion_weight"].clip(lower=0.0)

    return df
