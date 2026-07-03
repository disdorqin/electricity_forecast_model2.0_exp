"""Tests for corrections.intraday_tracker.policy — Phase 11."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.policy import (
    IntradayTrackerMainlineConfig,
    apply_mainline_intraday_policy,
)


def _make_df(n=3, cutoff=14, confidence=0.6, n_observed=5, residual_std=50.0,
             policy="HIGH_WEIGHT", fusion_weight=0.22):
    rows = []
    for i in range(n):
        rows.append({
            "business_day": "2026-02-15",
            "target_hour": cutoff + 1 + i,
            "cutoff_hour": cutoff,
            "mode": "INTRADAY",
            "intraday_confidence": confidence,
            "n_observed": n_observed,
            "residual_std_today": residual_std,
            "policy_decision": policy,
            "fusion_weight": fusion_weight,
        })
    return pd.DataFrame(rows)


class TestMainlinePolicy:
    def test_full_day_forces_disabled(self):
        df = _make_df()
        result = apply_mainline_intraday_policy(df, prediction_mode="FULL_DAY")
        assert (result["policy_decision"] == "DISABLED").all()
        assert (result["fusion_weight"] == 0.0).all()

    def test_cutoff_below_12_shadow(self):
        df = _make_df(cutoff=11, policy="LOW_WEIGHT", fusion_weight=0.12)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert (result["policy_decision"] == "SHADOW_ONLY").all()
        assert (result["fusion_weight"] == 0.0).all()

    def test_low_confidence_shadow(self):
        df = _make_df(confidence=0.30, policy="LOW_WEIGHT", fusion_weight=0.12)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert (result["policy_decision"] == "SHADOW_ONLY").all()

    def test_low_observed_disabled(self):
        df = _make_df(n_observed=2, policy="LOW_WEIGHT", fusion_weight=0.12)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert (result["policy_decision"] == "DISABLED").all()

    def test_high_residual_std_shadow(self):
        df = _make_df(residual_std=200.0, policy="LOW_WEIGHT", fusion_weight=0.12)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert (result["policy_decision"] == "SHADOW_ONLY").all()

    def test_low_weight_cap(self):
        df = _make_df(policy="LOW_WEIGHT", fusion_weight=0.20)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert (result["fusion_weight"] <= 0.12).all()

    def test_high_weight_requires_cutoff_14(self):
        df = _make_df(cutoff=13, policy="HIGH_WEIGHT", fusion_weight=0.22)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        # cutoff < 14 → downgraded to LOW_WEIGHT
        assert (result["policy_decision"] == "LOW_WEIGHT").all()

    def test_high_weight_requires_confidence_055(self):
        df = _make_df(cutoff=14, confidence=0.50, policy="HIGH_WEIGHT", fusion_weight=0.22)
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        # confidence < 0.55 → downgraded to LOW_WEIGHT
        assert (result["policy_decision"] == "LOW_WEIGHT").all()

    def test_fusion_weight_never_exceeds_025(self):
        df = _make_df(policy="HIGH_WEIGHT", fusion_weight=0.22)
        config = IntradayTrackerMainlineConfig(max_fusion_weight=0.25)
        result = apply_mainline_intraday_policy(df, config, prediction_mode="INTRADAY")
        assert (result["fusion_weight"] <= 0.25).all()

    def test_config_disabled(self):
        df = _make_df()
        config = IntradayTrackerMainlineConfig(enabled=False)
        result = apply_mainline_intraday_policy(df, config, prediction_mode="INTRADAY")
        assert (result["policy_decision"] == "DISABLED").all()

    def test_config_from_yaml_missing_file(self):
        config = IntradayTrackerMainlineConfig.from_yaml("/nonexistent/path.yaml")
        assert config.enabled is True
        assert config.min_cutoff_hour == 12

    def test_empty_df(self):
        df = pd.DataFrame()
        result = apply_mainline_intraday_policy(df, prediction_mode="INTRADAY")
        assert len(result) == 0
