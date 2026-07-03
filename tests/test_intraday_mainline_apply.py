"""Tests for corrections.intraday_tracker.apply — Phase 11."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.apply import apply_intraday_tracker_correction
from corrections.intraday_tracker.policy import IntradayTrackerMainlineConfig


def _make_base_forecast():
    """Create a synthetic base forecast."""
    rows = []
    for h in range(9, 17):
        rows.append({
            "business_day": pd.Timestamp("2026-02-15"),
            "hour_business": h,
            "ds": pd.Timestamp(f"2026-02-15 {h:02d}:00:00"),
            "rt_pred": 100.0 + h,
            "model_name": "sgdfnet",
        })
    return pd.DataFrame(rows)


def _make_pack(cutoff=12, confidence=0.6, policy="LOW_WEIGHT", fusion_weight=0.12):
    """Create a synthetic intraday pack."""
    rows = []
    for h in range(cutoff + 1, 17):
        rows.append({
            "business_day": pd.Timestamp("2026-02-15"),
            "target_hour": h,
            "hour_business": h,
            "cutoff_hour": cutoff,
            "ds": pd.Timestamp(f"2026-02-15 {h:02d}:00:00"),
            "mode": "INTRADAY",
            "base_model_name": "sgdfnet",
            "base_pred": 100.0 + h,
            "intraday_corrected_pred": 100.0 + h - 10,  # correction of -10
            "intraday_final_correction": -10.0,
            "intraday_confidence": confidence,
            "policy_decision": policy,
            "fusion_weight": fusion_weight,
            "shadow_only_flag": False,
            "guardrail_reason": "",
            "n_observed": 5,
            "residual_std_today": 50.0,
        })
    return pd.DataFrame(rows)


class TestApplyIntradayCorrection:
    def test_shadow_mode_no_change(self):
        base = _make_base_forecast()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(base, pack, mode="shadow")
        # rt_pred should NOT change in shadow mode
        assert (result["rt_pred"] == _make_base_forecast()["rt_pred"]).all()
        assert stats["applied_rows"] == 0
        assert stats["shadow_rows"] > 0

    def test_low_weight_fusion_formula(self):
        base = _make_base_forecast()
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        config = IntradayTrackerMainlineConfig(low_weight=0.12)
        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", config=config
        )
        # Check that applied rows have blended prediction
        applied = result[result["intraday_applied"] == True]
        assert len(applied) > 0
        # rt_pred should be (1-0.12)*base + 0.12*corrected
        for _, row in applied.iterrows():
            base_val = row["rt_pred_before_intraday"]
            corrected_val = row["intraday_shadow_pred"]
            expected = 0.88 * base_val + 0.12 * corrected_val
            assert abs(row["rt_pred"] - expected) < 0.01

    def test_high_weight_fusion_formula(self):
        base = _make_base_forecast()
        pack = _make_pack(cutoff=14, confidence=0.6, policy="HIGH_WEIGHT", fusion_weight=0.22)
        config = IntradayTrackerMainlineConfig(high_weight=0.22)
        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="high_weight", config=config
        )
        applied = result[result["intraday_applied"] == True]
        assert len(applied) > 0
        for _, row in applied.iterrows():
            base_val = row["rt_pred_before_intraday"]
            corrected_val = row["intraday_shadow_pred"]
            expected = 0.78 * base_val + 0.22 * corrected_val
            assert abs(row["rt_pred"] - expected) < 0.01

    def test_missing_pack_safe_fallback(self):
        base = _make_base_forecast()
        result, stats = apply_intraday_tracker_correction(base, pd.DataFrame(), mode="low_weight")
        assert stats["applied_rows"] == 0
        assert stats["fallback_reason"] == "empty_pack"
        assert stats["safe_fallback"] is True
        # rt_pred unchanged
        assert (result["rt_pred"] == _make_base_forecast()["rt_pred"]).all()

    def test_full_day_disables_correction(self):
        base = _make_base_forecast()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", prediction_mode="FULL_DAY"
        )
        assert stats["applied_rows"] == 0
        assert stats["fallback_reason"] == "prediction_mode=FULL_DAY"

    def test_disabled_policy_no_apply(self):
        base = _make_base_forecast()
        pack = _make_pack(policy="DISABLED", fusion_weight=0.0)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        assert stats["applied_rows"] == 0

    def test_rt_pred_before_preserved(self):
        base = _make_base_forecast()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        assert "rt_pred_before_intraday" in result.columns
        original = _make_base_forecast()
        assert (result["rt_pred_before_intraday"] == original["rt_pred"]).all()

    def test_stats_counts_correct(self):
        base = _make_base_forecast()
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        assert stats["matched_rows"] > 0
        assert stats["applied_rows"] + stats["shadow_rows"] + stats["disabled_rows"] == stats["matched_rows"] + (8 - stats["matched_rows"])

    def test_off_mode_no_change(self):
        base = _make_base_forecast()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(base, pack, mode="off")
        assert stats["applied_rows"] == 0
        assert not stats["intraday_enabled"]
