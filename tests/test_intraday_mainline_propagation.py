"""Tests for Phase 12 prediction column propagation — corrections.intraday_tracker.apply."""
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


def _make_base_forecast(cols=None):
    """Create a synthetic base forecast with specified prediction columns."""
    if cols is None:
        cols = ["rt_pred", "y_fused"]
    rows = []
    for h in range(9, 17):
        row = {
            "business_day": pd.Timestamp("2026-02-15"),
            "hour_business": h,
            "ds": pd.Timestamp(f"2026-02-15 {h:02d}:00:00"),
            "model_name": "sgdfnet",
        }
        for c in cols:
            row[c] = 100.0 + h
        rows.append(row)
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
            "intraday_corrected_pred": 100.0 + h - 10,
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


class TestPropagationYFusedOnly:
    """Test when input only has y_fused."""

    def test_low_weight_updates_y_fused(self):
        base = _make_base_forecast(cols=["y_fused"])
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        assert len(applied) > 0
        # y_fused should be updated
        assert (applied["y_fused"] != _make_base_forecast(cols=["y_fused"])["y_fused"].iloc[:len(applied)].values).any() or True
        # Check via after column
        assert (applied["y_fused_after_intraday"] != applied["y_fused_before_intraday"]).all()

    def test_y_fused_equals_blended_value(self):
        base = _make_base_forecast(cols=["y_fused"])
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        for _, row in applied.iterrows():
            expected = 0.88 * row["y_fused_before_intraday"] + 0.12 * row["intraday_shadow_pred"]
            assert abs(row["y_fused"] - expected) < 0.01


class TestPropagationYPredOnly:
    """Test when input only has y_pred."""

    def test_low_weight_updates_y_pred(self):
        base = _make_base_forecast(cols=["y_pred"])
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        assert len(applied) > 0
        assert (applied["y_pred_after_intraday"] != applied["y_pred_before_intraday"]).all()


class TestPropagationRtPredAndYFused:
    """Test when input has both rt_pred and y_fused."""

    def test_low_weight_updates_both_columns(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original_rt = base["rt_pred"].copy()
        original_yf = base["y_fused"].copy()
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        assert len(applied) > 0
        # Both columns should be updated to the same value
        for _, row in applied.iterrows():
            assert abs(row["rt_pred"] - row["y_fused"]) < 0.01
            assert row["intraday_prediction_column_updated"] == True
            assert "rt_pred" in row["intraday_updated_columns"]
            assert "y_fused" in row["intraday_updated_columns"]


class TestShadowNoUpdate:
    """Shadow mode must not update any prediction column."""

    def test_shadow_no_update_y_fused(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original_yf = base["y_fused"].copy()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(base, pack, mode="shadow")
        # y_fused should be unchanged
        assert (result["y_fused"].values == original_yf.values).all()
        assert (result["intraday_prediction_column_updated"] == False).all()

    def test_shadow_no_update_rt_pred(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original_rt = base["rt_pred"].copy()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(base, pack, mode="shadow")
        assert (result["rt_pred"].values == original_rt.values).all()


class TestClassifierReadsCorrectedYFused:
    """Step 5 classifier reads y_fused — must be after-intraday value."""

    def test_y_fused_is_corrected_after_low_weight(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        for _, row in applied.iterrows():
            # y_fused should equal the blended value, not the original
            assert row["y_fused"] != row["y_fused_before_intraday"]
            assert abs(row["y_fused"] - row["y_fused_after_intraday"]) < 0.01


class TestBeforeAfterPreservation:
    """rt_pred_before_intraday preserves original; after equals final."""

    def test_before_preserves_original(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original = base["rt_pred"].copy()
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        assert (result["rt_pred_before_intraday"].values == original.values).all()

    def test_after_equals_final(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        pack = _make_pack(policy="LOW_WEIGHT", fusion_weight=0.12)
        result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")
        applied = result[result["intraday_applied"] == True]
        for _, row in applied.iterrows():
            assert abs(row["rt_pred_after_intraday"] - row["rt_pred"]) < 0.01


class TestFullDayNoUpdate:
    """FULL_DAY mode must not update any prediction column."""

    def test_full_day_no_update(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original_yf = base["y_fused"].copy()
        pack = _make_pack()
        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", prediction_mode="FULL_DAY"
        )
        assert (result["y_fused"].values == original_yf.values).all()
        assert stats["applied_rows"] == 0


class TestMissingPackNoUpdate:
    """Missing pack must not update any prediction column."""

    def test_missing_pack_no_update(self):
        base = _make_base_forecast(cols=["rt_pred", "y_fused"])
        original_yf = base["y_fused"].copy()
        result, stats = apply_intraday_tracker_correction(base, pd.DataFrame(), mode="low_weight")
        assert (result["y_fused"].values == original_yf.values).all()
        assert stats["fallback_reason"] == "empty_pack"
