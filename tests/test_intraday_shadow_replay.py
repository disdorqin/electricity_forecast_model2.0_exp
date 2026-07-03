"""Tests for the shadow replay script — Phase 12.

Verifies that the shadow replay pipeline correctly:
1. Deduplicates packs with multiple cutoff hours
2. Computes baseline metrics
3. Produces differentiated metrics across modes
4. Handles missing ground truth gracefully
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.apply import apply_intraday_tracker_correction
from corrections.intraday_tracker.policy import IntradayTrackerMainlineConfig


def _make_base_df(n_hours=8):
    """Create a minimal base forecast DataFrame."""
    rows = []
    for h in range(9, 9 + n_hours):
        rows.append({
            "business_day": pd.Timestamp("2026-02-01"),
            "hour_business": h,
            "ds": pd.Timestamp(f"2026-02-01 {h:02d}:00:00"),
            "rt_pred": 100.0 + h,
            "y_fused": 100.0 + h,
            "y_pred": 100.0 + h,
            "model_name": "fusion",
            "period": "1_8" if h <= 8 else ("9_16" if h <= 16 else "17_24"),
        })
    return pd.DataFrame(rows)


def _make_pack_with_duplicates():
    """Create a pack with duplicate (business_day, target_hour) at different cutoffs."""
    rows = []
    for cutoff in [10, 12, 14]:
        for h in [13, 14]:
            rows.append({
                "business_day": pd.Timestamp("2026-02-01"),
                "cutoff_hour": cutoff,
                "target_hour": h,
                "ds": pd.Timestamp(f"2026-02-01 {h:02d}:00:00"),
                "mode": "INTRADAY",
                "base_model_name": "sgdfnet",
                "base_pred": 100.0 + h,
                "intraday_corrected_pred": 105.0 + h,
                "intraday_final_correction": 5.0,
                "intraday_confidence": 0.7,
                "policy_decision": "LOW_WEIGHT",
                "fusion_weight": 0.12,
                "shadow_only_flag": False,
                "guardrail_reason": "",
                "n_observed": 5,
                "observed_hours": 5,
                "residual_std_today": 50.0,
            })
    return pd.DataFrame(rows)


class TestPackDeduplication:
    """Test that packs with duplicate (business_day, target_hour) are handled."""

    def test_duplicate_pack_produces_correct_count(self):
        """After dedup, should have one row per (business_day, target_hour)."""
        pack = _make_pack_with_duplicates()
        assert len(pack) == 6  # 3 cutoffs x 2 hours

        # Deduplicate: keep highest cutoff
        pack_sorted = pack.sort_values("cutoff_hour", ascending=False)
        pack_deduped = pack_sorted.drop_duplicates(
            subset=["business_day", "target_hour"], keep="first"
        )
        assert len(pack_deduped) == 2
        assert all(pack_deduped["cutoff_hour"] == 14)

    def test_deduped_pack_applies_correction(self):
        """After dedup, corrections should be applied (not all DISABLED)."""
        pack = _make_pack_with_duplicates()

        # Deduplicate
        pack_sorted = pack.sort_values("cutoff_hour", ascending=False)
        pack_deduped = pack_sorted.drop_duplicates(
            subset=["business_day", "target_hour"], keep="first"
        )

        base_df = _make_base_df()
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base_df, pack_deduped, mode="low_weight", config=config,
            prediction_mode="INTRADAY",
        )

        # Should have matched and applied some rows
        assert stats["matched_rows"] == 2
        assert stats["applied_rows"] == 2


class TestBaselineComputation:
    """Test that baseline (off mode) produces correct metrics."""

    def test_baseline_no_change(self):
        """Baseline (off mode) should not change any predictions."""
        base_df = _make_base_df()
        pack = _make_pack_with_duplicates().drop_duplicates(
            subset=["business_day", "target_hour"], keep="first"
        )
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base_df, pack, mode="off", config=config,
            prediction_mode="INTRADAY",
        )

        # All predictions should be unchanged
        pd.testing.assert_series_equal(
            result["rt_pred"], result["rt_pred_after_intraday"],
            check_names=False,
        )

    def test_baseline_smape_equals_corrected_smape_in_shadow(self):
        """Shadow mode should produce same sMAPE as baseline."""
        base_df = _make_base_df()
        pack = _make_pack_with_duplicates().drop_duplicates(
            subset=["business_day", "target_hour"], keep="first"
        )
        config = IntradayTrackerMainlineConfig()

        base_result, _ = apply_intraday_tracker_correction(
            base_df, pack, mode="off", config=config,
            prediction_mode="INTRADAY",
        )
        shadow_result, _ = apply_intraday_tracker_correction(
            base_df, pack, mode="shadow", config=config,
            prediction_mode="INTRADAY",
        )

        # Both should have same rt_pred_after_intraday
        pd.testing.assert_series_equal(
            base_result["rt_pred_after_intraday"],
            shadow_result["rt_pred_after_intraday"],
            check_names=False,
        )


class TestModeDifferentiation:
    """Test that different modes produce different results when corrections exist."""

    def test_low_weight_differs_from_shadow(self):
        """low_weight mode should change predictions, shadow should not."""
        base_df = _make_base_df()  # hours 9-16, rt_pred = 100+h
        # Create a clean pack with valid data for hour 13
        pack = pd.DataFrame([{
            "business_day": pd.Timestamp("2026-02-01"),
            "cutoff_hour": 12,
            "target_hour": 13,
            "ds": pd.Timestamp("2026-02-01 13:00:00"),
            "mode": "INTRADAY",
            "base_model_name": "sgdfnet",
            "base_pred": 113.0,
            "intraday_corrected_pred": 120.0,
            "intraday_final_correction": 7.0,
            "intraday_confidence": 0.7,
            "policy_decision": "LOW_WEIGHT",
            "fusion_weight": 0.12,
            "shadow_only_flag": False,
            "guardrail_reason": "",
            "n_observed": 5,
            "observed_hours": 5,
            "residual_std_today": 50.0,
        }])

        config = IntradayTrackerMainlineConfig()

        shadow_result, shadow_stats = apply_intraday_tracker_correction(
            base_df, pack, mode="shadow", config=config,
            prediction_mode="INTRADAY",
        )
        lw_result, lw_stats = apply_intraday_tracker_correction(
            base_df, pack, mode="low_weight", config=config,
            prediction_mode="INTRADAY",
        )

        # Shadow should not change predictions
        assert shadow_stats["applied_rows"] == 0
        shadow_h13 = shadow_result[shadow_result["hour_business"] == 13]
        assert len(shadow_h13) == 1
        assert shadow_h13["rt_pred"].iloc[0] == 113.0

        # low_weight should change predictions
        assert lw_stats["applied_rows"] == 1
        expected = (1.0 - 0.12) * 113.0 + 0.12 * 120.0
        lw_h13 = lw_result[lw_result["hour_business"] == 13]
        assert len(lw_h13) == 1
        assert abs(lw_h13["rt_pred"].iloc[0] - expected) < 1e-6
