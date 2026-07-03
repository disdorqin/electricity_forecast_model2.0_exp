"""Smoke tests for the intraday pipeline — Phase 12.

Verifies end-to-end functionality of the intraday tracker integration
using tiny fixture data. These tests ensure the pipeline doesn't crash
and produces correct structural output.
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


def _make_tiny_fused():
    """Create a tiny fused predictions DataFrame (8 rows)."""
    rows = []
    for h in range(9, 17):
        rows.append({
            "business_day": pd.Timestamp("2026-02-01"),
            "hour_business": h,
            "ds": pd.Timestamp(f"2026-02-01 {h:02d}:00:00"),
            "y_fused": 100.0 + h * 2,
            "rt_pred": 100.0 + h * 2,
            "y_pred": 100.0 + h * 2,
            "model_name": "fusion",
            "period": "9_16",
            "target_day": "2026-02-01",
        })
    return pd.DataFrame(rows)


def _make_tiny_pack():
    """Create a tiny intraday pack (4 rows, cutoff=12, hours 13-16)."""
    rows = []
    for h in range(13, 17):
        rows.append({
            "business_day": pd.Timestamp("2026-02-01"),
            "cutoff_hour": 12,
            "target_hour": h,
            "ds": pd.Timestamp(f"2026-02-01 {h:02d}:00:00"),
            "mode": "INTRADAY",
            "base_model_name": "sgdfnet",
            "base_pred": 100.0 + h * 2,
            "intraday_corrected_pred": 100.0 + h * 2 + 5,
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


class TestSmokeShadowMode:
    """Smoke test: shadow mode should not change predictions."""

    def test_shadow_no_change_y_fused(self):
        base = _make_tiny_fused()
        pack = _make_tiny_pack()
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="shadow", config=config,
            prediction_mode="INTRADAY",
        )

        # y_fused should be unchanged
        pd.testing.assert_series_equal(
            result["y_fused"], base["y_fused"],
            check_names=False,
        )
        assert stats["shadow_rows"] == 4
        assert stats["applied_rows"] == 0


class TestSmokeLowWeight:
    """Smoke test: low_weight mode should change predictions."""

    def test_low_weight_changes_y_fused(self):
        base = _make_tiny_fused()
        pack = _make_tiny_pack()
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", config=config,
            prediction_mode="INTRADAY",
        )

        # y_fused should be changed for matched rows
        assert stats["applied_rows"] == 4
        # Check that y_fused was updated
        for h in range(13, 17):
            row = result[result["hour_business"] == h].iloc[0]
            base_val = 100.0 + h * 2
            corrected_val = 100.0 + h * 2 + 5
            expected = (1.0 - 0.12) * base_val + 0.12 * corrected_val
            assert abs(row["y_fused"] - expected) < 1e-6


class TestSmokeFullDay:
    """Smoke test: FULL_DAY mode should disable all corrections."""

    def test_full_day_disables(self):
        base = _make_tiny_fused()
        pack = _make_tiny_pack()
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", config=config,
            prediction_mode="FULL_DAY",
        )

        # y_fused should be unchanged
        pd.testing.assert_series_equal(
            result["y_fused"], base["y_fused"],
            check_names=False,
        )
        assert stats["applied_rows"] == 0
        assert stats["fallback_reason"] == "prediction_mode=FULL_DAY"


class TestSmokeMissingPack:
    """Smoke test: missing pack should trigger safe fallback."""

    def test_missing_pack_safe_fallback(self):
        base = _make_tiny_fused()
        empty_pack = pd.DataFrame()
        config = IntradayTrackerMainlineConfig()

        result, stats = apply_intraday_tracker_correction(
            base, empty_pack, mode="low_weight", config=config,
            prediction_mode="INTRADAY",
        )

        # y_fused should be unchanged
        pd.testing.assert_series_equal(
            result["y_fused"], base["y_fused"],
            check_names=False,
        )
        assert stats["applied_rows"] == 0
        assert stats["safe_fallback"] is True
        assert stats["fallback_reason"] == "empty_pack"


class TestSmokeManifestFields:
    """Smoke test: stats dict should contain all required fields."""

    def test_stats_has_required_fields(self):
        base = _make_tiny_fused()
        pack = _make_tiny_pack()
        config = IntradayTrackerMainlineConfig()

        _, stats = apply_intraday_tracker_correction(
            base, pack, mode="low_weight", config=config,
            prediction_mode="INTRADAY",
        )

        required_fields = [
            "intraday_enabled", "intraday_mode", "prediction_mode",
            "pack_rows", "matched_rows", "applied_rows", "shadow_rows",
            "disabled_rows", "avg_fusion_weight", "avg_confidence",
            "policy_counts", "guardrail_counts", "fallback_reason",
            "safe_fallback",
        ]
        for field in required_fields:
            assert field in stats, f"Missing required stats field: {field}"
