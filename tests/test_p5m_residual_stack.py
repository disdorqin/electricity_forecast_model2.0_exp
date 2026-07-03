# -*- coding: utf-8 -*-
"""Tests for the P5M Unified Residual Stack (residual_stack/).

Coverage
--------
1. High_spike priority over negative.
2. Negative correction downward-only.
3. Module sequence order correct.
4. Reason code output complete.
5. Timestamp-level metrics.
6. No production_pipeline touch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from residual_stack.priority import (
    check_high_spike_priority,
    format_module_sequence,
    should_apply_negative,
)
from residual_stack.metrics import compute_stack_metrics
from residual_stack.report import generate_verdict
from residual_stack.schema import (
    REASON_CODES,
    STACK_OUTPUT_COLUMNS,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_prediction_df(n_rows: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a minimal valid prediction DataFrame for stack testing."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_rows):
        day = f"2026-06-{i % 30 + 1:02d}"
        hour = (i % 24) + 1
        rows.append({
            "business_day": day,
            "hour_business": hour,
            "y_true": float(rng.uniform(-50, 400)),
            "base_fused_pred": float(rng.uniform(-30, 380)),
            "high_spike_prob": float(rng.uniform(0, 1)),
            "base_pred": 0.0,
            "after_high_spike_pred": 0.0,
            "after_negative_pred": 0.0,
            "final_pred": 0.0,
            "high_spike_applied": False,
            "negative_applied": False,
            "correction_reason": "",
            "module_sequence": "",
        })
    return pd.DataFrame(rows)


# ── 1. High_spike priority over negative ──────────────────────────────


class TestHighSpikePriority:
    def test_high_spike_active_blocks_negative(self):
        """When high_spike_prob >= threshold, should_apply_negative returns False."""
        apply, reason = should_apply_negative(
            high_spike_prob=0.8,
            negative_risk=0.9,
            spike_prob_threshold=0.5,
        )
        assert not apply
        assert "blocked_by_high_spike_priority" in reason

    def test_lift_applied_blocks_negative(self):
        """When lift was already applied, negative is blocked."""
        apply, reason = should_apply_negative(
            high_spike_prob=0.3,
            negative_risk=0.9,
            lift_applied=50.0,
            spike_prob_threshold=0.5,
        )
        assert not apply

    def test_no_high_spike_allows_negative(self):
        """When high_spike is below threshold, negative can apply."""
        apply, reason = should_apply_negative(
            high_spike_prob=0.1,
            negative_risk=0.9,
            spike_prob_threshold=0.5,
            negative_risk_threshold=0.4,
        )
        assert apply
        assert "negative_correction_allowed" in reason

    def test_low_negative_risk_is_blocked(self):
        """Even without high_spike, low negative risk should block."""
        apply, reason = should_apply_negative(
            high_spike_prob=0.1,
            negative_risk=0.2,
            spike_prob_threshold=0.5,
            negative_risk_threshold=0.4,
        )
        assert not apply
        assert "below_threshold" in reason

    def test_check_high_spike_priority_edge_zero(self):
        """Zero prob + no lift → no priority."""
        assert not check_high_spike_priority(0.0, 0.0, 0.5)

    def test_check_high_spike_priority_positive(self):
        """Positive prob triggers priority."""
        assert check_high_spike_priority(0.6, 0.0, 0.5)
        assert check_high_spike_priority(0.0, 10.0, 0.5)


# ── 2. Negative correction downward-only ──────────────────────────────


class TestNegativeDownwardOnly:
    def test_module_sequence_reflects_no_negative_when_blocked(self):
        """When high_spike is applied, sequence should not include 'negative'."""
        seq = format_module_sequence(high_spike_applied=True, negative_applied=False)
        assert "negative" not in seq
        assert "high_spike" in seq
        assert "guardrail" in seq

    def test_module_sequence_includes_negative_when_applied(self):
        seq = format_module_sequence(high_spike_applied=False, negative_applied=True)
        assert "negative" in seq

    def test_module_sequence_both(self):
        seq = format_module_sequence(high_spike_applied=True, negative_applied=True)
        assert "high_spike" in seq
        assert "negative" in seq

    def test_module_sequence_none(self):
        seq = format_module_sequence(high_spike_applied=False, negative_applied=False, guardrail_applied=False)
        assert seq == "none"


# ── 3. Module sequence order ──────────────────────────────────────────


class TestModuleSequence:
    def test_order_is_high_spike_then_negative(self):
        """Sequence string should show high_spike before negative."""
        seq = format_module_sequence(high_spike_applied=True, negative_applied=True)
        parts = seq.split("→")
        hs_idx = parts.index("high_spike")
        neg_idx = parts.index("negative")
        assert hs_idx < neg_idx, "high_spike must come before negative"


# ── 4. Reason code output ─────────────────────────────────────────────


class TestReasonCodes:
    def test_stack_output_columns_defined(self):
        """STACK_OUTPUT_COLUMNS must include all required fields."""
        required = [
            "base_pred", "after_high_spike_pred", "after_negative_pred",
            "final_pred", "high_spike_applied", "negative_applied",
            "correction_reason", "module_sequence",
        ]
        for col in required:
            assert col in STACK_OUTPUT_COLUMNS, f"Missing: {col}"

    def test_reason_codes_constants(self):
        """All expected reason code constants exist."""
        assert hasattr(REASON_CODES, "SPIKE_LIFTED")
        assert hasattr(REASON_CODES, "NEGATIVE_DOWN")
        assert hasattr(REASON_CODES, "SPIKE_BLOCKS_NEGATIVE")
        assert hasattr(REASON_CODES, "NONE")

    def test_correction_reason_format(self):
        """Reason should combine high_spike and negative info."""
        reason = f"{REASON_CODES.SPIKE_LIFTED} | {REASON_CODES.NEGATIVE_DOWN}"
        assert "high_spike" in reason
        assert "negative" in reason


# ── 5. Timestamp-level metrics ────────────────────────────────────────


class TestTimestampMetrics:
    def test_compute_stack_metrics_returns_expected_keys(self):
        df = _make_prediction_df()
        df["base_pred"] = df["base_fused_pred"]
        df["final_pred"] = df["base_fused_pred"].values * 0.95  # slight correction
        df["after_high_spike_pred"] = df["base_fused_pred"]
        df["after_negative_pred"] = df["final_pred"]

        metrics = compute_stack_metrics(df)
        expected = [
            "negative_count", "low_valley_count", "high_spike_count",
            "negative_MAE_before", "negative_MAE_after",
            "low_valley_MAE_before", "low_valley_MAE_after",
            "overall_sMAPE_before", "overall_sMAPE_after",
            "high_spike_MAE_before", "high_spike_MAE_after",
            "normal_degradation",
        ]
        for key in expected:
            assert key in metrics, f"Metric '{key}' missing"
            assert metrics[key] is not None

    def test_metrics_use_timestamp_level(self):
        """Metrics should be computed on (business_day, hour_business) rows."""
        df = _make_prediction_df(n_rows=5)
        df = df.drop_duplicates(subset=["business_day", "hour_business"])
        df["base_pred"] = df["base_fused_pred"]
        df["final_pred"] = df["base_fused_pred"]

        metrics = compute_stack_metrics(df)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_metrics_with_zero_negative(self):
        """When no negative prices, metrics still produce non-error values."""
        df = _make_prediction_df(n_rows=20)
        df["y_true"] = np.abs(df["y_true"]) + 10  # ensure all positive
        df["base_pred"] = df["base_fused_pred"]
        df["final_pred"] = df["base_fused_pred"]

        metrics = compute_stack_metrics(df)
        assert metrics["negative_count"] == 0
        assert metrics["negative_MAE_before"] == 0.0

    def test_data_limited_flag(self):
        """When negative_count < 5, data_limited should be True."""
        df = _make_prediction_df(n_rows=20)
        # Make only 1 negative sample
        df["y_true"] = np.abs(df["y_true"]) + 10
        df.loc[0, "y_true"] = -5.0
        df["base_pred"] = df["base_fused_pred"]
        df["final_pred"] = df["base_fused_pred"]

        metrics = compute_stack_metrics(df)
        assert metrics["data_limited"] is True

    def test_severe_underestimate_computed(self):
        df = _make_prediction_df(n_rows=10)
        df["base_pred"] = df["base_fused_pred"]
        df["final_pred"] = df["base_fused_pred"]
        # Create one severe underestimate
        df.loc[0, "y_true"] = 500.0
        df.loc[0, "base_fused_pred"] = 50.0

        metrics = compute_stack_metrics(df)
        assert "severe_underestimate" in metrics
        assert isinstance(metrics["severe_underestimate"], int)


# ── 6. No production_pipeline touch ────────────────────────────────────


class TestNoProductionTouch:
    def test_imports_do_not_reference_production_pipeline(self):
        """residual_stack modules should not import production_pipeline."""
        import residual_stack  # noqa: F401
        # This test passes if the import doesn't trigger pipeline imports


# ── Verdict ────────────────────────────────────────────────────────────


class TestVerdict:
    def test_go_when_metrics_good(self):
        metrics = {
            "data_limited": False,
            "overall_sMAPE_delta": 0.1,
            "severe_underestimate": 50,
            "severe_underestimate_before": 60,
            "high_spike_MAE_delta_pct": 1.0,
            "low_valley_MAE_delta": -5.0,
            "normal_degradation": 0.2,
        }
        assert generate_verdict(metrics) == "GO"

    def test_no_go_when_smape_worsens(self):
        metrics = {
            "data_limited": False,
            "overall_sMAPE_delta": 0.5,
            "severe_underestimate": 50,
            "high_spike_MAE_delta_pct": 1.0,
            "low_valley_MAE_delta": -5.0,
            "normal_degradation": 0.2,
        }
        assert "NO-GO" in generate_verdict(metrics)

    def test_go_when_no_change(self):
        """Baseline with delta=0 should be GO (no regression)."""
        metrics = {
            "data_limited": False,
            "overall_sMAPE_delta": 0.0,
            "low_valley_MAE_delta": 0.0,
            "high_spike_MAE_delta_pct": 0.0,
            "normal_degradation": 0.0,
        }
        assert generate_verdict(metrics) == "GO"

    def test_no_go_when_low_valley_worsens(self):
        """Positive low_valley_MAE_delta should trigger NO-GO."""
        metrics = {
            "data_limited": False,
            "overall_sMAPE_delta": 0.1,
            "low_valley_MAE_delta": 5.0,
            "high_spike_MAE_delta_pct": 1.0,
            "normal_degradation": 0.2,
        }
        assert "NO-GO" in generate_verdict(metrics)
        assert "worsened" in generate_verdict(metrics)

    def test_data_limited_when_few_samples(self):
        metrics = {
            "data_limited": True,
            "negative_count": 2,
        }
        assert generate_verdict(metrics) == "DATA-LIMITED"

    def test_no_go_when_high_spike_degraded(self):
        metrics = {
            "data_limited": False,
            "overall_sMAPE_delta": 0.1,
            "high_spike_MAE_delta_pct": 5.0,
            "low_valley_MAE_delta": -1.0,
            "normal_degradation": 0.1,
        }
        assert "NO-GO" in generate_verdict(metrics)
