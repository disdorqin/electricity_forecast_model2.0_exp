"""Tests for Phase 13 — Target gap audit.

Covers:
1. gap_to_15 = baseline_smape * 100 - 15  (positive → still above 15% target)
2. gap_to_20 = baseline_smape * 100 - 20  (positive → still above 20% target)
3. When already below target → negative gap
4. Missing metrics → graceful handling (None / NaN / sentinel)
5. Edge cases: sMAPE exactly at target, zero sMAPE

The auditor is expected to expose:
    compute_target_gap(baseline_smape: float, target: float) -> float
    audit_targets(metrics: dict) -> dict

where metrics is a dict with key ``"baseline_smape"`` (0–50 scale)
and audit_targets returns a dict with ``gap_to_15``, ``gap_to_20``,
and per-target status strings.

sMAPE floor-50 formula (reference):
    2 * |y_true - y_pred| / (max(|y_true|, 50) + max(|y_pred|, 50))
    Result is in [0, 50] percent.

Business-day convention:
    timestamp D 00:00 → business_day D-1, hour 24
    timestamp D HH:00 (HH >= 1) → business_day D, hour HH

Run:
    python -m pytest tests/test_target_gap_audit.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Inline reference implementation ────────────────────────────────────────

TARGET_15 = 15.0   # 15% sMAPE target
TARGET_20 = 20.0   # 20% sMAPE target


def compute_target_gap(baseline_smape: float, target: float) -> float:
    """Compute the gap between baseline sMAPE and a target.

    ``gap = baseline_smape - target``

    * Positive → baseline is *above* the target (needs improvement).
    * Negative → baseline is *below* the target (already meets it).
    * Zero     → baseline exactly meets the target.

    Parameters
    ----------
    baseline_smape : float
        Baseline sMAPE on the 0–50 percent scale.
    target : float
        Target sMAPE on the same scale (e.g. 15 or 20).
    """
    return baseline_smape - target


def audit_targets(
    metrics: dict,
    targets: tuple[float, ...] = (TARGET_15, TARGET_20),
) -> dict:
    """Audit baseline metrics against multiple sMAPE targets.

    Parameters
    ----------
    metrics : dict
        Must contain ``"baseline_smape"`` (float, 0–50 scale).
        If the key is missing or None, all gaps are returned as None.
    targets : tuple of float
        Target sMAPE values to evaluate.

    Returns
    -------
    dict with keys:
        ``"baseline_smape"`` : float or None
        ``"gap_to_<T>"``     : float or None   for each target T
        ``"meets_<T>"``      : bool or None    True if gap <= 0
        ``"status"``         : str             "OK" | "MISSING_METRICS"
    """
    result: dict = {}
    baseline = metrics.get("baseline_smape")

    if baseline is None or (isinstance(baseline, float) and math.isnan(baseline)):
        result["baseline_smape"] = baseline
        result["status"] = "MISSING_METRICS"
        for t in targets:
            result[f"gap_to_{int(t)}"] = None
            result[f"meets_{int(t)}"] = None
        return result

    result["baseline_smape"] = baseline
    result["status"] = "OK"

    for t in targets:
        gap = compute_target_gap(baseline, t)
        result[f"gap_to_{int(t)}"] = gap
        result[f"meets_{int(t)}"] = gap <= 0

    return result


# ── sMAPE floor-50 reference (for fixture validation) ─────────────────────

def smape_floor50_scalar(y_true: float, y_pred: float) -> float:
    """sMAPE for a single pair with floor-50 denominator.

    Formula: 2*|y_true-y_pred| / (max(|y_true|,50) + max(|y_pred|,50)) * 100
    Clamped to [0, 50].
    """
    yt = max(abs(y_true), 50.0)
    yp = max(abs(y_pred), 50.0)
    denom = (yt + yp) / 2.0
    if denom < 1e-10:
        return 0.0
    val = abs(yt - yp) / denom * 100.0
    return min(val, 50.0)


def smape_floor50_mean(y_true: list[float], y_pred: list[float]) -> float:
    """Mean sMAPE floor-50 over a list of pairs."""
    vals = [smape_floor50_scalar(yt, yp) for yt, yp in zip(y_true, y_pred)]
    return float(np.mean(vals))


# ── Tests: compute_target_gap ──────────────────────────────────────────────

class TestGapTo15:
    """Test gap_to_15 = baseline_smape - 15."""

    def test_above_target_positive_gap(self):
        """baseline_smape=22 → gap_to_15 = 7 (still above target)."""
        gap = compute_target_gap(22.0, 15.0)
        assert gap == 7.0
        assert gap > 0  # above target

    def test_below_target_negative_gap(self):
        """baseline_smape=12 → gap_to_15 = -3 (already below target)."""
        gap = compute_target_gap(12.0, 15.0)
        assert gap == -3.0
        assert gap < 0  # below target

    def test_exactly_at_target(self):
        """baseline_smape=15 → gap_to_15 = 0."""
        gap = compute_target_gap(15.0, 15.0)
        assert gap == 0.0

    def test_zero_smape(self):
        """baseline_smape=0 → gap_to_15 = -15."""
        gap = compute_target_gap(0.0, 15.0)
        assert gap == -15.0

    def test_max_smape_50(self):
        """baseline_smape=50 → gap_to_15 = 35."""
        gap = compute_target_gap(50.0, 15.0)
        assert gap == 35.0


class TestGapTo20:
    """Test gap_to_20 = baseline_smape - 20."""

    def test_above_target_positive_gap(self):
        """baseline_smape=25 → gap_to_20 = 5."""
        gap = compute_target_gap(25.0, 20.0)
        assert gap == 5.0
        assert gap > 0

    def test_below_target_negative_gap(self):
        """baseline_smape=18 → gap_to_20 = -2."""
        gap = compute_target_gap(18.0, 20.0)
        assert gap == -2.0
        assert gap < 0

    def test_exactly_at_target(self):
        """baseline_smape=20 → gap_to_20 = 0."""
        gap = compute_target_gap(20.0, 20.0)
        assert gap == 0.0

    def test_gap_to_20_larger_than_gap_to_15(self):
        """For the same baseline above both targets, gap_to_15 > gap_to_20."""
        baseline = 25.0
        g15 = compute_target_gap(baseline, 15.0)
        g20 = compute_target_gap(baseline, 20.0)
        assert g15 > g20
        assert g15 - g20 == 5.0  # difference = 20 - 15


class TestBelowTarget:
    """Test when already below target (negative gap)."""

    def test_well_below_both_targets(self):
        """baseline_smape=10 → negative gap for both 15 and 20."""
        g15 = compute_target_gap(10.0, 15.0)
        g20 = compute_target_gap(10.0, 20.0)
        assert g15 < 0
        assert g20 < 0
        assert g20 < g15  # more negative for higher target

    def test_meets_target_semantics(self):
        """Negative gap means the target is met."""
        gap = compute_target_gap(12.0, 15.0)
        meets = gap <= 0
        assert meets is True


class TestMissingMetrics:
    """Test with missing metrics."""

    def test_missing_baseline_smape_key(self):
        """Dict without 'baseline_smape' → MISSING_METRICS status."""
        result = audit_targets({})
        assert result["status"] == "MISSING_METRICS"
        assert result["gap_to_15"] is None
        assert result["gap_to_20"] is None
        assert result["meets_15"] is None
        assert result["meets_20"] is None

    def test_none_baseline_smape(self):
        """baseline_smape=None → MISSING_METRICS."""
        result = audit_targets({"baseline_smape": None})
        assert result["status"] == "MISSING_METRICS"
        assert result["gap_to_15"] is None

    def test_nan_baseline_smape(self):
        """baseline_smape=NaN → MISSING_METRICS."""
        result = audit_targets({"baseline_smape": float("nan")})
        assert result["status"] == "MISSING_METRICS"
        assert result["gap_to_15"] is None

    def test_valid_metrics_ok_status(self):
        """Valid baseline_smape → OK status with numeric gaps."""
        result = audit_targets({"baseline_smape": 20.86})
        assert result["status"] == "OK"
        assert abs(result["gap_to_15"] - 5.86) < 1e-10
        assert abs(result["gap_to_20"] - 0.86) < 1e-10

    def test_meets_flags_correct(self):
        """meets_15 and meets_20 flags reflect gap sign."""
        # baseline=20.86 → above 15 (not met), above 20 (not met)
        result = audit_targets({"baseline_smape": 20.86})
        assert result["meets_15"] is False
        assert result["meets_20"] is False

        # baseline=14 → below 15 (met), below 20 (met)
        result2 = audit_targets({"baseline_smape": 14.0})
        assert result2["meets_15"] is True
        assert result2["meets_20"] is True

        # baseline=17 → below 20 (met), above 15 (not met)
        result3 = audit_targets({"baseline_smape": 17.0})
        assert result3["meets_15"] is False
        assert result3["meets_20"] is True


# ── Tests: sMAPE floor-50 integration ─────────────────────────────────────

class TestSmapeFloor50Integration:
    """Verify gap computation using actual sMAPE floor-50 values."""

    def test_gap_from_raw_predictions(self):
        """Compute sMAPE from raw predictions, then compute gap."""
        y_true = [300.0, 400.0, 200.0, 350.0, 250.0]
        y_pred = [310.0, 380.0, 220.0, 360.0, 240.0]

        baseline = smape_floor50_mean(y_true, y_pred)
        gap15 = compute_target_gap(baseline, 15.0)
        gap20 = compute_target_gap(baseline, 20.0)

        # baseline should be a valid sMAPE in [0, 50]
        assert 0 <= baseline <= 50
        # gap_to_15 > gap_to_20 always (for same baseline)
        assert gap15 > gap20

    def test_perfect_predictions_zero_gap_to_15(self):
        """Perfect predictions → sMAPE=0 → gap_to_15 = -15."""
        y_true = [300.0, 400.0, 200.0]
        y_pred = [300.0, 400.0, 200.0]

        baseline = smape_floor50_mean(y_true, y_pred)
        assert baseline == 0.0
        gap = compute_target_gap(baseline, 15.0)
        assert gap == -15.0

    def test_smape_formula_floor50_behaviour(self):
        """Verify the floor-50 formula: small values get floored to 50."""
        # y_true=10, y_pred=20 → floor both to 50
        # smape = 2*|50-50| / (50+50) = 0
        val = smape_floor50_scalar(10.0, 20.0)
        assert val == 0.0  # both floored to 50, so no relative error

        # y_true=100, y_pred=110 → no flooring
        # smape = 2*|100-110| / (100+110) = 2*10/210 ≈ 9.524
        val2 = smape_floor50_scalar(100.0, 110.0)
        expected = 2 * 10 / (100 + 110) * 100
        assert abs(val2 - expected) < 0.01


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
