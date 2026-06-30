"""Tests for P3.2 Rolling Base + Spike Correction pipeline.

Covers:
1. build_prediction_pack produces CSV with correct columns
2. Correction pipeline runs end-to-end with rolling predictions
3. compute_all_metrics works on corrected output (no crashes)
4. GO assessment logic (threshold checks)
5. Comparison table generation

Run:
    python -m pytest tests/test_p32_rolling_base_correction.py -v
    python tests/test_p32_rolling_base_correction.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from extreme.realtime_high_spike.apply_correction import (
    CorrectionMode,
    CorrectionProfile,
    get_profile,
)
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from evaluate_p32_rolling_base_correction import (
    build_prediction_pack,
    build_comparison_table,
    assess_go,
    GO_THRESHOLDS,
    PHASE2_BEST_METRICS,
    P31_BEST_METRICS,
)


# ── Helpers ────────────────────────────────────────────────────────────

def make_mock_rolling_predictions(n_days: int = 35, seed: int = 42) -> pd.DataFrame:
    """Create synthetic rolling predictions like P3.1 severe_softmax output."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    rows: list[dict] = []

    for i in range(n_days):
        day = start + pd.Timedelta(days=i)
        bd = day.strftime("%Y-%m-%d")
        for hb in range(1, 25):
            ts = day + pd.Timedelta(hours=hb - 1)
            hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hb - 6) / 24)
            y_true = 300.0 * hour_factor + rng.normal(0, 20)
            # rolling prediction slightly better than naive
            base_pred = y_true * 0.92 + rng.normal(0, 15)
            rows.append({
                "business_day": bd,
                "hour_business": hb,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "base_fused_pred": round(base_pred, 2),
                "y_true": round(y_true, 2),
            })

    return pd.DataFrame(rows)


def make_mock_risk_predictions(n_days: int = 35, seed: int = 42) -> pd.DataFrame:
    """Create synthetic risk predictions matching rolling predictions."""
    rng = np.random.default_rng(seed + 1)
    start = pd.Timestamp("2026-01-01")
    rows: list[dict] = []

    for i in range(n_days):
        day = start + pd.Timedelta(days=i)
        bd = day.strftime("%Y-%m-%d")
        for hb in range(1, 25):
            ts = day + pd.Timedelta(hours=hb - 1)
            spike_prob = rng.uniform(0, 1)
            rows.append({
                "business_day": bd,
                "hour_business": hb,
                "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "high_spike_prob": round(spike_prob, 6),
                "spike_risk_score": round(spike_prob, 6),
                "spike_risk_flag": 1 if spike_prob > 0.7 else 0,
            })

    return pd.DataFrame(rows)


# ── Test 1: build_prediction_pack ──────────────────────────────────────

def test_build_prediction_pack_columns():
    """build_prediction_pack produces CSV with required columns."""
    roll = make_mock_rolling_predictions(n_days=10)

    with tempfile.TemporaryDirectory() as tmp:
        roll_path = Path(tmp) / "rolling.csv"
        roll.to_csv(roll_path, index=False)

        out_dir = Path(tmp) / "pack"
        pack_path = build_prediction_pack(roll_path, out_dir)

        assert pack_path.exists(), "Prediction pack CSV not created"
        pack = pd.read_csv(pack_path)

        required = {"business_day", "hour_business", "base_fused_pred", "y_true"}
        missing = required - set(pack.columns)
        assert not missing, f"Missing columns: {missing}"

        # Should have 1 row per timestamp
        n_timestamps = roll["business_day"].nunique() * 24
        assert len(pack) == n_timestamps, (
            f"Expected {n_timestamps} rows, got {len(pack)}"
        )


def test_build_prediction_pack_preserves_values():
    """Prediction pack values match the rolling predictions."""
    roll = make_mock_rolling_predictions(n_days=5)

    with tempfile.TemporaryDirectory() as tmp:
        roll_path = Path(tmp) / "rolling.csv"
        roll.to_csv(roll_path, index=False)

        out_dir = Path(tmp) / "pack"
        pack_path = build_prediction_pack(roll_path, out_dir)
        pack = pd.read_csv(pack_path)

        # Values should match
        pd.testing.assert_series_equal(
            pack["base_fused_pred"], roll["base_fused_pred"],
            check_names=False,
        )
        pd.testing.assert_series_equal(
            pack["y_true"], roll["y_true"],
            check_names=False,
        )


# ── Test 2: End-to-end correction pipeline ─────────────────────────────

def test_end_to_end_correction_runs():
    """Correction pipeline runs end-to-end without error."""
    n_days = 10
    roll = make_mock_rolling_predictions(n_days=n_days)
    risk = make_mock_risk_predictions(n_days=n_days)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Save inputs
        roll_path = tmp_path / "rolling.csv"
        roll.to_csv(roll_path, index=False)
        risk_path = tmp_path / "risk.csv"
        risk.to_csv(risk_path, index=False)

        # Build prediction pack
        pack_path = build_prediction_pack(roll_path, tmp_path)

        # Run correction with medium profile
        profile = CorrectionProfile(
            name="medium",
            spike_prob_threshold=0.60,
            max_lift_ratio=0.35,
            max_absolute_lift=350.0,
            protect_normal_hours=True,
            period_9_16_boost=1.15,
        )
        profile_out = tmp_path / "medium"
        profile_out.mkdir(exist_ok=True)

        from extreme.realtime_high_spike.apply_correction import run_correction
        result = run_correction(
            prediction_pack_path=str(pack_path),
            risk_predictions_path=str(risk_path),
            profile=profile,
        )

        # Verify output columns
        assert "final_pred" in result.columns, "Missing final_pred"
        assert "base_fused_pred" in result.columns, "Missing base_fused_pred"
        assert "lift_applied" in result.columns, "Missing lift_applied"
        assert "reason_code" in result.columns, "Missing reason_code"

        # Verify no NaN in critical columns
        assert result["final_pred"].notna().all(), "NaN in final_pred"
        assert result["base_fused_pred"].notna().all(), "NaN in base_fused_pred"

        print(f"  [OK] End-to-end correction produced {len(result)} rows")


# ── Test 3: compute_all_metrics works after correction ─────────────────

def test_metrics_after_correction():
    """compute_all_metrics produces correct keys and non-NaN values."""
    n_days = 10
    roll = make_mock_rolling_predictions(n_days=n_days)
    risk = make_mock_risk_predictions(n_days=n_days)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        roll_path = tmp_path / "rolling.csv"
        roll.to_csv(roll_path, index=False)
        risk_path = tmp_path / "risk.csv"
        risk.to_csv(risk_path, index=False)

        pack_path = build_prediction_pack(roll_path, tmp_path)

        profile = CorrectionProfile(
            name="medium",
            spike_prob_threshold=0.60,
            max_lift_ratio=0.35,
            max_absolute_lift=350.0,
            protect_normal_hours=True,
            period_9_16_boost=1.15,
        )

        from extreme.realtime_high_spike.apply_correction import run_correction
        from scripts.evaluate_realtime_spike_correction import compute_all_metrics
        result = run_correction(
            prediction_pack_path=str(pack_path),
            risk_predictions_path=str(risk_path),
            profile=profile,
        )

        metrics = compute_all_metrics(result)

        # Required keys
        required_keys = [
            "realtime_overall_smape_floor50",
            "realtime_base_smape_floor50",
            "severe_underestimate_count",
            "false_lift_rate",
            "normal_hours_degradation",
            "total_hours",
            "lift_applied_count",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"
            assert metrics[key] is not None, f"None value for {key}"

        # Values should be in reasonable range
        assert 0 <= metrics["realtime_overall_smape_floor50"] <= 50
        assert metrics["severe_underestimate_count"] >= 0
        assert 0 <= metrics["false_lift_rate"] <= 1.0

        print(f"  [OK] Metrics computed: sMAPE={metrics['realtime_overall_smape_floor50']:.2f}, "
              f"severe={metrics['severe_underestimate_count']}")
        for k, v in sorted(metrics.items()):
            if not isinstance(v, (dict, list)):
                print(f"    {k}: {v}")


# ── Test 4: GO assessment logic ────────────────────────────────────────

def test_assess_go_recognizes_pass():
    """GO assessment correctly identifies a passing profile."""
    metrics = {
        "realtime_overall_smape_floor50": 19.00,  # <= 19.50 ✅
        "severe_underestimate_count": 60,          # <= 63 ✅
        "false_lift_rate": 0.05,                   # <= 0.10 ✅
        "normal_hours_degradation": 0.30,          # <= 0.50 ✅
    }
    verdict = assess_go(metrics)
    assert verdict["verdict"] == "GO", (
        f"Expected GO, got {verdict['verdict']}: {verdict['criteria']}"
    )


def test_assess_go_recognizes_fail():
    """GO assessment correctly identifies a failing profile."""
    metrics = {
        "realtime_overall_smape_floor50": 21.00,  # > 19.50 ❌
        "severe_underestimate_count": 80,          # > 63 ❌
        "false_lift_rate": 0.15,                   # > 0.10 ❌
        "normal_hours_degradation": 1.20,          # > 0.50 ❌
    }
    verdict = assess_go(metrics)
    assert verdict["verdict"] == "NO-GO", (
        f"Expected NO-GO, got {verdict['verdict']}"
    )
    assert not verdict["all_criteria_met"]


def test_assess_go_edge_severe():
    """GO assessment: severe <= 63 is the critical edge."""
    # Exactly at threshold
    metrics_at = {
        "realtime_overall_smape_floor50": 19.50,
        "severe_underestimate_count": 63,
        "false_lift_rate": 0.10,
        "normal_hours_degradation": 0.50,
    }
    assert assess_go(metrics_at)["verdict"] == "GO"

    # Just over threshold
    metrics_over = dict(metrics_at)
    metrics_over["severe_underestimate_count"] = 64
    assert assess_go(metrics_over)["verdict"] == "NO-GO"


# ── Test 5: Comparison table generation ────────────────────────────────

def test_comparison_table_generated():
    """Build_comparison_table produces valid markdown."""
    profile_metrics = {
        "medium": {
            "realtime_overall_smape_floor50": 19.50,
            "severe_underestimate_count": 60,
            "false_lift_rate": 0.05,
            "normal_hours_degradation": 0.30,
            "lift_applied_count": 150,
            "total_hours": 2880,
        },
        "conservative": {
            "realtime_overall_smape_floor50": 20.00,
            "severe_underestimate_count": 55,
            "false_lift_rate": 0.02,
            "normal_hours_degradation": 0.10,
            "lift_applied_count": 80,
            "total_hours": 2880,
        },
        "aggressive": {
            "realtime_overall_smape_floor50": 19.00,
            "severe_underestimate_count": 75,
            "false_lift_rate": 0.20,
            "normal_hours_degradation": 1.00,
            "lift_applied_count": 300,
            "total_hours": 2880,
        },
    }

    table = build_comparison_table(profile_metrics)

    # Should be a string with pipe characters
    assert "|" in table, "Table should contain pipe characters"
    assert "sMAPE" in table, "Table should contain metric names"
    assert "Medium" in table or "medium" in table, "Table should contain profile names"
    assert "Conservative" in table or "conservative" in table
    assert "Aggressive" in table or "aggressive" in table

    # Check baseline values
    assert f"{PHASE2_BEST_METRICS['realtime_overall_smape_floor50']}" in table
    assert f"{P31_BEST_METRICS['realtime_overall_smape_floor50']}" in table

    print(f"  [OK] Comparison table generated ({len(table)} chars)")
    print(table)


# ── Run directly ────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("build_prediction_pack_columns", test_build_prediction_pack_columns),
        ("build_prediction_pack_preserves_values", test_build_prediction_pack_preserves_values),
        ("end_to_end_correction_runs", test_end_to_end_correction_runs),
        ("metrics_after_correction", test_metrics_after_correction),
        ("assess_go_recognizes_pass", test_assess_go_recognizes_pass),
        ("assess_go_recognizes_fail", test_assess_go_recognizes_fail),
        ("assess_go_edge_severe", test_assess_go_edge_severe),
        ("comparison_table_generated", test_comparison_table_generated),
    ]

    n_pass = 0
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
            n_pass += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail > 0 else 0)
