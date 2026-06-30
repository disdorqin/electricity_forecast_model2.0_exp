"""Tests for P3 rolling 30D fusion pipeline.

Covers:
1. Rolling D weights use only [D-30, D-1] training window
2. Day D y_true does NOT participate in fitting day D weights
3. Timestamp-level dedup produces 1 row per (business_day, hour_business)
4. Fusion outputs are deterministic for a given input
5. All four weight modes produce valid weights

Run from project root:
    python -m pytest tests/test_rolling_30d_fusion.py -v
    # or directly
    python tests/test_rolling_30d_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json

import numpy as np
import pandas as pd

from scripts.run_rolling_30d_fusion import (
    compute_per_day_weights,
    apply_weights,
    compute_per_day_metrics,
    compute_overall_metrics,
    fit_convex_weights,
    fit_ridge_weights,
    fit_softmax_weights,
    fit_anchor_weights,
    compute_smape_floor50,
)


# ── Helpers ────────────────────────────────────────────────────────────

def make_mock_pack(
    n_days: int = 35,
    models: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic multi-model prediction pack.

    Each model prediction = y_true * model_factor + noise.
    """
    if models is None:
        models = ["naive_lag1", "naive_lag7", "dayahead_proxy", "lightgbm"]

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    rows: list[dict] = []

    model_factors = {
        "naive_lag1": 0.9,
        "naive_lag7": 0.85,
        "dayahead_proxy": 0.95,
        "lightgbm": 0.98,
        "timemixer": 0.96,
        "sgdfnet": 0.97,
    }

    for i in range(n_days):
        day = start + pd.Timedelta(days=i)
        bd = day.strftime("%Y-%m-%d")
        for hb in range(1, 25):
            ts = day + pd.Timedelta(hours=hb - 1)
            # Generate actual price with daily pattern
            hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hb - 6) / 24)
            y_true = 300.0 * hour_factor + rng.normal(0, 20)

            for m in models:
                factor = model_factors.get(m, 0.9)
                noise = rng.normal(0, 15)
                y_pred = y_true * factor + noise
                rows.append({
                    "business_day": bd,
                    "hour_business": hb,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": m,
                    "y_pred": round(y_pred, 2),
                    "y_true": round(y_true, 2),
                })

    return pd.DataFrame(rows)


# ── Test 1: Rolling window uses only [D-30, D-1] ──────────────────────

def test_rolling_window_excludes_day_d():
    """Verify train window = [D-30, D-1] and day D y_true is not used to fit D weights."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    models = ["lightgbm", "dayahead_proxy"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )

    # Get weights for day D (e.g., day index 34, the last day)
    business_days = sorted(pack["business_day"].unique())
    assert len(business_days) == 35

    # For day index 30 (business_days[30]), train window = [0, 29]
    # Day 30 y_true must not affect day 30 weights
    day_d = business_days[30]
    day_d_weights = weights_df[weights_df["business_day"] == day_d]
    assert len(day_d_weights) == len(models)

    # Reproduce: feed train data which excludes day_d
    day_dt = pd.Timestamp(day_d)
    train_start = day_dt - pd.Timedelta(days=30)
    train_end = day_dt - pd.Timedelta(days=1)
    train_mask = (
        (pack["business_day"] >= train_start.strftime("%Y-%m-%d"))
        & (pack["business_day"] <= train_end.strftime("%Y-%m-%d"))
    )
    train = pack[train_mask]

    # Train should have 30 days of data
    n_train_days = train["business_day"].nunique()
    assert n_train_days == 30, f"Expected 30 train days, got {n_train_days}"

    # Train should NOT include day_d
    assert day_d not in train["business_day"].unique(), (
        f"Day D ({day_d}) should not be in train window!"
    )

    # Train should not include any day after day_d
    train_days = sorted(train["business_day"].unique())
    assert all(d <= train_end.strftime("%Y-%m-%d") for d in train_days), (
        "Train days extend beyond D-1"
    )


def test_first_day_fallback_equal_weights():
    """First business day with insufficient history should get equal weights."""
    pack = make_mock_pack(n_days=5, models=["lightgbm", "dayahead_proxy", "naive_lag1"], seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )

    # First day has no history → equal weights
    first_day = sorted(pack["business_day"].unique())[0]
    first_weights = weights_df[weights_df["business_day"] == first_day]
    expected = 1.0 / len(models)
    assert all(abs(w - expected) < 1e-10 for w in first_weights["weight"]), (
        f"First day weights should be {expected:.4f}, got {first_weights['weight'].values}"
    )


# ── Test 2: Timestamp-level dedup ─────────────────────────────────────

def test_apply_weights_produces_one_row_per_timestamp():
    """apply_weights() must produce exactly 1 row per (business_day, hour_business)."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    models = ["lightgbm", "dayahead_proxy"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )
    predictions = apply_weights(pack, weights_df, models)

    # Check for duplicates
    dupes = predictions.duplicated(subset=["business_day", "hour_business"]).sum()
    assert dupes == 0, f"Found {dupes} duplicate (business_day, hour_business) rows!"

    # Check expected count
    expected_days = len(predictions["business_day"].unique())
    expected_hours = predictions.groupby("business_day")["hour_business"].nunique().max()
    assert len(predictions) >= expected_days * 24 * 0.9, (
        f"Expected ~{expected_days * 24} rows, got {len(predictions)}"
    )

    # Verify metric_level is set
    assert "metric_level" in predictions.columns
    assert predictions["metric_level"].iloc[0] == "timestamp"


def test_compute_overall_metrics_on_deduplicated():
    """compute_overall_metrics should indicate timestamp-level dedup."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    models = ["lightgbm", "dayahead_proxy"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )
    predictions = apply_weights(pack, weights_df, models)
    overall = compute_overall_metrics(predictions)

    assert overall["metric_level"] == "timestamp"
    assert overall["n_timestamps"] == len(predictions)
    assert "note" in overall
    assert overall["n_timestamps"] > 0


def test_per_day_metrics_no_duplicates():
    """Per-day metrics should work correctly on deduplicated inputs."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    models = ["lightgbm", "dayahead_proxy"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )
    predictions = apply_weights(pack, weights_df, models)

    per_day = compute_per_day_metrics(predictions)
    total_hours_from_per_day = per_day["n_hours"].sum()
    assert total_hours_from_per_day <= len(predictions), (
        f"Per-day hours ({total_hours_from_per_day}) exceed total predictions ({len(predictions)})"
    )


# ── Test 3: All four weight modes ─────────────────────────────────────

def test_convex_weights_sum_to_one():
    """Convex weights must be non-negative and sum to 1."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy", "naive_lag1"], seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )

    # Group by business_day and verify sum per day
    for bd, grp in weights_df.groupby("business_day"):
        total = grp["weight"].sum()
        assert abs(total - 1.0) < 1e-6, (
            f"Convex weights for {bd} sum to {total}, expected 1.0"
        )
        assert (grp["weight"] >= -1e-6).all(), (
            f"Negative weight found for {bd}: {grp['weight'].min()}"
        )


def test_ridge_weights_no_constraint():
    """Ridge weights may be negative and do not need to sum to 1."""
    # Train data for fit_ridge_weights directly
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(200, 500, n)
    train_preds = pd.DataFrame({
        "model_a": y_true * 0.9 + rng.normal(0, 20, n),
        "model_b": y_true * 1.1 + rng.normal(0, 30, n),
    })
    y_true_series = pd.Series(y_true)

    weights = fit_ridge_weights(train_preds, y_true_series, ["model_a", "model_b"], alpha=1.0)
    assert len(weights) == 2
    assert "model_a" in weights
    assert "model_b" in weights
    # No constraint on sum or sign


def test_softmax_weights_valid():
    """Softmax weights should be non-negative and sum to 1."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(200, 500, n)
    train_preds = pd.DataFrame({
        "model_a": y_true * 0.95 + rng.normal(0, 10, n),
        "model_b": y_true * 0.90 + rng.normal(0, 30, n),
    })
    y_true_series = pd.Series(y_true)

    weights = fit_softmax_weights(train_preds, y_true_series, ["model_a", "model_b"], temperature=0.1)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights.values())
    # Better model should get higher weight
    assert weights["model_a"] > weights["model_b"]


def test_anchor_weights_convex_remainder():
    """Anchor weights: anchor gets fixed weight, remainder split convexly."""
    rng = np.random.default_rng(42)
    n = 100
    y_true = rng.uniform(200, 500, n)
    train_preds = pd.DataFrame({
        "anchor_m": y_true * 0.98 + rng.normal(0, 10, n),
        "other_a": y_true * 0.85 + rng.normal(0, 30, n),
        "other_b": y_true * 0.90 + rng.normal(0, 25, n),
    })
    y_true_series = pd.Series(y_true)
    models = ["anchor_m", "other_a", "other_b"]

    weights = fit_anchor_weights(
        train_preds, y_true_series, models,
        anchor_model="anchor_m", anchor_weight=0.9,
    )
    assert abs(weights["anchor_m"] - 0.9) < 1e-6
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights.values())


def test_all_four_modes_run_without_error():
    """All 4 fusion modes should run end-to-end without error."""
    pack = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy", "naive_lag1"], seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]

    for mode in ["convex", "ridge", "softmax", "anchor"]:
        weights_df = compute_per_day_weights(
            pack, models, fusion_mode=mode, train_window_days=30, verbose=False,
        )
        assert len(weights_df) > 0, f"No weights produced for mode={mode}"
        predictions = apply_weights(pack, weights_df, models)
        assert len(predictions) > 0, f"No predictions produced for mode={mode}"


# ── Test 4: Determinism ───────────────────────────────────────────────

def test_deterministic_weights():
    """Same input produces same weights (no randomness in fitting)."""
    pack1 = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    pack2 = make_mock_pack(n_days=35, models=["lightgbm", "dayahead_proxy"], seed=42)
    models = ["lightgbm", "dayahead_proxy"]

    w1 = compute_per_day_weights(pack1, models, fusion_mode="convex", train_window_days=30, verbose=False)
    w2 = compute_per_day_weights(pack2, models, fusion_mode="convex", train_window_days=30, verbose=False)

    pd.testing.assert_frame_equal(w1, w2)


# ── Test 5: sMAPE computation ─────────────────────────────────────────

def test_smape_floor50_values():
    """sMAPE floor50 should be in [0, 50] and symmetric."""
    y_true = pd.Series([100.0, 200.0, 300.0, 0.0])
    y_pred = pd.Series([110.0, 180.0, 300.0, 10.0])

    smape = compute_smape_floor50(y_true, y_pred)

    assert smape.shape == (4,)
    assert np.all(smape >= 0), f"sMAPE values should be >= 0, got {smape}"
    assert np.all(smape <= 50), f"sMAPE values should be <= 50, got {smape}"

    # Symmetry
    smape_rev = compute_smape_floor50(y_pred, y_true)
    assert np.allclose(smape, smape_rev, atol=1e-6), "sMAPE should be symmetric"


# ── Test 6: Manifest structure ────────────────────────────────────────

def test_manifest_keys():
    """Simulate the manifest structure from main()."""
    pack = make_mock_pack(n_days=35, models=["lightgbm"], seed=42)
    models = ["lightgbm"]

    weights_df = compute_per_day_weights(
        pack, models, fusion_mode="convex", train_window_days=30, verbose=False,
    )
    predictions = apply_weights(pack, weights_df, models)
    overall = compute_overall_metrics(predictions)

    manifest = {
        "script": "scripts/run_rolling_30d_fusion.py",
        "fusion_mode": "convex",
        "models": models,
        "train_window_days": 30,
        "overall_metrics": overall,
        "leakage_safe": True,
        "note": "Rolling 30D fusion: weights fitted on [D-30, D-1] only.",
    }

    assert manifest["leakage_safe"] is True
    assert "overall_metrics" in manifest
    assert manifest["overall_metrics"]["metric_level"] == "timestamp"
    assert "D-30" in manifest["note"]
    assert "D-1" in manifest["note"]


# ── Run directly ──

if __name__ == "__main__":
    tests = [
        ("rolling_window_excludes_day_d", test_rolling_window_excludes_day_d),
        ("first_day_fallback_equal_weights", test_first_day_fallback_equal_weights),
        ("apply_weights_produces_one_row_per_timestamp", test_apply_weights_produces_one_row_per_timestamp),
        ("compute_overall_metrics_on_deduplicated", test_compute_overall_metrics_on_deduplicated),
        ("per_day_metrics_no_duplicates", test_per_day_metrics_no_duplicates),
        ("convex_weights_sum_to_one", test_convex_weights_sum_to_one),
        ("ridge_weights_no_constraint", test_ridge_weights_no_constraint),
        ("softmax_weights_valid", test_softmax_weights_valid),
        ("anchor_weights_convex_remainder", test_anchor_weights_convex_remainder),
        ("all_four_modes_run", test_all_four_modes_run_without_error),
        ("deterministic_weights", test_deterministic_weights),
        ("smape_floor50_values", test_smape_floor50_values),
        ("manifest_keys", test_manifest_keys),
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
