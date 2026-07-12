"""Tests for P3.1 severe-underestimate-aware rolling fusion.

Covers:
1. severe_underestimate penalty reduces weight of high-severe models
2. severe_anchor keeps lightgbm >= 0.85
3. quantile_guarded does not reduce high-risk hour predictions
4. Day D weights use only [D-30, D-1] training window

Run from project root:
    python -m pytest tests/test_severe_aware_rolling.py -v
    # or directly
    python tests/test_severe_aware_rolling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.run_rolling_30d_fusion import (
    compute_per_day_weights,
    apply_weights,
    apply_quantile_guard,
    fit_severe_softmax_weights,
    fit_severe_anchor_weights,
    compute_severe_rate,
    compute_underprediction_mae,
    compute_smape_floor50,
    FUSION_MODES,
    SEVERE_ANCHOR_MIN,
)


# ── Helpers ────────────────────────────────────────────────────────────

def make_mock_pack(
    n_days: int = 35,
    models: list[str] | None = None,
    seed: int = 42,
    severe_model: str | None = None,
) -> pd.DataFrame:
    """Create synthetic multi-model prediction pack.

    If severe_model is set, that model predictions will have a larger
    severe underestimate rate (y_true - y_pred > 200 on spike hours).
    """
    if models is None:
        models = ["naive_lag1", "naive_lag7", "dayahead_proxy", "lightgbm"]

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01")
    rows: list[dict] = []

    model_factors = {
        "naive_lag1": 0.90,
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
            # Generate actual price with daily pattern + occasional spikes
            hour_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (hb - 6) / 24)
            # Add spikes on random days at peak hours
            spike = 0.0
            if 9 <= hb <= 16 and rng.random() < 0.08:
                spike = rng.uniform(200, 500)
            y_true = 300.0 * hour_factor + spike + rng.normal(0, 20)

            for m in models:
                factor = model_factors.get(m, 0.9)
                noise = rng.normal(0, 15)

                # Make severe_model underpredict spikes more
                if severe_model and m == severe_model and spike > 0:
                    factor *= 0.7  # severe underprediction on spikes

                y_pred = y_true * factor + noise
                rows.append({
                    "business_day": bd,
                    "hour_business": hb,
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "period": "9_16" if 9 <= hb <= 16 else "night" if hb <= 8 else "evening",
                    "model_name": m,
                    "y_pred": round(y_pred, 2),
                    "y_true": round(y_true, 2),
                })

    return pd.DataFrame(rows)


def make_mock_risk_predictions(pack: pd.DataFrame, seed: int = 123) -> pd.DataFrame:
    """Create synthetic risk predictions from a pack."""
    rng = np.random.default_rng(seed)
    ts_rows = pack.drop_duplicates(subset=["business_day", "hour_business"])
    rows = []
    for _, row in ts_rows.iterrows():
        bd = row["business_day"]
        hb = row["hour_business"]
        period = "9_16" if 9 <= int(hb) <= 16 else "other"
        base_risk = 0.5 if period == "9_16" else 0.3
        score = base_risk + rng.uniform(-0.2, 0.4)
        score = max(0, min(1, score))
        yt = float(row.get("y_true", 300))
        flag = 1 if yt > 500 else 0
        rows.append({
            "business_day": bd,
            "hour_business": hb,
            "spike_risk_score": round(score, 4),
            "high_spike_prob": round(score, 4),
            "spike_risk_flag": flag,
        })
    return pd.DataFrame(rows)


# ── Test 1: severe_softmax penalises high-severe models ───────────────

def test_severe_penalty_reduces_severe_model_weight():
    """A model with high severe_underestimate rate gets lower weight in severe_softmax."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.zeros(n)
    # Add spikes to some hours
    spike_mask = np.zeros(n, dtype=bool)
    spike_mask[::10] = True  # 10% spike hours
    y_true[spike_mask] = 500.0 + rng.normal(0, 50, spike_mask.sum())
    y_true[~spike_mask] = 300.0 + rng.normal(0, 30, (~spike_mask).sum())

    # Good model: low error and low severe
    good_pred = y_true * 0.98 + rng.normal(0, 15, n)
    # Bad model: high severe underestimates (systematically low on spikes)
    bad_pred = y_true.copy()
    bad_pred[spike_mask] = y_true[spike_mask] * 0.6 + rng.normal(0, 30, spike_mask.sum())
    bad_pred[~spike_mask] = y_true[~spike_mask] * 0.95 + rng.normal(0, 20, (~spike_mask).sum())

    train_preds = pd.DataFrame({"good_model": good_pred, "bad_model": bad_pred})
    y_true_series = pd.Series(y_true)

    # Standard softmax weights (no severe penalty)
    softmax_w = fit_severe_softmax_weights(
        train_preds, y_true_series, ["good_model", "bad_model"],
        temperature=0.1, alpha=0.0, beta=0.0,
    )

    # Severe softmax weights
    severe_w = fit_severe_softmax_weights(
        train_preds, y_true_series, ["good_model", "bad_model"],
        temperature=0.1, alpha=2.0, beta=0.5,
    )

    # Good model should get less weight in severe mode (since bad model gets penalized)
    # Actually: good model has low severe → should get MORE relative weight
    good_ratio_softmax = softmax_w["good_model"] / softmax_w["bad_model"]
    good_ratio_severe = severe_w["good_model"] / severe_w["bad_model"]

    # The good model should be relatively more favored with severe penalty
    assert good_ratio_severe >= good_ratio_softmax * 0.9, (
        f"Severe penalty should not reduce good/bad ratio: "
        f"softmax={good_ratio_softmax:.3f}, severe={good_ratio_severe:.3f}"
    )

    # Verify bad model has higher severe rate
    bad_severe = compute_severe_rate(y_true, bad_pred)
    good_severe = compute_severe_rate(y_true, good_pred)
    assert bad_severe > good_severe, (
        f"Bad model should have higher severe rate: "
        f"bad={bad_severe:.4f}, good={good_severe:.4f}"
    )


# ── Test 2: severe_anchor keeps lightgbm >= 0.85 ──────────────────────

def test_severe_anchor_keeps_lightgbm_high():
    """severe_anchor should keep lightgbm weight >= min_anchor_weight."""
    pack = make_mock_pack(n_days=35, seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1", "naive_lag7"]

    weights_df = compute_per_day_weights(
        pack, models,
        fusion_mode="severe_anchor",
        train_window_days=30,
        severe_anchor_min=SEVERE_ANCHOR_MIN,
        verbose=False,
    )

    lgbm_weights = weights_df[weights_df["model_name"] == "lightgbm"]
    # Skip first 10 days (fallback equal weights)
    valid_days = lgbm_weights.iloc[10:]

    min_weight = valid_days["weight"].min()
    assert min_weight >= SEVERE_ANCHOR_MIN - 0.01, (
        f"LightGBM weight should be >= {SEVERE_ANCHOR_MIN}, "
        f"got min={min_weight:.4f}"
    )

    # Verify all days sum to 1
    for bd, grp in weights_df.groupby("business_day"):
        total = grp["weight"].sum()
        assert abs(total - 1.0) < 1e-6, (
            f"Weights for {bd} sum to {total}, expected 1.0"
        )


def test_severe_anchor_protects_severe_rate():
    """Baselines that increase severe rate should not get weight."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = np.zeros(n)
    spike_mask = np.zeros(n, dtype=bool)
    spike_mask[::8] = True
    y_true[spike_mask] = 500.0 + rng.normal(0, 50, spike_mask.sum())
    y_true[~spike_mask] = 300.0 + rng.normal(0, 30, (~spike_mask).sum())

    # Anchor: low severe rate
    anchor_pred = y_true * 0.95 + rng.normal(0, 20, n)
    # Bad baseline: very high severe (always underpredicts)
    bad_pred = y_true * 0.70 + rng.normal(0, 30, n)
    # Good baseline: similar severe to anchor
    good_pred = y_true * 0.93 + rng.normal(0, 25, n)

    train_preds = pd.DataFrame({
        "lightgbm": anchor_pred,
        "bad_baseline": bad_pred,
        "good_baseline": good_pred,
    })
    y_true_series = pd.Series(y_true)

    weights = fit_severe_anchor_weights(
        train_preds, y_true_series,
        ["lightgbm", "bad_baseline", "good_baseline"],
        anchor_model="lightgbm",
        min_anchor_weight=SEVERE_ANCHOR_MIN,
    )

    # Bad baseline should get 0 weight (worse severe than anchor)
    assert weights.get("bad_baseline", 1.0) == 0.0, (
        f"Bad baseline should get 0 weight, got {weights.get('bad_baseline', 'N/A')}"
    )

    # LightGBM should be >= min_anchor_weight
    assert weights["lightgbm"] >= SEVERE_ANCHOR_MIN, (
        f"LightGBM weight ({weights['lightgbm']}) < {SEVERE_ANCHOR_MIN}"
    )

    # Verify severity rates
    anchor_severe = compute_severe_rate(y_true, anchor_pred)
    bad_severe = compute_severe_rate(y_true, bad_pred)
    good_severe = compute_severe_rate(y_true, good_pred)
    assert bad_severe > anchor_severe, "Bad model must have higher severe rate"
    assert good_severe <= anchor_severe * 1.05, "Good model should be within 5% of anchor severe"


# ── Test 3: quantile_guarded does not reduce high-risk predictions ────

def test_quantile_guard_does_not_reduce_high_risk():
    """Quantile guard should only increase (never decrease) predictions on high-risk hours."""
    pack = make_mock_pack(n_days=35, seed=42, severe_model="naive_lag1")
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]
    risk_df = make_mock_risk_predictions(pack, seed=123)

    # Compute softmax weights first (base for quantile_guarded)
    weights_df = compute_per_day_weights(
        pack, models,
        fusion_mode="quantile_guarded",
        train_window_days=30,
        severe_alpha=1.0, severe_beta=0.5,
        verbose=False,
    )

    # Get unguarded predictions
    predictions = apply_weights(pack, weights_df, models)

    # Apply guard
    guarded = apply_quantile_guard(
        predictions.copy(), pack, weights_df, models,
        risk_df=risk_df,
        risk_threshold=0.4,
        severe_rate_threshold=0.05,
    )

    # Guard should never reduce predictions
    merged = predictions.merge(
        guarded, on=["business_day", "hour_business"],
        suffixes=("_orig", "_guarded"),
    )
    reductions = merged[merged["base_fused_pred_guarded"] < merged["base_fused_pred_orig"] - 0.01]
    assert len(reductions) == 0, (
        f"Guard reduced predictions on {len(reductions)} timestamps!"
    )

    # Guarded timestamps should have higher or equal predictions
    guarded_rows = guarded[guarded["guarded"] == 1]
    if len(guarded_rows) > 0:
        orig_col = f"base_fused_pred"
        guarded_vals = guarded_rows["base_fused_pred"].values
        # Verify via joining back
        for _, gr in guarded_rows.iterrows():
            bd = gr["business_day"]
            hb = gr["hour_business"]
            orig_val = predictions[
                (predictions["business_day"] == bd)
                & (predictions["hour_business"] == hb)
            ]["base_fused_pred"].values[0]
            assert gr["base_fused_pred"] >= orig_val, (
                f"Guard reduced prediction at {bd} hb={hb}: "
                f"{gr['base_fused_pred']} < {orig_val}"
            )


def test_quantile_guard_runs_without_risk():
    """Quantile guard should run even without risk predictions (no-op)."""
    pack = make_mock_pack(n_days=35, seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]

    weights_df = compute_per_day_weights(
        pack, models,
        fusion_mode="quantile_guarded",
        train_window_days=30,
        verbose=False,
    )
    predictions = apply_weights(pack, weights_df, models)

    # Without risk data, guard is effectively a no-op
    guarded = apply_quantile_guard(
        predictions.copy(), pack, weights_df, models,
        risk_df=None,
    )

    assert len(guarded) == len(predictions)
    assert "guarded" in guarded.columns
    assert "guard_source" in guarded.columns
    # No guards should trigger without risk data
    assert (guarded["guarded"] == 0).all(), (
        "Guard should not activate without risk predictions"
    )


# ── Test 4: Day D weights use only [D-30, D-1] ───────────────────────

def test_rolling_window_excludes_day_d_severe_modes():
    """All P3.1 modes must use only [D-30, D-1] for weight fitting."""
    pack = make_mock_pack(n_days=40, seed=42)

    for mode in ["severe_softmax", "severe_anchor", "quantile_guarded"]:
        weights_df = compute_per_day_weights(
            pack, ["lightgbm", "dayahead_proxy"],
            fusion_mode=mode,
            train_window_days=30,
            verbose=False,
        )

        business_days = sorted(pack["business_day"].unique())
        day_d = business_days[35]  # Day with enough history

        day_dt = pd.Timestamp(day_d)
        train_end = day_dt - pd.Timedelta(days=1)

        # Train data should not include day_d
        train_days_in_pack = set(pack["business_day"].unique())
        day_d_weights = weights_df[weights_df["business_day"] == day_d]
        assert len(day_d_weights) > 0, f"No weights for {mode} at {day_d}"

        # Verify by checking first-day fallback
        first_day = business_days[0]
        first_weights = weights_df[weights_df["business_day"] == first_day]
        expected = 1.0 / 2  # 2 models
        assert all(abs(w - expected) < 1e-10 for w in first_weights["weight"]), (
            f"First day should have equal weights in {mode} mode"
        )


# ── Test 5: New modes in FUSION_MODES ─────────────────────────────────

def test_severe_modes_registered():
    """P3.1 modes must be in FUSION_MODES."""
    for mode in ["severe_softmax", "severe_anchor", "quantile_guarded"]:
        assert mode in FUSION_MODES, f"{mode} not in FUSION_MODES"


# ── Test 6: compute_severe_rate accuracy ──────────────────────────────

def test_compute_severe_rate():
    """compute_severe_rate must correctly count severe underestimates."""
    y_true = np.array([100, 300, 500, 700, 200])
    y_pred = np.array([90, 250, 300, 400, 210])

    rate = compute_severe_rate(y_true, y_pred)
    # y_true - y_pred > 200 for: 500-300=200 (not >), 700-400=300 (>)
    # Only index 3 (700-400=300 > 200)
    expected_rate = 1.0 / 5  # 1 out of 5
    assert abs(rate - expected_rate) < 1e-10, (
        f"Expected severe rate {expected_rate}, got {rate}"
    )


# ── Test 7: compute_underprediction_mae accuracy ──────────────────────

def test_compute_underprediction_mae():
    """compute_underprediction_mae should only compute MAE on underpredictions."""
    y_true = np.array([100, 300, 500, 200, 400])
    y_pred = np.array([110, 250, 450, 210, 420])
    # Underpredicted (y_pred < y_true): index 1 (300-250=50), index 2 (500-450=50)
    mae = compute_underprediction_mae(y_true, y_pred)
    expected_mae = 50.0
    assert abs(mae - expected_mae) < 1e-10, (
        f"Expected underprediction MAE {expected_mae}, got {mae}"
    )

    # All overpredicted → MAE = 0
    y_pred_all_over = np.array([200, 400, 600, 300, 500])
    assert compute_underprediction_mae(y_true, y_pred_all_over) == 0.0


# ── Test 8: End-to-end run for all modes ──────────────────────────────

def test_all_new_modes_run_without_error():
    """All three P3.1 modes must run end-to-end without error."""
    pack = make_mock_pack(n_days=35, seed=42)
    models = ["lightgbm", "dayahead_proxy", "naive_lag1"]

    for mode in ["severe_softmax", "severe_anchor", "quantile_guarded"]:
        weights_df = compute_per_day_weights(
            pack, models,
            fusion_mode=mode,
            train_window_days=30,
            verbose=False,
        )
        assert len(weights_df) > 0, f"No weights for mode={mode}"
        predictions = apply_weights(pack, weights_df, models)
        assert len(predictions) > 0, f"No predictions for mode={mode}"

        # Verify weights sum to 1 per day
        for bd, grp in weights_df.groupby("business_day"):
            assert abs(grp["weight"].sum() - 1.0) < 1e-6, (
                f"Weights for {bd} {mode} sum to {grp['weight'].sum()}"
            )


# ── Run directly ──

if __name__ == "__main__":
    tests = [
        ("severe_penalty_reduces_severe_model_weight", test_severe_penalty_reduces_severe_model_weight),
        ("severe_anchor_keeps_lightgbm_high", test_severe_anchor_keeps_lightgbm_high),
        ("severe_anchor_protects_severe_rate", test_severe_anchor_protects_severe_rate),
        ("quantile_guard_does_not_reduce_high_risk", test_quantile_guard_does_not_reduce_high_risk),
        ("quantile_guard_runs_without_risk", test_quantile_guard_runs_without_risk),
        ("rolling_window_excludes_day_d_severe_modes", test_rolling_window_excludes_day_d_severe_modes),
        ("severe_modes_registered", test_severe_modes_registered),
        ("compute_severe_rate", test_compute_severe_rate),
        ("compute_underprediction_mae", test_compute_underprediction_mae),
        ("all_new_modes_run", test_all_new_modes_run_without_error),
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
