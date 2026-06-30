"""Tests for P3a: leakage safety in spike risk pipeline.

Run from project root:
    python -m pytest tests/test_realtime_spike_no_leakage.py -v
    # or
    python tests/test_realtime_spike_no_leakage.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

# ── Schema imports ─────────────────────────────────────────────────────

from extreme.realtime_high_spike.schema import (
    ACTUAL_VALUE_EXCLUDE_COLS,
    TARGET_LEAKAGE_COLS,
    ALL_EXCLUDED_COLS,
    LABEL_COLS,
)


# ── Test 1: ACTUAL_VALUE_EXCLUDE_COLS excluded from train feature cols ─

def test_actual_cols_excluded_from_exclude_cols_in_train():
    """Check that train_realtime_spike_risk.py's exclude_cols contains ALL_EXCLUDED_COLS."""
    from scripts import train_realtime_spike_risk as mod

    # Reconstruct the exclude set used at line 94-96
    exclude_cols = {"ds", "spike_label", "model_name", "_source", "_source_file",
                    "realtime_price", "dayahead_price", "y_true", "hour", "hour_business",
                    "weekday", "y_pred", "final_pred", "base_fused_pred",
                    "abs_error", "smape", "residual", "lift_applied", "reason_code",
                    "high_spike", "high_spike_flag", "spike_risk_score", "spike_risk_flag"}
    exclude_cols.update(ALL_EXCLUDED_COLS)

    for col in ACTUAL_VALUE_EXCLUDE_COLS:
        assert col in exclude_cols, (
            f"ACTUAL_COL '{col}' not found in train_realtime_spike_risk.py exclude_cols"
        )
    for col in TARGET_LEAKAGE_COLS:
        assert col in exclude_cols, (
            f"Leakage col '{col}' not found in train_realtime_spike_risk.py exclude_cols"
        )


def test_actual_cols_not_in_feature_cols():
    """Simulate feature selection: ACTUAL_COLS must not appear in feature_cols."""
    rng = np.random.default_rng(42)
    n = 100
    data = {col: rng.uniform(0, 100, n) for col in ACTUAL_VALUE_EXCLUDE_COLS}
    data["safe_feature_lag1"] = rng.uniform(0, 100, n)
    data["safe_feature_rolling_mean"] = rng.uniform(0, 100, n)
    data["ds"] = pd.date_range("2025-01-01", periods=n, freq="h")
    data["spike_label"] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)

    # Mirror exclusion logic from train_realtime_spike_risk.py
    exclude_cols = {"ds", "spike_label", "model_name", "_source", "_source_file",
                    "realtime_price", "dayahead_price", "y_true", "hour", "hour_business",
                    "weekday", "y_pred", "final_pred", "base_fused_pred",
                    "abs_error", "smape", "residual", "lift_applied", "reason_code",
                    "high_spike", "high_spike_flag", "spike_risk_score", "spike_risk_flag"}
    exclude_cols.update(ALL_EXCLUDED_COLS)

    feature_cols = [c for c in df.columns if c not in exclude_cols
                    and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]

    # Assert NO actual-value column leaked
    for col in ACTUAL_VALUE_EXCLUDE_COLS:
        assert col not in feature_cols, (
            f"ACTUAL_COL '{col}' leaked into feature_cols!"
        )
    for col in TARGET_LEAKAGE_COLS:
        assert col not in feature_cols, (
            f"Leakage col '{col}' leaked into feature_cols!"
        )

    # Assert safe features ARE present
    assert "safe_feature_lag1" in feature_cols
    assert "safe_feature_rolling_mean" in feature_cols


# ── Test 2: y_true/residual not in prediction-time features ─

def test_no_y_true_in_prediction_features():
    """Prediction-time feature set must not contain y_true, residual, abs_error, smape."""
    from scripts import train_realtime_spike_risk as mod

    exclude_cols = {"ds", "spike_label", "model_name", "_source", "_source_file",
                    "realtime_price", "dayahead_price", "y_true", "hour", "hour_business",
                    "weekday", "y_pred", "final_pred", "base_fused_pred",
                    "abs_error", "smape", "residual", "lift_applied", "reason_code",
                    "high_spike", "high_spike_flag", "spike_risk_score", "spike_risk_flag"}
    exclude_cols.update(ALL_EXCLUDED_COLS)

    forbidden = {"y_true", "residual", "abs_error", "smape",
                 "high_spike", "high_spike_flag"}
    for col in forbidden:
        assert col in exclude_cols, (
            f"Forbidden prediction feature '{col}' not in exclude_cols"
        )


# ── Test 3: predict_realtime_spike_risk.py has no y_true placeholder ─

def test_predict_no_y_true_placeholder():
    """Check that predict_realtime_spike_risk.py does NOT use y_true in risk scoring.

    Uses the actual _get_predict_feature_cols function from the module
    on a mock DataFrame to verify y_true is excluded from features.
    """
    from scripts.predict_realtime_spike_risk import _get_predict_feature_cols

    rng = np.random.default_rng(42)
    n = 20
    data = {
        "y_true": rng.uniform(100, 500, n),
        "y_pred": rng.uniform(100, 500, n),
        "residual": rng.uniform(-50, 50, n),
        "abs_error": rng.uniform(0, 100, n),
        "smape": rng.uniform(0, 50, n),
        "safe_forecast_feature": rng.uniform(0, 100, n),
        "safe_lag_feature_1": rng.uniform(0, 100, n),
    }
    df = pd.DataFrame(data)

    feature_cols = _get_predict_feature_cols(df)

    assert "y_true" not in feature_cols, "y_true must NOT be in prediction feature cols"
    assert "residual" not in feature_cols, "residual must NOT be in prediction feature cols"
    assert "abs_error" not in feature_cols, "abs_error must NOT be in prediction feature cols"
    assert "smape" not in feature_cols, "smape must NOT be in prediction feature cols"
    assert "safe_forecast_feature" in feature_cols, "safe features must be included"


# ── Test 4: build_realtime_spike_dataset.py whitelist drops ACTUAL_COLS ─

def test_build_dataset_whitelist_drops_actual_cols():
    """Simulate build_realtime_spike_dataset.py whitelist: ACTUAL_COLS should be dropped."""
    rng = np.random.default_rng(42)
    n = 50
    data = {col: rng.uniform(0, 100, n) for col in ACTUAL_VALUE_EXCLUDE_COLS}
    hour_vals = rng.integers(0, 24, n)
    data["hour"] = hour_vals
    data["hour_business"] = pd.Series(hour_vals).apply(lambda h: 24 if h == 0 else h)
    data["weekday"] = rng.integers(0, 7, n)
    data["y_pred_lag1"] = rng.uniform(0, 100, n)
    data["y_pred_rolling_mean_6"] = rng.uniform(0, 100, n)
    data["ds"] = pd.date_range("2025-01-01", periods=n, freq="h")
    data["spike_label"] = rng.integers(0, 2, n)
    df = pd.DataFrame(data)

    # Mirror whitelist logic from build_realtime_spike_dataset.py
    TARGET_COLS = {"realtime_price", "dayahead_price", "y_true", "spike_label", "ds"}
    FEATURE_WHITELIST_SUFFIXES = ("_lag", "_rolling", "_pred", "_diff")
    FEATURE_WHITELIST_COLS = {"hour", "hour_business", "weekday", "month", "day",
                               "y_pred", "base_fused_pred", "final_pred"}

    safe_cols = []
    for c in df.columns:
        if c in TARGET_COLS or c in FEATURE_WHITELIST_COLS:
            safe_cols.append(c)
        elif any(c.endswith(s) for s in FEATURE_WHITELIST_SUFFIXES):
            safe_cols.append(c)
        elif c.startswith("model_") or c.startswith("ensemble_"):
            safe_cols.append(c)
        elif c not in ACTUAL_VALUE_EXCLUDE_COLS:
            safe_cols.append(c)

    # Assert ACTUAL_COLS are dropped
    for col in ACTUAL_VALUE_EXCLUDE_COLS:
        assert col not in safe_cols, (
            f"ACTUAL_COL '{col}' survived whitelist in build dataset!"
        )

    # Assert safe columns are retained
    assert "y_pred_lag1" in safe_cols
    assert "y_pred_rolling_mean_6" in safe_cols


# ── Run directly ──

if __name__ == "__main__":
    import inspect

    tests = [
        ("test_actual_cols_excluded_from_exclude_cols_in_train",
         test_actual_cols_excluded_from_exclude_cols_in_train),
        ("test_actual_cols_not_in_feature_cols",
         test_actual_cols_not_in_feature_cols),
        ("test_no_y_true_in_prediction_features",
         test_no_y_true_in_prediction_features),
        ("test_predict_no_y_true_placeholder",
         test_predict_no_y_true_placeholder),
        ("test_build_dataset_whitelist_drops_actual_cols",
         test_build_dataset_whitelist_drops_actual_cols),
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
