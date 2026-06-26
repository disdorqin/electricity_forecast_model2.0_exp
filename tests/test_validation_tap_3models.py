# -*- coding: utf-8 -*-
"""Mock tests for LightGBM/SGDFNet/TimesFM validation tap.

These tests verify the correctness of the 3-model validation tap pipeline
without running real model inference. They test:
  1. 3x10 block spec generation
  2. Learner fold splitting
  3. Long-table column consistency
  4. Runtime report format
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_dummy_predictions(predict_date: str, model_name: str, n_days: int = 30) -> pd.DataFrame:
    """Create dummy 24-hour predictions for `n_days` days.

    Returns a DataFrame with: ds, target_day, y_pred, pred_y.
    """
    D = pd.Timestamp(predict_date).date()
    rows = []
    for day_offset in range(n_days, 0, -1):
        d = D - pd.Timedelta(days=day_offset)  # D-30, D-29, ..., D-1
        for hour in range(1, 25):  # business hours 1-24
            # Python Timestamp: hour 0-23; business hour 24 -> next day 00:00
            if hour == 24:
                ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=0) + pd.Timedelta(days=1)
            else:
                ts = pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=hour)
            rows.append({
                "ds": ts,
                "target_day": d.strftime("%Y-%m-%d"),
                "y_pred": float(100.0 + hour * 10),
                "pred_y": float(100.0 + hour * 10),
                "model_name": model_name,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════
# Test 1: 3x10 Block Spec Generation
# ══════════════════════════════════════════════════════════════════════

def test_generate_3x10_block_specs():
    """Verify generate_3x10_block_specs produces correct 3 blocks."""
    from pipelines.validation_tap_light_sgdf_timesfm import generate_3x10_block_specs

    predict_date = "2026-02-01"
    D = pd.Timestamp(predict_date).date()
    blocks = generate_3x10_block_specs(predict_date)

    assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"

    # Block 0: train_end = D-31, predict D-30 ~ D-21
    b0 = blocks[0]
    assert b0["block_id"] == 0
    assert date.fromisoformat(b0["train_end"]) == D - pd.Timedelta(days=31)
    assert date.fromisoformat(b0["test_start"]) == D - pd.Timedelta(days=30)
    assert date.fromisoformat(b0["test_end"]) == D - pd.Timedelta(days=21)

    # Block 1: train_end = D-21, predict D-20 ~ D-11
    b1 = blocks[1]
    assert b1["block_id"] == 1
    assert date.fromisoformat(b1["train_end"]) == D - pd.Timedelta(days=21)
    assert date.fromisoformat(b1["test_start"]) == D - pd.Timedelta(days=20)
    assert date.fromisoformat(b1["test_end"]) == D - pd.Timedelta(days=11)

    # Block 2: train_end = D-11, predict D-10 ~ D-1
    b2 = blocks[2]
    assert b2["block_id"] == 2
    assert date.fromisoformat(b2["train_end"]) == D - pd.Timedelta(days=11)
    assert date.fromisoformat(b2["test_start"]) == D - pd.Timedelta(days=10)
    assert date.fromisoformat(b2["test_end"]) == D - pd.Timedelta(days=1)

    print("PASS: test_generate_3x10_block_specs")


# ══════════════════════════════════════════════════════════════════════
# Test 2: Learner Fold Splitting
# ══════════════════════════════════════════════════════════════════════

def test_split_month_predictions_to_learner_folds():
    """Verify split_month_predictions_to_learner_folds maps 30 days to 10 folds."""
    from pipelines.validation_tap_light_sgdf_timesfm import (
        split_month_predictions_to_learner_folds,
        build_date_to_fold_map,
    )

    predict_date = "2026-02-01"
    D = pd.Timestamp(predict_date).date()

    # Create dummy 30-day predictions (D-30 ~ D-1)
    dummy = _make_dummy_predictions(predict_date, "test_model", n_days=30)

    # Split into learner folds
    result = split_month_predictions_to_learner_folds(dummy, predict_date)

    # Check all 10 folds present
    fold_ids = sorted(result["tap_fold_id"].unique().tolist())
    assert fold_ids == list(range(10)), f"Expected folds 0-9, got {fold_ids}"

    # Check 30 days of predictions → 720 rows (30 days × 24 hours)
    assert len(result) == 720, f"Expected 720 rows (30d × 24h), got {len(result)}"

    # Verify age_block mapping
    for fold_id in range(10):
        fold_df = result[result["tap_fold_id"] == fold_id]
        assert all(fold_df["age_block"] == 9 - fold_id), \
            f"age_block mismatch for fold {fold_id}"

    # Verify age_days (should be between 1 and 30 for D-30 to D-1)
    assert all(result["age_days"].between(1, 30)), \
        f"age_days out of range [1,30]: {result['age_days'].describe()}"
    # Verify age_days for each fold: fold 0 has days 28-30, fold 9 has days 1-3
    for fold_id in range(10):
        fold_df = result[result["tap_fold_id"] == fold_id]
        expected_min = 30 - 3 * fold_id - 2
        expected_max = 30 - 3 * fold_id
        assert fold_df["age_days"].min() >= expected_min, \
            f"fold {fold_id}: age_days min < {expected_min}"
        assert fold_df["age_days"].max() <= expected_max, \
            f"fold {fold_id}: age_days max > {expected_max}"

    # Verify horizon_day (1, 2, or 3 within each fold)
    for fold_id in range(10):
        fold_df = result[result["tap_fold_id"] == fold_id]
        horizons = fold_df["horizon_day"].unique()
        assert set(horizons) == {1, 2, 3}, \
            f"horizon_day mismatch for fold {fold_id}: {horizons}"

    print("PASS: test_split_month_predictions_to_learner_folds")


# ══════════════════════════════════════════════════════════════════════
# Test 3: Long-table Column Consistency
# ══════════════════════════════════════════════════════════════════════

def test_normalize_block_predictions_columns():
    """Verify normalize_block_predictions_to_tap produces required columns."""
    from pipelines.validation_tap_light_sgdf_timesfm import (
        normalize_block_predictions_to_tap,
    )

    predict_date = "2026-02-01"
    dummy = _make_dummy_predictions(predict_date, "test_model", n_days=30)

    # Normalize
    result = normalize_block_predictions_to_tap(
        dummy, predict_date=predict_date, task="dayahead", model_name="test_model",
    )

    # Required columns for learner
    required_cols = [
        "task", "model_name", "tap_fold_id", "learner_tap_fold_id",
        "age_block", "horizon_day", "age_days",
        "y_pred", "y_true",
        "tap_source", "source_confidence",
        "ds", "hour_business", "period",
        "target_day", "business_day",
    ]
    for col in required_cols:
        assert col in result.columns, f"Missing required column: {col}"

    # Verify tap_fold_id 0..9
    fold_ids = sorted(result["tap_fold_id"].unique().tolist())
    assert fold_ids == list(range(10)), f"Expected 0-9 folds, got {fold_ids}"

    # Verify source_confidence
    assert all(result["source_confidence"] == 1.0), "source_confidence should be 1.0 for rolling models"

    # Verify 30 days × 24 hours = 720 rows
    assert len(result) == 720, f"Expected 720 rows, got {len(result)}"

    print("PASS: test_normalize_block_predictions_columns")


# ══════════════════════════════════════════════════════════════════════
# Test 4: Runtime Report Format
# ══════════════════════════════════════════════════════════════════════

def test_runtime_report_format(tmp_path: Path):
    """Verify save_runtime_report produces correct CSV format."""
    from pipelines.validation_tap_light_sgdf_timesfm import save_runtime_report

    # Create a temp output dir
    output_dir = tmp_path / "test_runtime"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create sample runtime rows
    rows = [
        {
            "model_name": "lightgbm",
            "target": "dayahead",
            "resource": "cpu",
            "tap_strategy": "rolling_cutoff_3x10",
            "model_update_block_id": 0,
            "learner_tap_fold_id": None,
            "runtime_seconds": 2.5,
            "cache_hit": False,
            "status": "complete",
            "error_message": "",
        },
        {
            "model_name": "timesfm",
            "target": "dayahead",
            "resource": "timesfm",
            "tap_strategy": "direct_inference_daily",
            "model_update_block_id": "daily_2026-01-30",
            "learner_tap_fold_id": None,
            "runtime_seconds": 15.3,
            "cache_hit": True,
            "status": "cached",
            "error_message": "",
        },
    ]

    save_runtime_report(rows, output_dir)

    # Verify report file exists
    report_path = output_dir / "runtime_report.csv"
    assert report_path.exists(), f"runtime_report.csv not found at {report_path}"

    # Read and verify
    df = pd.read_csv(report_path)
    assert len(df) == len(rows), f"Expected {len(rows)} rows, got {len(df)}"
    assert list(df.columns) == [
        "model_name", "target", "resource", "tap_strategy",
        "model_update_block_id", "learner_tap_fold_id",
        "runtime_seconds", "cache_hit", "status", "error_message",
    ], f"Column mismatch: {list(df.columns)}"

    print("PASS: test_runtime_report_format")


# ══════════════════════════════════════════════════════════════════════
# Test 5: build_date_to_fold_map
# ══════════════════════════════════════════════════════════════════════

def test_build_date_to_fold_map():
    """Verify the date-to-fold mapping for 30 validation days."""
    from pipelines.validation_tap_light_sgdf_timesfm import build_date_to_fold_map

    predict_date = "2026-02-01"
    D = pd.Timestamp(predict_date).date()
    date_map = build_date_to_fold_map(predict_date)

    # Check all 30 days mapped to correct folds
    expected = {}
    for fold_id in range(10):
        start = D - pd.Timedelta(days=30 - 3 * fold_id)
        for offset in range(3):
            d = start + pd.Timedelta(days=offset)
            expected[d] = fold_id

    assert date_map == expected, f"Date mapping mismatch"

    # Verify D-30 -> fold 0, D-1 -> fold 9
    assert date_map[D - pd.Timedelta(days=30)] == 0
    assert date_map[D - pd.Timedelta(days=1)] == 9

    print("PASS: test_build_date_to_fold_map")


# ══════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Running validation tap tests...")
    test_generate_3x10_block_specs()
    test_split_month_predictions_to_learner_folds()
    test_normalize_block_predictions_columns()
    test_runtime_report_format(Path("tests"))
    test_build_date_to_fold_map()
    print("\nAll tests PASSED.")
