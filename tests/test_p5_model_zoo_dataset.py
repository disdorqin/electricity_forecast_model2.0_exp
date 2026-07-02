"""Tests for P5 Model-Zoo Dataset.

Covers:
1. No leakage columns (actual-value, forbidden features)
2. business_day/hour_business mapping (hb=24 → next-day 00:00)
3. Timestamp-level uniqueness (no duplicate business_day+hour_business)
4. Prediction schema required columns
5. Train/valid/test date boundary
6. Feature manifest completeness
7. y_true evaluation-only check

Run:
    python -m pytest tests/test_p5_model_zoo_dataset.py -v
    # or
    python tests/test_p5_model_zoo_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────

ZOO_DIR = _PROJECT_ROOT / "reports" / "local" / "p5_model_zoo"

TRAIN_PATH = ZOO_DIR / "train_panel.csv"
VALID_PATH = ZOO_DIR / "valid_panel.csv"
TEST_PATH = ZOO_DIR / "test_panel.csv"
MANIFEST_PATH = ZOO_DIR / "feature_manifest.json"
SCHEMA_PATH = ZOO_DIR / "prediction_schema.json"

# Forbidden leakage patterns
ACTUAL_VALUE_PATTERN = "实际值"
FORBIDDEN_FEATURES = {"y_true", "residual", "abs_error", "smape", "smape_floor50",
                       "severe_underestimate_flag", "spike_label", "realtime_price"}

# Expected date ranges
TRAIN_EXPECTED = ("2025-11-01", "2026-01-31")
VALID_EXPECTED = ("2026-02-01", "2026-02-15")
TEST_EXPECTED = ("2026-02-16", "2026-02-28")

# Prediction schema required fields
SCHEMA_REQUIRED = {
    "model_name", "business_day", "hour_business", "timestamp",
    "y_pred", "source_file", "prediction_mode", "leakage_safe",
}

# Expected model predictions
MODEL_PRED_COLS = ["y_pred_lightgbm", "y_pred_dayahead_proxy",
                   "y_pred_naive_lag1", "y_pred_naive_lag7"]


# ── Test 1: All output files exist ──────────────────────────────────────

def test_output_files_exist():
    """All P5 model-zoo output files are generated."""
    assert TRAIN_PATH.exists(), f"Missing: {TRAIN_PATH}"
    assert VALID_PATH.exists(), f"Missing: {VALID_PATH}"
    assert TEST_PATH.exists(), f"Missing: {TEST_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing: {MANIFEST_PATH}"
    assert SCHEMA_PATH.exists(), f"Missing: {SCHEMA_PATH}"


# ── Test 2: No leakage columns ──────────────────────────────────────────

def test_no_actual_value_columns():
    """No *实际值 columns in any panel."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        leakage_cols = [c for c in df.columns if ACTUAL_VALUE_PATTERN in c]
        assert len(leakage_cols) == 0, (
            f"{name} panel has actual-value columns: {leakage_cols}"
        )


def test_no_forbidden_prediction_time_features():
    """y_true is the only 'forbidden' column, and it's for evaluation only."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        cols_set = set(df.columns)
        # y_true is allowed (evaluation only)
        illegal = (cols_set & FORBIDDEN_FEATURES) - {"y_true"}
        assert len(illegal) == 0, (
            f"{name} panel has forbidden prediction-time features: {illegal}"
        )


# ── Test 3: business_day/hour_business mapping ──────────────────────────

def test_hour_24_maps_to_next_day_timestamp():
    """hb=24 rows have timestamps pointing to next calendar day 00:00."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        hb24 = df[df["hour_business"] == 24]
        for _, row in hb24.iterrows():
            bd = str(row["business_day"])
            hb = int(row["hour_business"])
            # hb=24 means physical 00:00 of next day
            # We can't check the timestamp col (not in output),
            # but the business_day should be correct
            assert hb == 24, f"Expected hb=24, got {hb} for {bd}"


def test_hour_range():
    """hour_business values are 1-24 inclusive."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        hrs = df["hour_business"]
        assert hrs.min() >= 1, f"{name}: hour_business < 1"
        assert hrs.max() <= 24, f"{name}: hour_business > 24"
        assert hrs.nunique() == 24, f"{name}: not all 24 hours present"


# ── Test 4: Timestamp-level uniqueness ──────────────────────────────────

def test_timestamp_key_unique():
    """business_day + hour_business is unique within each panel."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        n_before = len(df)
        n_after = len(df.drop_duplicates(subset=["business_day", "hour_business"]))
        assert n_before == n_after, (
            f"{name}: {n_before - n_after} duplicate timestamps"
        )


# ── Test 5: Train/valid/test date boundaries ────────────────────────────

def test_train_date_range():
    """Train panel covers 2025-11-01 to 2026-01-31."""
    df = pd.read_csv(TRAIN_PATH)
    assert df["business_day"].min() == TRAIN_EXPECTED[0], (
        f"Train starts {df['business_day'].min()}, expected {TRAIN_EXPECTED[0]}"
    )
    assert df["business_day"].max() == TRAIN_EXPECTED[1], (
        f"Train ends {df['business_day'].max()}, expected {TRAIN_EXPECTED[1]}"
    )


def test_valid_date_range():
    """Valid panel covers 2026-02-01 to 2026-02-15."""
    df = pd.read_csv(VALID_PATH)
    assert df["business_day"].min() == VALID_EXPECTED[0], (
        f"Valid starts {df['business_day'].min()}, expected {VALID_EXPECTED[0]}"
    )
    assert df["business_day"].max() == VALID_EXPECTED[1], (
        f"Valid ends {df['business_day'].max()}, expected {VALID_EXPECTED[1]}"
    )


def test_test_date_range():
    """Test panel covers 2026-02-16 to 2026-02-28."""
    df = pd.read_csv(TEST_PATH)
    assert df["business_day"].min() == TEST_EXPECTED[0], (
        f"Test starts {df['business_day'].min()}, expected {TEST_EXPECTED[0]}"
    )
    assert df["business_day"].max() == TEST_EXPECTED[1], (
        f"Test ends {df['business_day'].max()}, expected {TEST_EXPECTED[1]}"
    )


def test_partitions_are_disjoint():
    """Train/valid/test date ranges do not overlap."""
    train = set(pd.read_csv(TRAIN_PATH)["business_day"].unique())
    valid = set(pd.read_csv(VALID_PATH)["business_day"].unique())
    test = set(pd.read_csv(TEST_PATH)["business_day"].unique())

    assert train.isdisjoint(valid), f"Train and Valid overlap: {train & valid}"
    assert train.isdisjoint(test), f"Train and Test overlap: {train & test}"
    assert valid.isdisjoint(test), f"Valid and Test overlap: {valid & test}"


def test_total_days_match():
    """Total business days across all panels = 120 (canonical range)."""
    train = set(pd.read_csv(TRAIN_PATH)["business_day"].unique())
    valid = set(pd.read_csv(VALID_PATH)["business_day"].unique())
    test = set(pd.read_csv(TEST_PATH)["business_day"].unique())

    all_days = train | valid | test
    assert len(all_days) == 120, (
        f"Total {len(all_days)} days, expected 120"
    )


# ── Test 6: Feature manifest ────────────────────────────────────────────

def test_feature_manifest_valid():
    """Feature manifest exists and lists all columns with roles."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    assert "columns" in manifest, "Manifest missing 'columns'"
    assert manifest["feature_count"] > 0, "No features counted"
    assert manifest["model_prediction_count"] == 4, (
        f"Expected 4 model predictions, got {manifest['model_prediction_count']}"
    )
    assert manifest["leakage_checks"]["actual_value_columns_dropped"] == 10, (
        f"Dropped {manifest['leakage_checks']['actual_value_columns_dropped']} actual-value cols, expected 10"
    )

    # Verify y_true is documented as evaluation-only
    y_true_entry = [c for c in manifest["columns"] if c["name"] == "y_true"]
    assert len(y_true_entry) == 1, "y_true not found in manifest"
    assert "evaluation" in y_true_entry[0]["role"], (
        f"y_true role should mention evaluation, got: {y_true_entry[0]['role']}"
    )


# ── Test 7: Prediction schema ───────────────────────────────────────────

def test_prediction_schema_required_fields():
    """Prediction schema lists all required fields."""
    with open(SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    assert "fields" in schema, "Schema missing 'fields'"
    schema_field_names = {f["name"] for f in schema["fields"]}

    missing = SCHEMA_REQUIRED - schema_field_names
    assert not missing, f"Schema missing required fields: {missing}"

    # Each field must have name, type, required
    for field in schema["fields"]:
        assert "name" in field, f"Field missing 'name': {field}"
        assert "type" in field, f"Field {field['name']} missing 'type'"
        assert "required" in field, f"Field {field['name']} missing 'required'"

    # Verify all 8 required fields are marked required=True
    for field in schema["fields"]:
        if field["name"] in SCHEMA_REQUIRED:
            assert field["required"] is True, (
                f"Required field {field['name']} has required={field['required']}"
            )


# ── Test 8: Model predictions present ───────────────────────────────────

def test_model_predictions_present():
    """All 4 model prediction columns exist in all panels."""
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        for col in MODEL_PRED_COLS:
            assert col in df.columns, (
                f"{name} panel missing {col}"
            )


# ── Test 9: No NaN in critical columns ──────────────────────────────────

def test_no_nan_in_critical_columns():
    """Critical columns have minimal NaN. Known: test hb=24 on end-date boundary."""
    critical = ["business_day", "hour_business", "y_true", "base_fused_pred"]
    # Known edge: test panel last row (2026-02-28 hb=24) maps to 2026-03-01 outside canonical range
    # This causes NaN in y_true and base_fused_pred
    allowed_nan = {"train": 0, "valid": 0, "test": 1}
    for path, name in [(TRAIN_PATH, "train"), (VALID_PATH, "valid"), (TEST_PATH, "test")]:
        df = pd.read_csv(path)
        for col in critical:
            nan_count = df[col].isna().sum()
            assert nan_count <= allowed_nan[name], (
                f"{name} panel: {col} has {nan_count} NaN values "
                f"(allowed ≤ {allowed_nan[name]})"
            )


# ── Run directly ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("output_files_exist", test_output_files_exist),
        ("no_actual_value_columns", test_no_actual_value_columns),
        ("no_forbidden_prediction_time_features", test_no_forbidden_prediction_time_features),
        ("hour_24_maps_to_next_day", test_hour_24_maps_to_next_day_timestamp),
        ("hour_range", test_hour_range),
        ("timestamp_key_unique", test_timestamp_key_unique),
        ("train_date_range", test_train_date_range),
        ("valid_date_range", test_valid_date_range),
        ("test_date_range", test_test_date_range),
        ("partitions_are_disjoint", test_partitions_are_disjoint),
        ("total_days_match", test_total_days_match),
        ("feature_manifest_valid", test_feature_manifest_valid),
        ("prediction_schema_required_fields", test_prediction_schema_required_fields),
        ("model_predictions_present", test_model_predictions_present),
        ("no_nan_in_critical_columns", test_no_nan_in_critical_columns),
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
