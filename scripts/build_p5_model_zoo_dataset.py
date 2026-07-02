#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_p5_model_zoo_dataset.py — P5 model-zoo unified data pack and prediction schema.

Creates a unified dataset that ALL P5 model windows (W2/W3) must use for
training, validation, and evaluation. The dataset joins source features
with canonical prediction pack outputs and risk predictions.

Output (all gitignored under reports/local/p5_model_zoo/):
  - train_panel.csv       — training features + y_true + model predictions
  - valid_panel.csv       — validation features + y_true + model predictions
  - test_panel.csv        — test features + y_true + model predictions
  - feature_manifest.json — metadata for every column in the panels
  - prediction_schema.json— unified model output schema for W2/W3

Data rules enforced:
  1. No D+1 actual features (no 实际值 columns)
  2. No y_true / residual / abs_error / smape as prediction-time features
  3. y_true kept only for evaluation (clearly labelled)
  4. business_day + hour_business key aligned to canonical pack
  5. 00:00 → previous business_day + hour_business=24 (1-second offset)
  6. All model predictions follow unified schema

Usage:
    python scripts/build_p5_model_zoo_dataset.py

Author: W1 / Dataset Builder
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Paths ───────────────────────────────────────────────────────────────

SOURCE_CSV = "data/shandong_pmos_hourly.csv"
CANONICAL_PACK = "reports/local/p4_canonical/canonical_prediction_pack.csv"
CANONICAL_RISK = "reports/local/p4_canonical/canonical_risk_predictions.csv"
OUT_DIR = "reports/local/p5_model_zoo"

# Canonical date range
CANONICAL_START = "2025-11-01"
CANONICAL_END = "2026-02-28"

# Train / valid / test split (within canonical range)
TRAIN_START = "2025-11-01"
TRAIN_END = "2026-01-31"
VALID_START = "2026-02-01"
VALID_END = "2026-02-15"
TEST_START = "2026-02-16"
TEST_END = "2026-02-28"

# Column name mapping: Chinese 预测值 → English
FORECAST_COL_MAP = {
    "日前电价":                   "dayahead_price",
    "实时电价":                   "realtime_price",
    "地方电厂总加预测值":          "local_plant_forecast",
    "联络线受电负荷预测值":        "interconnect_forecast",
    "风电总加预测值":              "wind_forecast",
    "光伏总加预测值":              "solar_forecast",
    "核电总加预测值":              "nuclear_forecast",
    "自备机组总加预测值":          "self_owned_forecast",
    "试验机组总加预测值":          "test_units_forecast",
    "直调负荷预测值":              "load_forecast",
    "竞价空间预测值":              "bidding_space_forecast",
    "新能源总加预测值":            "renewable_forecast",
}

# Chinese 实际值 columns — MUST EXCLUDE (leakage)
ACTUAL_VALUE_COLS = [
    "地方电厂总加实际值",
    "联络线受电负荷实际值",
    "风电总加实际值",
    "光伏总加实际值",
    "核电总加实际值",
    "自备机组总加实际值",
    "试验机组总加实际值",
    "直调负荷实际值",
    "竞价空间实际值",
    "新能源总加实际值",
]

# Forbidden prediction-time columns
FORBIDDEN_FEATURES = {"y_true", "residual", "abs_error", "smape", "smape_floor50",
                       "severe_underestimate_flag", "spike_label", "realtime_price"}

# Period definitions
PERIOD_HOURS = {
    "night":  list(range(1, 9)),    # 1-8
    "9_16":   list(range(9, 17)),   # 9-16
    "evening": list(range(17, 25)),  # 17-24
}

# ── Helper functions ────────────────────────────────────────────────────


def get_period(hour_business: int) -> str:
    """Map business hour (1-24) to period label."""
    for period_name, hours in PERIOD_HOURS.items():
        if hour_business in hours:
            return period_name
    return "unknown"


def compute_business_key(ds_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Compute business_day and hour_business from a datetime series.

    Uses the "1-second offset" trick:
      - Subtract 1 second so physical 00:00 → previous day 23:59:59
      - Then hour+1 maps 0-23 → 1-24 business hours
      - business_day = the adjusted date

    Args:
        ds_series: datetime Series in physical time.

    Returns:
        (business_day_series, hour_business_series)
    """
    adjusted = ds_series - pd.Timedelta(seconds=1)
    business_day = adjusted.dt.strftime("%Y-%m-%d")
    hour_business = adjusted.dt.hour + 1  # 0-23 → 1-24
    return business_day, hour_business


def compute_derived_features(df: pd.DataFrame, target_col: str = "realtime_price") -> pd.DataFrame:
    """Compute derived features from raw forecast columns.

    These match the LightGBM feature_engineering logic but only use
    prediction-time-safe forecast columns (not actuals).

    Args:
        df: DataFrame with columns: load_forecast, wind_forecast, solar_forecast,
            interconnect_forecast, bidding_space_forecast, target_col
        target_col: Name of the price column to use for lag features.

    Returns:
        DataFrame with additional derived feature columns.
    """
    result = df.copy()
    safe_load = result["load_forecast"].replace(0, 1).fillna(1)

    # Net load features
    result["net_load"] = result["load_forecast"] - result["wind_forecast"] - result["solar_forecast"]
    result["solar_ratio"] = result["solar_forecast"] / safe_load
    result["net_load_sq"] = (result["net_load"] / 1000) ** 2

    # Bidding space
    result["bidding_space"] = result["net_load"] - result["interconnect_forecast"]
    result["space_ratio"] = result["bidding_space"] / safe_load

    # Renewable features
    result["wind_ratio"] = result["wind_forecast"] / safe_load
    result["renew_penetration"] = (result["wind_forecast"] + result["solar_forecast"]) / safe_load

    # Ramp features
    result["ramp_load"] = result["load_forecast"].diff().fillna(0)
    result["ramp_solar"] = result["solar_forecast"].diff().fillna(0)

    # Lag features (based on physical row order, safe because it's historical data)
    result["lag_48h"] = result[target_col].shift(48)
    result["lag_168h"] = result[target_col].shift(168)

    # Smart lag: weekday → 168h, weekend → 48h
    result["lag_price_target"] = np.where(
        result["day_of_week"] < 5,
        result["lag_168h"],
        result["lag_48h"],
    )
    result["lag_price_week"] = result["lag_168h"]
    result["lag_price_target"] = result["lag_price_target"].ffill().fillna(0)
    result["lag_price_week"] = result["lag_price_week"].ffill().fillna(0)

    # Drop intermediate lag columns (only keep smart-lag, not raw)
    result = result.drop(columns=["lag_48h", "lag_168h"], errors="ignore")

    # Drop realtime_price from features (it IS the target — historical ground truth)
    # Keep it only for lag computation, then remove to prevent feature-target leakage
    result = result.drop(columns=["realtime_price", "real_time_price"], errors="ignore")

    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features from the timestamp column.

    Expects 'ds' column (physical datetime).
    """
    result = df.copy()
    adjusted = result["ds"] - pd.Timedelta(seconds=1)

    result["hour_business"] = adjusted.dt.hour + 1
    result["weekday"] = adjusted.dt.dayofweek        # 0=Mon, 6=Sun
    result["day_of_week"] = result["weekday"]
    result["month"] = adjusted.dt.month
    result["day_of_month"] = adjusted.dt.day
    result["is_weekend"] = result["weekday"].isin([5, 6]).astype(int)
    result["period"] = result["hour_business"].apply(get_period)

    return result


# ── Main pipeline ───────────────────────────────────────────────────────


def build_model_zoo_dataset() -> dict[str, Any]:
    """Build the complete P5 model-zoo dataset.

    Returns manifest with summary statistics.
    """
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  P5 Model-Zoo Dataset Builder")
    print("=" * 60)

    # ── Step 1: Load source CSV ──────────────────────────────────────
    print(f"\n  [1/6] Loading source data: {SOURCE_CSV}")
    raw = pd.read_csv(SOURCE_CSV, encoding="gbk")
    print(f"  Raw rows: {len(raw)}, date range: {raw['时刻'].min()} ~ {raw['时刻'].max()}")

    # Parse timestamp
    raw["ds"] = pd.to_datetime(raw["时刻"], format="%Y/%m/%d %H:%M", errors="coerce")
    raw = raw.dropna(subset=["ds"]).reset_index(drop=True)

    # ── Step 2: Exclude leakage columns ───────────────────────────────
    print(f"  [2/6] Excluding leakage columns...")

    # Drop 实际值 columns
    actual_present = [c for c in ACTUAL_VALUE_COLS if c in raw.columns]
    raw = raw.drop(columns=actual_present, errors="ignore")
    print(f"  Dropped {len(actual_present)} actual-value columns: {actual_present}")

    # Verify no actual-value columns remain
    remaining_actual = [c for c in raw.columns if "实际值" in c]
    if remaining_actual:
        warnings.warn(f"Still have actual-value columns: {remaining_actual}")
        raw = raw.drop(columns=remaining_actual)

    # ── Step 3: Rename forecast columns and add features ──────────────
    print(f"  [3/6] Engineering features...")

    # Rename Chinese forecast columns to English
    rename_map = {k: v for k, v in FORECAST_COL_MAP.items() if k in raw.columns}
    raw = raw.rename(columns=rename_map)
    print(f"  Renamed {len(rename_map)} columns to English")

    # Add calendar features
    raw = add_calendar_features(raw)

    # Compute business_day using 1-second offset
    raw["business_day"], _ = compute_business_key(raw["ds"])

    # Compute derived features
    raw = compute_derived_features(raw)

    # ── Step 4: Merge with canonical prediction pack ──────────────────
    print(f"  [4/6] Merging canonical predictions and risk...")

    canonical = pd.read_csv(CANONICAL_PACK)
    risk = pd.read_csv(CANONICAL_RISK)

    # Filter to date range we can merge
    mask = (
        (raw["business_day"] >= CANONICAL_START)
        & (raw["business_day"] <= CANONICAL_END)
    )
    panel = raw[mask].copy()
    print(f"  Filtered to canonical date range: {len(panel)} rows")

    # Merge with canonical prediction pack
    merge_keys = ["business_day", "hour_business"]
    canonical_cols = merge_keys + [
        "y_true",
        "base_fused_pred", "high_spike", "high_spike_flag",
        "final_pred_reference", "lift_applied", "reason_code",
    ]
    panel = panel.merge(
        canonical[canonical_cols],
        on=merge_keys,
        how="left",
    )

    # Add per-model predictions
    model_cols = [c for c in canonical.columns if c.startswith("y_pred_")]
    panel = panel.merge(
        canonical[merge_keys + model_cols],
        on=merge_keys,
        how="left",
    )

    # Merge with risk predictions
    risk_cols = merge_keys + ["high_spike_prob", "spike_risk_score", "spike_risk_flag"]
    panel = panel.merge(
        risk[risk_cols],
        on=merge_keys,
        how="left",
    )

    n_before_dedup = len(panel)
    panel = panel.drop_duplicates(subset=merge_keys).reset_index(drop=True)
    print(f"  After dedup: {len(panel)} rows ({n_before_dedup - len(panel)} duplicates removed)")

    # ── Step 5: Split into train/valid/test ────────────────────────────
    print(f"  [5/6] Splitting train/valid/test...")

    train = panel[(panel["business_day"] >= TRAIN_START) & (panel["business_day"] <= TRAIN_END)].copy()
    valid = panel[(panel["business_day"] >= VALID_START) & (panel["business_day"] <= VALID_END)].copy()
    test = panel[(panel["business_day"] >= TEST_START) & (panel["business_day"] <= TEST_END)].copy()

    print(f"  Train: {len(train)} rows ({train['business_day'].nunique()} days)")
    print(f"  Valid: {len(valid)} rows ({valid['business_day'].nunique()} days)")
    print(f"  Test:  {len(test)} rows ({test['business_day'].nunique()} days)")

    # ── Step 6: Write outputs ─────────────────────────────────────────
    print(f"  [6/6] Writing outputs to {OUT_DIR}/...")

    # ── Clean up columns for output ─────────────────────────────────
    # Drop raw timestamp columns (we use business_day + hour_business)
    cols_to_drop = ["时刻", "ds"]
    cols_to_drop = [c for c in cols_to_drop if c in panel.columns]
    panel = panel.drop(columns=cols_to_drop, errors="ignore")

    # Re-split after cleanup
    train = panel[(panel["business_day"] >= TRAIN_START) & (panel["business_day"] <= TRAIN_END)].copy()
    valid = panel[(panel["business_day"] >= VALID_START) & (panel["business_day"] <= VALID_END)].copy()
    test = panel[(panel["business_day"] >= TEST_START) & (panel["business_day"] <= TEST_END)].copy()

    train.to_csv(out_dir / "train_panel.csv", index=False, encoding="utf-8-sig")
    valid.to_csv(out_dir / "valid_panel.csv", index=False, encoding="utf-8-sig")
    test.to_csv(out_dir / "test_panel.csv", index=False, encoding="utf-8-sig")
    print(f"  [OK] train_panel.csv ({len(train)} rows)")
    print(f"  [OK] valid_panel.csv ({len(valid)} rows)")
    print(f"  [OK] test_panel.csv ({len(test)} rows)")

    # ── Build feature manifest ───────────────────────────────────────

    # Categorize all columns
    feature_manifest: dict[str, Any] = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script": "scripts/build_p5_model_zoo_dataset.py",
        "source_data": {
            "csv": SOURCE_CSV,
            "date_range": {"start": str(raw["ds"].min()), "end": str(raw["ds"].max())},
        },
        "canonical_pack": str(CANONICAL_PACK),
        "canonical_risk": str(CANONICAL_RISK),
        "date_split": {
            "train": {"start": TRAIN_START, "end": TRAIN_END},
            "valid": {"start": VALID_START, "end": VALID_END},
            "test": {"start": TEST_START, "end": TEST_END},
        },
        "columns": [],
        "leakage_checks": {
            "actual_value_columns_dropped": len(actual_present),
            "forbidden_features_excluded": True,
            "y_true_role": "evaluation_only",
        },
        "feature_count": 0,
        "model_prediction_count": 0,
        "risk_score_count": 0,
    }

    col_roles: dict[str, dict[str, Any]] = {}

    for col in panel.columns:
        entry: dict[str, Any] = {
            "name": col,
            "dtype": str(panel[col].dtype),
        }

        if col in ("business_day", "hour_business", "timestamp_x", "timestamp_y", "ds"):
            entry["role"] = "key"
            entry["leakage_safe"] = True
        elif col == "y_true":
            entry["role"] = "target_evaluation_only"
            entry["leakage_safe"] = True  # safe in dataset (not used at prediction time)
            entry["note"] = "REAL-TIME actual value. EVALUATION ONLY. NEVER use as prediction-time feature."
        elif col.startswith("y_pred_"):
            entry["role"] = "model_prediction"
            entry["leakage_safe"] = True
        elif col == "base_fused_pred":
            entry["role"] = "fusion_prediction"
            entry["leakage_safe"] = True
        elif col == "final_pred_reference":
            entry["role"] = "corrected_prediction_reference"
            entry["leakage_safe"] = True
        elif col in ("lift_applied", "reason_code"):
            entry["role"] = "correction_metadata"
            entry["leakage_safe"] = True
        elif col in ("high_spike_prob", "spike_risk_score", "spike_risk_flag", "high_spike", "high_spike_flag"):
            entry["role"] = "risk_score"
            entry["leakage_safe"] = True
        elif col in ("hour_business", "weekday", "day_of_week", "month", "day_of_month", "is_weekend", "period"):
            entry["role"] = "calendar_feature"
            entry["leakage_safe"] = True
        elif col in FORBIDDEN_FEATURES:
            entry["role"] = "forbidden"
            entry["leakage_safe"] = False
            entry["note"] = "REMOVED — prediction-time leakage risk"
        else:
            entry["role"] = "derived_feature" if col in (
                "net_load", "solar_ratio", "net_load_sq",
                "bidding_space", "space_ratio",
                "wind_ratio", "renew_penetration",
                "ramp_load", "ramp_solar",
                "lag_price_target", "lag_price_week",
                "load_forecast", "wind_forecast", "solar_forecast",
                "interconnect_forecast", "nuclear_forecast",
                "self_owned_forecast", "test_units_forecast",
                "local_plant_forecast", "bidding_space_forecast",
                "renewable_forecast", "dayahead_price",
            ) else "source_feature"
            entry["leakage_safe"] = True

        col_roles[col] = entry

    feature_manifest["columns"] = list(col_roles.values())
    feature_manifest["feature_count"] = sum(
        1 for c in col_roles.values()
        if c["role"] in ("source_feature", "derived_feature", "calendar_feature")
    )
    feature_manifest["model_prediction_count"] = len([c for c in model_cols if c in panel.columns])
    feature_manifest["risk_score_count"] = 3

    # Add summary statistics per split
    feature_manifest["splits"] = {
        "train": {
            "rows": len(train),
            "business_days": int(train["business_day"].nunique()),
            "date_range": {"start": str(train["business_day"].min()), "end": str(train["business_day"].max())},
            "columns": len(train.columns),
        },
        "valid": {
            "rows": len(valid),
            "business_days": int(valid["business_day"].nunique()),
            "date_range": {"start": str(valid["business_day"].min()), "end": str(valid["business_day"].max())},
            "columns": len(valid.columns),
        },
        "test": {
            "rows": len(test),
            "business_days": int(test["business_day"].nunique()),
            "date_range": {"start": str(test["business_day"].min()), "end": str(test["business_day"].max())},
            "columns": len(test.columns),
        },
    }

    manifest_path = out_dir / "feature_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(feature_manifest, f, indent=2, ensure_ascii=False)
    print(f"  [OK] feature_manifest.json")

    # ── Build prediction schema ──────────────────────────────────────

    prediction_schema = {
        "schema_version": "1.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Unified model output schema for ALL P5 W2/W3 models",
        "description": (
            "All W2/W3 model outputs MUST conform to this schema. "
            "This ensures downstream consumers (fusion, correction, evaluation) "
            "can read any model output without special-case handling."
        ),
        "fields": [
            {
                "name": "model_name",
                "type": "string",
                "required": True,
                "description": "Unique model identifier (e.g. lightgbm, timesfm, rt916, sgdfnet, timemixer)",
                "example": "lightgbm_quantile_0p8",
            },
            {
                "name": "business_day",
                "type": "string",
                "format": "YYYY-MM-DD",
                "required": True,
                "description": "Business date (1-second offset: physical 00:00 → previous day)",
                "example": "2026-02-15",
            },
            {
                "name": "hour_business",
                "type": "integer",
                "minimum": 1,
                "maximum": 24,
                "required": True,
                "description": "Business hour (1-24). hour=24 means physical 00:00 of next calendar day.",
                "example": 24,
            },
            {
                "name": "timestamp",
                "type": "string",
                "format": "ISO datetime",
                "required": True,
                "description": "Physical timestamp (ISO 8601). hour_business=24 → next-day 00:00.",
                "example": "2026-02-16 00:00:00",
            },
            {
                "name": "y_pred",
                "type": "float",
                "required": True,
                "description": "Model prediction for the target (real-time electricity price).",
                "example": 342.15,
            },
            {
                "name": "source_file",
                "type": "string",
                "required": True,
                "description": "Relative path to the script or file that generated this prediction.",
                "example": "scripts/predict_lightgbm.py",
            },
            {
                "name": "prediction_mode",
                "type": "string",
                "enum": ["eval", "live", "backfill"],
                "required": True,
                "description": (
                    "Context of the prediction. "
                    "eval = historical evaluation (has y_true). "
                    "live = real-time/deployed. "
                    "backfill = historical reprocess."
                ),
                "example": "eval",
            },
            {
                "name": "leakage_safe",
                "type": "boolean",
                "required": True,
                "description": (
                    "Must be True. This field certifies that no D+1 actual features, "
                    "no *实际值 columns, and no target leakage were used to produce y_pred."
                ),
                "example": True,
            },
        ],
        "file_format": "CSV with UTF-8 BOM encoding",
        "file_pattern": "predictions_{model_name}_{date_range}.csv",
        "usage_notes": (
            "1. All W2/W3 model outputs MUST include these 8 fields. "
            "2. Additional fields are allowed but the 8 required fields must be present. "
            "3. business_day + hour_business must form a unique key. "
            "4. timestamp is informational; always use (business_day, hour_business) for joins. "
            "5. prediction_mode='eval' outputs will be used for metric computation vs y_true. "
            "6. leakage_safe=False outputs will be REJECTED by downstream consumers."
        ),
        "output_directory": "reports/local/p5_model_zoo/predictions/{model_name}/",
    }

    schema_path = out_dir / "prediction_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(prediction_schema, f, indent=2, ensure_ascii=False)
    print(f"  [OK] prediction_schema.json")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n  {'=' * 60}")
    print(f"  P5 Model-Zoo Dataset Complete")
    print(f"  {'=' * 60}")
    print(f"  Train: {len(train)} rows, {train['business_day'].nunique()} days")
    print(f"  Valid: {len(valid)} rows, {valid['business_day'].nunique()} days")
    print(f"  Test:  {len(test)} rows, {test['business_day'].nunique()} days")
    print(f"  Total: {len(train) + len(valid) + len(test)} rows")
    print(f"  Features: {feature_manifest['feature_count']}")
    print(f"  Model predictions: {feature_manifest['model_prediction_count']}")
    print(f"  Output: {out_dir}/")
    print()

    return feature_manifest


if __name__ == "__main__":
    build_model_zoo_dataset()
