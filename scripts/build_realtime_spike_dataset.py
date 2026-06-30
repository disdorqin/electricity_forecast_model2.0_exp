#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_realtime_spike_dataset.py — Build spike training dataset from prediction pack.

Extracts feature-target pairs for spike risk modeling.

Leakage-safe column handling:
  - After merging raw data with predictions, all ACTUAL_VALUE_EXCLUDE_COLS
    are dropped before feature construction.
  - Only prediction-derived + calendar features are kept for training.
  - y_true and spike_label retained only for evaluation/labeling.

Unified CLI:
  --data-path, --runs-root, --prediction-pack, --target,
  --start-date, --end-date, --out-dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root in sys.path for schema import
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_realtime_spike_dataset")

from extreme.realtime_high_spike.schema import ACTUAL_VALUE_EXCLUDE_COLS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build spike training dataset")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default=None, help="Prediction run root")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-31", help="End date")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/spike_dataset", help="Output directory")
    parser.add_argument("--spike-threshold", type=float, default=500.0, help="Price threshold for spike label")
    parser.add_argument("--forecast-horizon", type=int, default=24, help="Forecast horizon in hours")
    return parser.parse_args(argv)


def load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        logger.warning("Data file not found: %s", data_path)
        return pd.DataFrame()
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    for enc in ("gbk", "gb18030", "utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot read {path}")


def load_predictions(pack_path: str) -> pd.DataFrame:
    if not pack_path:
        return pd.DataFrame()
    path = Path(pack_path)
    if not path.exists():
        logger.warning("Prediction pack not found: %s", pack_path)
        return pd.DataFrame()
    for enc in ("utf-8", "gbk", "utf-8-sig"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return pd.DataFrame()
    rename_map = {"时刻": "ds", "prediction": "y_pred", "pred": "y_pred",
                   "actual": "y_true", "model": "model_name"}
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    return df


def build_features(df: pd.DataFrame, args) -> pd.DataFrame:
    """Build feature matrix for spike prediction."""
    if df.empty:
        return df

    # Standardise timestamp
    if "ds" not in df.columns:
        cn_map = {"时刻": "ds", "时间": "ds", "datetime": "ds"}
        for old_cn, new_cn in cn_map.items():
            if old_cn in df.columns:
                df = df.rename(columns={old_cn: new_cn})
                break
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"]).copy()
        df["hour"] = df["ds"].dt.hour
        df["hour_business"] = df["hour"].apply(lambda h: 24 if h == 0 else h)
        df["weekday"] = df["ds"].dt.weekday

    # Map price columns
    price_map = {"实时电价": "realtime_price", "日前电价": "dayahead_price",
                  "实时出清价": "realtime_price", "日前出清价": "dayahead_price"}
    df = df.rename(columns={c: price_map.get(c, c) for c in df.columns})

    # Create spike label
    target_price = "realtime_price" if args.target == "realtime" else "dayahead_price"
    if target_price in df.columns:
        df["spike_label"] = (df[target_price] > args.spike_threshold).astype(int)
        logger.info("Spike label: %d spikes out of %d rows (threshold=%.0f)",
                    df["spike_label"].sum(), len(df), args.spike_threshold)

    # Create lag features for prediction columns
    for col in ["y_pred"]:
        if col in df.columns:
            for lag in [1, 2, 3, 6, 12, 24]:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # Rolling statistics
    for col in ["y_pred"]:
        if col in df.columns:
            df[f"{col}_rolling_mean_6"] = df[col].rolling(6, min_periods=1).mean()
            df[f"{col}_rolling_std_6"] = df[col].rolling(6, min_periods=1).std()
            df[f"{col}_rolling_max_6"] = df[col].rolling(6, min_periods=1).max()

    return df


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_df = load_data(args.data_path)
    logger.info("Loaded data: %d rows", len(data_df))

    # Load predictions
    pred_df = load_predictions(args.prediction_pack) if args.prediction_pack else pd.DataFrame()
    logger.info("Loaded predictions: %d rows", len(pred_df))

    # Merge
    if not data_df.empty and not pred_df.empty and "ds" in data_df.columns and "ds" in pred_df.columns:
        merged = pd.merge(data_df, pred_df, on="ds", how="left", suffixes=("", "_pred"))
    else:
        merged = data_df.copy()

    # Build features
    feature_df = build_features(merged, args)

    # Leakage-safe column whitelist: drop raw ACTUAL_VALUE_EXCLUDE_COLS
    # that may have leaked through the raw-data merge.
    cols_before = set(feature_df.columns)
    keep_exceptions = {"realtime_price", "dayahead_price", "y_true", "spike_label", "ds"}
    safe = [c for c in feature_df.columns
            if c not in ACTUAL_VALUE_EXCLUDE_COLS or c in keep_exceptions]
    dropped = cols_before - set(safe)
    if dropped:
        logger.info(
            "Dropped %d actual-value columns from dataset: %s",
            len(dropped), sorted(dropped)[:10],
        )
    feature_df = feature_df[safe].copy()

    # Save
    spike_csv = out_dir / "spike_training_dataset.csv"
    feature_df.to_csv(spike_csv, index=False, encoding="utf-8-sig")
    logger.info("Spike dataset written: %s (%d rows)", spike_csv, len(feature_df))

    # Summary
    if "spike_label" in feature_df.columns:
        logger.info("Spike distribution:\n%s", feature_df["spike_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
