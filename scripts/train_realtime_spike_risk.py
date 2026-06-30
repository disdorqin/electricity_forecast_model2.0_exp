#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_realtime_spike_risk.py — Train spike risk model.

Reads the spike training dataset produced by build_realtime_spike_dataset.py
and trains a classifier to predict spike probability.

Unified CLI:
  --data-path, --runs-root, --prediction-pack, --target,
  --start-date, --end-date, --out-dir
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root in sys.path for schema import
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.schema import ALL_EXCLUDED_COLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_realtime_spike_risk")

# Optional sklearn import
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("scikit-learn not available; training will be a no-op")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train spike risk prediction model")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default=None, help="Prediction run root")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-31", help="End date")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/spike_model", help="Output directory")
    parser.add_argument("--dataset", default=None, help="Path to spike training dataset CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate dataset: --dataset, then default location, then --out-dir parent
    dataset_path = None
    if args.dataset:
        dataset_path = Path(args.dataset)
    if not dataset_path or not dataset_path.exists():
        default_candidates = [
            Path(args.out_dir) / ".." / "spike_dataset" / "spike_training_dataset.csv",
            Path("reports/local/p0_full_run/spike_dataset/spike_training_dataset.csv"),
        ]
        for cand in default_candidates:
            resolved = cand.resolve()
            if resolved.exists():
                dataset_path = resolved
                break

    if not dataset_path or not dataset_path.exists():
        logger.error("Spike training dataset not found. Use --dataset to specify path.")
        logger.info("Expected at: %s", out_dir / ".." / "spike_dataset" / "spike_training_dataset.csv")
        sys.exit(1)

    logger.info("Loading dataset from: %s", dataset_path)
    df = pd.read_csv(dataset_path, encoding="utf-8-sig")
    logger.info("Dataset: %d rows, %d cols", len(df), list(df.columns))

    if "spike_label" not in df.columns:
        logger.error("spike_label column not found in dataset")
        sys.exit(1)

    # Prepare features — exclude ALL actual-value and target-leakage columns
    # Using ALL_EXCLUDED_COLS from schema.py for leakage-safe feature selection.
    exclude_cols = {"ds", "spike_label", "model_name", "_source", "_source_file",
                    "realtime_price", "dayahead_price", "y_true", "hour", "hour_business",
                    "weekday", "y_pred", "final_pred", "base_fused_pred"}
    exclude_cols.update(ALL_EXCLUDED_COLS)
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]

    # Drop constant columns
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]

    X = df[feature_cols].fillna(0).values
    y = df["spike_label"].values

    logger.info("Features: %d, positive class: %d / %d (%.4f)",
                len(feature_cols), y.sum(), len(y), y.sum() / max(len(y), 1))

    # Train
    if HAS_SKLEARN and len(feature_cols) > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y)

        model = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=args.random_state,
            class_weight="balanced", n_jobs=-1)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        roc_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")

        logger.info("Test ROC-AUC: %.4f", roc_auc)
        logger.info("Classification report:\n%s", classification_report(y_test, y_pred))

        # Save model info
        feature_importance = sorted(zip(feature_cols, model.feature_importances_),
                                     key=lambda x: -x[1])

        model_info = {
            "model_type": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "n_features": len(feature_cols),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "roc_auc": float(roc_auc) if not np.isnan(roc_auc) else None,
            "classification_report": {k: v for k, v in report.items() if isinstance(v, dict)},
            "top_features": [(name, float(imp)) for name, imp in feature_importance[:20]],
        }
        info_path = out_dir / "spike_model_info.json"
        info_path.write_text(json.dumps(model_info, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Model info written: %s", info_path)
    else:
        logger.info("Skipping training (no sklearn or no features). Writing placeholder.")
        model_info = {"status": "skipped", "reason": "no sklearn or empty features"}
        info_path = out_dir / "spike_model_info.json"
        info_path.write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    logger.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
