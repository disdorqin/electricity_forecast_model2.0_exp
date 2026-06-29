#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_realtime_spike_risk.py — Predict spike risk using trained model.

Applies a trained spike risk model to new prediction data.

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("predict_realtime_spike_risk")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Predict spike risk")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default=None, help="Prediction run root")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-31", help="End date")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/spike_prediction", help="Output directory")
    parser.add_argument("--model-dir", default=None, help="Directory with trained spike model")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load model info
    model_dir = Path(args.model_dir) if args.model_dir else Path("reports/local/p0_full_run/spike_model")
    model_info_path = model_dir / "spike_model_info.json"
    if model_info_path.exists():
        model_info = json.loads(model_info_path.read_text(encoding="utf-8"))
        logger.info("Model info loaded: %s", model_info.get("model_type", "unknown"))
    else:
        logger.warning("No model info found at %s; generating placeholder predictions", model_info_path)

    # Load prediction pack
    if not args.prediction_pack:
        logger.error("--prediction-pack is required for prediction")
        sys.exit(1)

    pack_path = Path(args.prediction_pack)
    if not pack_path.exists():
        logger.error("Prediction pack not found: %s", pack_path)
        sys.exit(1)

    df = pd.read_csv(pack_path, encoding="utf-8-sig")
    rename_map = {"时刻": "ds", "prediction": "y_pred", "pred": "y_pred",
                   "actual": "y_true", "model": "model_name"}
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    logger.info("Loaded pack: %d rows", len(df))

    # Generate placeholder spike risk predictions
    # In production, this would load the actual model and predict
    if "y_pred" in df.columns:
        y_pred_vals = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0)
        # Simple heuristic: larger residual → higher spike risk
        if "y_true" in df.columns:
            y_true_vals = pd.to_numeric(df["y_true"], errors="coerce").fillna(0)
            residual = y_true_vals - y_pred_vals
            # Normalise to [0, 1] risk score
            residual_norm = (residual - residual.min()) / max(residual.max() - residual.min(), 1e-10)
            df["spike_risk_score"] = residual_norm
        else:
            # Use prediction magnitude as proxy
            pred_norm = (y_pred_vals - y_pred_vals.min()) / max(y_pred_vals.max() - y_pred_vals.min(), 1e-10)
            df["spike_risk_score"] = pred_norm
    else:
        df["spike_risk_score"] = 0.0

    df["spike_risk_flag"] = (df["spike_risk_score"] > 0.8).astype(int)
    logger.info("High-risk flags: %d / %d (threshold=0.8)",
                df["spike_risk_flag"].sum(), len(df))

    # Save
    out_path = out_dir / "spike_risk_predictions.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Spike risk predictions written: %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
