#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_realtime_spike_risk.py — Predict spike risk using trained model.

Applies a trained spike risk model to new prediction data.

Leakage safety:
  - No y_true, residual, abs_error, sMAPE, or actual-value columns may be
    used as prediction-time features.
  - When a trained model artifact exists, it is loaded via joblib and used
    for predict_proba.  The feature set is built from prediction-time-safe
    columns only (calendar, exogenous forecasts, model predictions).
  - When no model artifact is available, a forecast-error-free heuristic
    based on prediction magnitude is used as fallback.  This heuristic
    uses ONLY y_pred and calendar columns — never y_true.

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

# Optional sklearn import for trained model loading
try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


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

    # Locate model directory
    model_dir = Path(args.model_dir) if args.model_dir else Path("reports/local/p0_full_run/spike_model")
    model_info_path = model_dir / "spike_model_info.json"

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
    logger.info("Loaded pack: %d rows, %d cols", len(df), len(df.columns))

    # ── Attempt to load a trained model and predict ──────────────────
    # IMPORTANT: No y_true allowed in prediction-time inference.
    risk_source = "fallback_heuristic"
    model_artifact = model_dir / "spike_risk_model.joblib"

    if model_info_path.exists() and model_artifact.exists() and HAS_SKLEARN:
        try:
            import joblib
            model = joblib.load(model_artifact)
            feature_cols = _get_predict_feature_cols(df)
            if len(feature_cols) > 0:
                X_pred = df[feature_cols].fillna(0).values
                df["spike_risk_score"] = model.predict_proba(X_pred)[:, 1]
                risk_source = "model_inference"
                logger.info(
                    "Trained model loaded from %s (%d features, risk_source=%s)",
                    model_artifact, len(feature_cols), risk_source,
                )
            else:
                logger.warning("No overlapping feature columns for model prediction; fallback to heuristic")
        except Exception as e:
            logger.warning("Model load failed: %s; falling back to heuristic", e)

    # ── Leakage-safe fallback heuristic (NO y_true) ──────────────────
    # This heuristic uses ONLY prediction-time-available columns:
    #   - y_pred (prediction magnitude)
    #   - Calendar features are available but not used here
    #   - NEVER y_true, residual, abs_error, or smape
    if "spike_risk_score" not in df.columns:
        if "y_pred" in df.columns:
            y_pred_vals = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0)
            pred_min = y_pred_vals.min()
            pred_range = y_pred_vals.max() - pred_min
            if pred_range > 1e-10:
                df["spike_risk_score"] = (y_pred_vals - pred_min) / pred_range
            else:
                df["spike_risk_score"] = 0.0
        else:
            df["spike_risk_score"] = 0.0
        logger.info("Fallback heuristic applied (risk_source=%s)", risk_source)

    df["spike_risk_flag"] = (df["spike_risk_score"] > 0.8).astype(int)
    logger.info("High-risk flags: %d / %d (threshold=0.8, risk_source=%s)",
                df["spike_risk_flag"].sum(), len(df), risk_source)

    # Save
    out_path = out_dir / "spike_risk_predictions.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Spike risk predictions written: %s (%d rows, risk_source=%s)",
                out_path, len(df), risk_source)

    # Write risk-source manifest
    manifest = {
        "script": "predict_realtime_spike_risk.py",
        "risk_source": risk_source,
        "model_artifact_loaded": model_artifact.exists() if model_artifact else False,
        "n_rows": len(df),
        "n_high_risk": int(df["spike_risk_flag"].sum()),
        "leakage_safe": True,
        "note": (
            "No y_true, residual, abs_error, sMAPE, or actual-value columns "
            "used at prediction time."
        ),
    }
    manifest_path = out_dir / "risk_prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Risk manifest written: %s", manifest_path)


def _get_predict_feature_cols(df: pd.DataFrame) -> list[str]:
    """Get feature columns matching trained model, excluding leakage columns."""
    # ACTUAL_COLS — same-hour actuals are prediction-time unknown
    ACTUAL_COLS = [
        "地方电厂总加实际值", "联络线受电负荷实际值", "风电总加实际值",
        "光伏总加实际值", "核电总加实际值", "自备机组总加实际值",
        "试验机组总加实际值", "直调负荷实际值", "竞价空间实际值", "新能源总加实际值",
    ]
    exclude = {"ds", "spike_label", "model_name", "_source", "_source_file",
               "realtime_price", "dayahead_price", "y_true",
               "abs_error", "smape", "residual", "lift_applied", "reason_code",
               "high_spike", "high_spike_flag", "spike_risk_score", "spike_risk_flag"}
    exclude.update(ACTUAL_COLS)
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]


if __name__ == "__main__":
    main()
