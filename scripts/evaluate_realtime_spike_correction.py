#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_realtime_spike_correction.py — Evaluate P0 spike correction.

Compares predictions with and without correction to measure improvement.

Unified CLI:
  --data-path, --runs-root, --prediction-pack, --target,
  --start-date, --end-date, --out-dir
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_realtime_spike_correction")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate spike correction effect")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default="daily_runs", help="Prediction run root")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-31", help="End date")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/correction_eval", help="Output directory")
    parser.add_argument("--baseline-key", default="y_pred", help="Column name for uncorrected prediction")
    parser.add_argument("--corrected-key", default="y_fused_corrected", help="Column name for corrected prediction")
    return parser.parse_args(argv)


def smape_floor50(y_true, y_pred):
    y_t = np.maximum(np.asarray(y_true, dtype=float), 50)
    y_p = np.maximum(np.asarray(y_pred, dtype=float), 50)
    denom = np.abs(y_t) + np.abs(y_p)
    mask = denom > 1e-10
    return float(np.mean(2 * np.abs(y_t[mask] - y_p[mask]) / denom[mask]) * 100)


def load_predictions(pack_path: str) -> pd.DataFrame:
    if not pack_path or not Path(pack_path).exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(pack_path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        return pd.DataFrame()
    rename = {"时刻": "ds", "prediction": "y_pred", "pred": "y_pred",
              "model": "model_name", "y_fused_corrected": "y_corrected",
              "actual": "y_true"}
    df = df.rename(columns={c: rename.get(c, c) for c in df.columns})
    return df


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_predictions(args.prediction_pack)
    if df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    logger.info("Predictions: %d rows", len(df))

    results = []
    for model in df.get("model_name", ["all"]).unique() if "model_name" in df.columns else ["all"]:
        subset = df[df["model_name"] == model] if "model_name" in df.columns and model != "all" else df

        y_t = pd.to_numeric(subset.get("y_true", subset.get("realtime_price", float("nan"))), errors="coerce")
        y_p = pd.to_numeric(subset.get(args.baseline_key, subset.get("y_pred", float("nan"))), errors="coerce")
        y_c = pd.to_numeric(subset.get(args.corrected_key, subset.get("y_corrected", float("nan"))), errors="coerce")

        valid = y_t.notna() & y_p.notna()
        if valid.sum() < 10:
            continue

        smape_b = smape_floor50(y_t[valid], y_p[valid])
        if y_c.notna().sum() > 10:
            valid_c = y_t.notna() & y_c.notna()
            smape_a = smape_floor50(y_t[valid_c], y_c[valid_c])
            improvement = smape_b - smape_a
        else:
            smape_a = float("nan")
            improvement = float("nan")

        results.append({
            "model_name": model,
            "n": int(valid.sum()),
            "smape_before": round(smape_b, 4),
            "smape_after": round(smape_a, 4) if not np.isnan(smape_a) else None,
            "improvement": round(improvement, 4) if not np.isnan(improvement) else None,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "correction_evaluation.csv", index=False, encoding="utf-8-sig")
    logger.info("Results:\n%s", result_df.to_string())

    summary_path = out_dir / "correction_evaluation_report.md"
    lines = [
        "# Spike Correction Evaluation Report",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Range**: {args.start_date} ~ {args.end_date}",
        "",
        "## Results",
        "",
    ]
    if not result_df.empty:
        lines.append(result_df.to_string())
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report: %s", summary_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
