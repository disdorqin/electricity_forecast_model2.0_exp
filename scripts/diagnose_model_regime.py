#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_model_regime.py — Model Regime Diagnostic Tool (P0)

Analyses prediction patterns across different market regimes
(e.g., high-price spikes, low-price valleys, normal operation).

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diagnose_model_regime")

REGIME_LABELS = [
    "extreme_high",   # > 90th percentile
    "high",           # > 75th percentile
    "normal",         # 25th–75th
    "low",            # < 25th percentile
    "extreme_low",    # < 10th percentile
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Model regime diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default=None, help="Prediction run root directory")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/regime", help="Output directory")
    return parser.parse_args(argv)


def load_data(data_path: str) -> pd.DataFrame:
    path = Path(data_path)
    if not path.exists():
        logger.warning("Data file not found: %s", data_path)
        return pd.DataFrame()

    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        for enc in ("gbk", "gb18030", "utf-8", "utf-8-sig"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"Cannot read {path}")

    # Map Chinese column names
    cn_map = {
        "时刻": "ds", "日前电价": "dayahead_price", "实时电价": "realtime_price",
        "日前出清价": "dayahead_price", "实时出清价": "realtime_price",
    }
    df = df.rename(columns={c: cn_map.get(c, c) for c in df.columns})
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
        df = df.dropna(subset=["ds"]).copy()
    return df


def load_predictions(pack_path: str) -> pd.DataFrame:
    if not pack_path:
        return pd.DataFrame()
    path = Path(pack_path)
    if not path.exists():
        logger.warning("Prediction pack not found: %s", pack_path)
        return pd.DataFrame()
    for enc in ("utf-8", "gbk"):
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
    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if "y_pred" in df.columns:
        df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce")
    return df


def classify_regime(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Classify each row into a price regime."""
    percentiles = df[price_col].quantile([0.10, 0.25, 0.75, 0.90])
    p10, p25, p75, p90 = percentiles[0.10], percentiles[0.25], percentiles[0.75], percentiles[0.90]

    def _regime(val):
        if val > p90: return "extreme_high"
        if val > p75: return "high"
        if val >= p25: return "normal"
        if val >= p10: return "low"
        return "extreme_low"

    df["regime"] = df[price_col].apply(_regime)
    df["regime_p10"] = p10
    df["regime_p25"] = p25
    df["regime_p75"] = p75
    df["regime_p90"] = p90
    logger.info("Regime thresholds: p10=%.2f, p25=%.2f, p75=%.2f, p90=%.2f", p10, p25, p75, p90)
    return df


def compute_regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime statistics."""
    if df.empty or "regime" not in df.columns:
        return pd.DataFrame()
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    group_cols = [c for c in ["regime"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    stats = df.groupby(group_cols).agg(
        count=("ds", "count"),
        **{f"{c}_mean": (c, "mean") for c in numeric if c != "regime"},
        **{f"{c}_std": (c, "std") for c in numeric[:3] if c != "regime"},
    ).reset_index()
    return stats


def compute_model_regime_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-regime error metrics."""
    if df.empty or "regime" not in df.columns or "y_pred" not in df.columns:
        return pd.DataFrame()
    if "model_name" not in df.columns:
        df["model_name"] = "unknown"

    results = []
    for (model, regime), group in df.groupby(["model_name", "regime"]):
        y_true = group.get("y_true", group.get("realtime_price", None))
        y_pred = group["y_pred"]
        if y_true is None:
            continue
        valid = y_true.notna() & y_pred.notna()
        y_t = y_true[valid]
        y_p = y_pred[valid]
        if len(y_t) == 0:
            continue
        smape = np.mean(2 * np.abs(y_p - y_t) / (np.abs(y_p) + np.abs(y_t) + 1e-10)) * 100
        mae = np.mean(np.abs(y_p - y_t))
        results.append({
            "model_name": model,
            "regime": regime,
            "count": len(y_t),
            "smape": smape,
            "mae": mae,
        })
    return pd.DataFrame(results)


def write_regime_report(regime_df: pd.DataFrame, model_perf: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    if not regime_df.empty:
        regime_df.to_csv(out_dir / "regime_classification.csv", index=False, encoding="utf-8-sig")

    if not model_perf.empty:
        model_perf.to_csv(out_dir / "model_regime_performance.csv", index=False, encoding="utf-8-sig")

    # Write markdown summary
    lines = ["# Model Regime Diagnosis Report", ""]
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    if not model_perf.empty:
        lines.append("## Per-Model Regime Performance")
        lines.append("")
        lines.append("| model | regime | count | sMAPE | MAE |")
        lines.append("|-------|--------|-------|-------|-----|")
        for _, row in model_perf.iterrows():
            lines.append(f"| {row['model_name']} | {row['regime']} | {int(row['count'])} | {row['smape']:.2f} | {row['mae']:.2f} |")
        lines.append("")

    if not regime_df.empty:
        lines.append("## Regime Distribution")
        lines.append("")
        dist = regime_df["regime"].value_counts()
        for reg in REGIME_LABELS:
            cnt = dist.get(reg, 0)
            pct = cnt / len(regime_df) * 100
            lines.append(f"- **{reg}**: {cnt} ({pct:.1f}%)")
        lines.append("")

    report_path = out_dir / "regime_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Regime report written: %s", report_path)


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.data_path)
    if df.empty:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Filter date range
    if "ds" in df.columns:
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
        df = df[(df["ds"] >= start) & (df["ds"] < end)].copy()
        logger.info("Data rows in range: %d", len(df))
    else:
        logger.info("No ds column, using all rows")

    # Load predictions if available
    pred_df = load_predictions(args.prediction_pack) if args.prediction_pack else pd.DataFrame()

    # Classify regime
    price_col = "realtime_price" if args.target == "realtime" else "dayahead_price"
    if price_col in df.columns:
        df = classify_regime(df, price_col)
    else:
        logger.warning("Price column '%s' not found in data", price_col)
        df["regime"] = "unknown"

    # Compute stats
    regime_stats = compute_regime_stats(df)
    logger.info("Regime stats:\n%s", regime_stats)

    # Merge predictions for model performance
    if not pred_df.empty and "ds" in pred_df.columns and "ds" in df.columns:
        merged = df.merge(pred_df[["ds", "y_pred", "model_name"]].drop_duplicates(subset=["ds"]),
                          on="ds", how="inner", suffixes=("", "_pred"))
        if "y_pred" in merged.columns:
            merged["y_true"] = merged.get("realtime_price", merged.get("dayahead_price"))
            model_perf = compute_model_regime_performance(merged)
        else:
            model_perf = pd.DataFrame()
    else:
        model_perf = pd.DataFrame()

    write_regime_report(df, model_perf, out_dir)

    logger.info("Done. Output: %s", out_dir)


if __name__ == "__main__":
    main()
