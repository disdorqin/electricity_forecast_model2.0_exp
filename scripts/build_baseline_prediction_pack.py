#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_baseline_prediction_pack.py — Level 0 Baseline Prediction Pack for P0 window.

Generates naive baselines so the downstream spike/risk/correction pipeline
can be validated end-to-end, even without real model predictions.

Baseline models:
  - naive_lag1:    y_pred = y_true from same hour 1 day ago (shift 24)
  - naive_lag7:    y_pred = y_true from same hour 7 days ago (shift 168)
  - dayahead_proxy: y_pred = dayahead price (col 1) as realtime estimate
  - baseline_fusion: simple average of lag1 + lag7 + dayahead_proxy

Output:
  prediction_pack_realtime_level0_{start}_{end}.csv
  baseline_pack_manifest.json
  baseline_coverage_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── Column indices in the raw xlsx ──
COL_TIMESTAMP = 0    # 时刻
COL_DAYAHEAD = 1     # 日前电价
COL_REALTIME = 2     # 实时电价 (y_true)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Level 0 baseline prediction pack")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx",
                        help="Raw data path (xlsx or csv)")
    parser.add_argument("--target", default="realtime", choices=["realtime"])
    parser.add_argument("--start-date", default="2025-11-01")
    parser.add_argument("--end-date", default="2026-02-28")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/prediction_pack_level0")
    return parser.parse_args(argv)


def load_raw(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".xlsx":
        df = pd.read_excel(str(path))
    else:
        for enc in ("utf-8", "utf-8-sig", "gbk"):
            try:
                df = pd.read_csv(str(path), encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        else:
            raise ValueError(f"Cannot read {path}")

    raw = pd.DataFrame()
    raw["ds"] = pd.to_datetime(df.iloc[:, COL_TIMESTAMP], errors="coerce")
    raw["y_true"] = pd.to_numeric(df.iloc[:, COL_REALTIME], errors="coerce")
    raw["dayahead_price"] = pd.to_numeric(df.iloc[:, COL_DAYAHEAD], errors="coerce")
    raw = raw.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    return raw


def build_business_time(raw: pd.DataFrame) -> pd.DataFrame:
    """Add business_day, hour_business, period columns per shared contract."""
    ts = raw["ds"]
    raw["hour_business"] = ts.dt.hour.replace({0: 24}).astype(int)
    raw["business_day"] = (
        ts - pd.to_timedelta((ts.dt.hour == 0).astype(int), unit="D")
    ).dt.normalize()
    # Period mapping
    def _period(h: int) -> str:
        if 9 <= h <= 16:
            return "9_16"
        elif 1 <= h <= 8:
            return "night"
        elif h in (24,):
            return "night"
        else:
            return "evening"
    raw["period"] = raw["hour_business"].apply(_period)
    return raw


def build_baseline_predictions(raw: pd.DataFrame) -> pd.DataFrame:
    """Generate naive baseline predictions for the P0 window."""
    df = raw.copy()

    # naıve lag1: same hour yesterday
    df["naive_lag1"] = df["y_true"].shift(24)

    # naıve lag7: same hour 7 days ago
    df["naive_lag7"] = df["y_true"].shift(24 * 7)

    # dayahead proxy
    df["dayahead_proxy"] = df["dayahead_price"]

    # baseline fusion = average of available
    pred_cols = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
    df["baseline_fusion"] = df[pred_cols].mean(axis=1)

    return df


def filter_p0_window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["ds"] >= start) & (df["ds"] <= end)
    return df[mask].copy()


def assemble_prediction_pack(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the prediction pack in long format (one row per model per hour)."""
    rows = []
    model_names = ["naive_lag1", "naive_lag7", "dayahead_proxy", "baseline_fusion"]

    for _, row in df.iterrows():
        for model in model_names:
            y_pred = row[model]
            if pd.isna(y_pred):
                continue
            base_fused = row["baseline_fusion"]
            rows.append({
                "business_day": row["business_day"],
                "hour_business": row["hour_business"],
                "timestamp": row["ds"],
                "period": row["period"],
                "target": "realtime",
                "model_name": model,
                "y_pred": round(float(y_pred), 4),
                "base_fused_pred": round(float(base_fused) if pd.notna(base_fused) else float(y_pred), 4),
                "final_pred": round(float(y_pred), 4),  # no correction yet
                "y_true": round(float(row["y_true"]), 4) if pd.notna(row["y_true"]) else None,
                "residual": None,  # computed downstream
                "abs_error": None,
                "smape_floor50": None,
                "high_spike_flag": 0,
                "severe_underestimate_flag": 0,
                "source_file": "baseline_level0",
                "coverage_status": "available",
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        # Compute residuals
        result["residual"] = result["y_true"] - result["y_pred"]
        result["abs_error"] = result["residual"].abs()
        # sMAPE_floor50 per row
        yt = np.maximum(result["y_true"].fillna(50).values, 50)
        yp = np.maximum(result["y_pred"].fillna(50).values, 50)
        denom = (np.abs(yt) + np.abs(yp)) / 2.0
        smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
        result["smape_floor50"] = np.minimum(smape, 50.0).round(4)
        # Flags
        result["high_spike_flag"] = (result["y_true"] > result["y_pred"] + 200).astype(int)
        result["severe_underestimate_flag"] = (result["y_true"] - result["y_pred"] > 200).astype(int)

    return result


def write_outputs(pack: pd.DataFrame, out_dir: Path, start: str, end: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV pack
    start_c = start.replace("-", "_")
    end_c = end.replace("-", "_")
    csv_path = out_dir / f"prediction_pack_realtime_level0_{start_c}_{end_c}.csv"
    pack.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Pack written: {csv_path} ({len(pack)} rows)")

    # Manifest
    coverage = {
        "date_range": {"start": start, "end": end},
        "total_rows": len(pack),
        "models": sorted(pack["model_name"].unique().tolist()) if not pack.empty else [],
        "date_coverage": int(pack["business_day"].nunique()) if not pack.empty and "business_day" in pack.columns else 0,
    }
    manifest = {
        "level": 0,
        "label": "baseline (no real model predictions)",
        "models": ["naive_lag1", "naive_lag7", "dayahead_proxy", "baseline_fusion"],
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": str(Path.cwd() / "data/shandong_pmos_hourly.xlsx"),
        "disclaimer": "Baseline only. Not for final evaluation.",
        "coverage": coverage,
    }
    manifest_path = out_dir / "baseline_pack_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    # Coverage report
    if not pack.empty:
        daily = pack.groupby("business_day").agg(
            rows=("y_pred", "count"),
            models=("model_name", lambda x: x.nunique()),
            avg_smape=("smape_floor50", "mean"),
        ).reset_index()
        lines = [
            "# Baseline Coverage Report",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Level**: 0 (naive baselines only)",
            f"**Date range**: {start} ~ {end}",
            f"**Total rows**: {len(pack)}",
            f"**Models**: {', '.join(sorted(pack['model_name'].unique()))}",
            "",
            "## Daily Coverage",
            "",
            "| Date | Rows | Models | Avg sMAPE |",
            "|------|------|--------|-----------|",
        ]
        for _, r in daily.iterrows():
            lines.append(f"| {r['business_day']} | {r['rows']} | {r['models']} | {r['avg_smape']:.2f} |")
        missing_mask = ~pd.to_datetime(pd.date_range(start, end).strftime("%Y-%m-%d")).isin(daily["business_day"])
        n_missing = missing_mask.sum()
        if n_missing:
            lines.append("")
            lines.append(f"**Missing dates**: {n_missing}")

        report_path = out_dir / "baseline_coverage_report.md"
        report_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Coverage report: {report_path}")


def main():
    args = parse_args()
    print(f"Loading raw data from {args.data_path} ...")
    raw = load_raw(args.data_path)
    print(f"  Raw rows: {len(raw)}, range: {raw['ds'].min()} ~ {raw['ds'].max()}")

    raw = build_business_time(raw)
    raw = build_baseline_predictions(raw)
    p0 = filter_p0_window(raw, args.start_date, args.end_date)
    print(f"  P0 window rows: {len(p0)}")

    pack = assemble_prediction_pack(p0)
    print(f"  Pack rows: {len(pack)} (long format, {pack['model_name'].nunique()} models)")

    out_dir = Path(args.out_dir)
    write_outputs(pack, out_dir, args.start_date, args.end_date)
    print("\nDone. Level 0 baseline pack ready for smoke test.")


if __name__ == "__main__":
    main()
