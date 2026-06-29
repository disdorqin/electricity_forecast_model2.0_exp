#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_backtest_prediction_pack.py — P0 Prediction Pack Builder

Scans daily_runs/ (or outputs/) for historical model predictions over a
date range and assembles a single consolidated prediction pack CSV.

Path resolution priority (per date):
  1. {runs_root}/{date}/realtime/model_outputs/{model}/*.csv
  2. {runs_root}/{date}/realtime/real/all_model_forecasts_long.csv
  3. {runs_root}/{date}/realtime/fused/fused_predictions.csv
  4. {runs_root}/{date}/realtime/final/realtime_final_predictions.csv
  5. {runs_root}/{date}/final/realtime_final_predictions.csv
  6. {runs_root}/{date}/final/realtime_final_predictions_corrected.csv
  7. {runs_root}/{date}/compat_fusion/realtime/fused_predictions_corrected.csv
  8. outputs/{date}/... (legacy fallback)

Outputs:
  - prediction_pack_realtime_{start_YYYY}_{MM}_{end_YYYY}_{MM}.csv
  - coverage_report.csv
  - gap_report.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("build_backtest_prediction_pack")

# Standardised column names expected in prediction files
DS_COLS = ["ds", "时刻", "time", "datetime"]
PRED_COLS = ["y_pred", "pred", "prediction", "forecast", "预测值"]
TRUE_COLS = ["y_true", "true", "actual", "real", "真实值", "实际值"]
MODEL_COLS = ["model", "model_name", "model_id"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a consolidated prediction pack from daily_runs/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data (used for validation only)")
    parser.add_argument("--runs-root", default="daily_runs", help="Prediction run root directory")
    parser.add_argument("--prediction-pack", default=None, help="Path to pre-built prediction pack CSV (skip scan if given)")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/prediction_pack", help="Output directory")
    parser.add_argument("--models", default="all", help="Comma-separated model list or 'all'")
    parser.add_argument("--runs-root-fallback", default="outputs", help="Legacy fallback runs root")
    return parser.parse_args(argv)


def _detect_ds_col(df: pd.DataFrame) -> str | None:
    for col in DS_COLS:
        if col in df.columns:
            return col
    return None


def _detect_pred_col(df: pd.DataFrame) -> str | None:
    for col in PRED_COLS:
        if col in df.columns:
            return col
    return None


def _detect_true_col(df: pd.DataFrame) -> str | None:
    for col in TRUE_COLS:
        if col in df.columns:
            return col
    return None


def _detect_model_col(df: pd.DataFrame) -> str | None:
    for col in MODEL_COLS:
        if col in df.columns:
            return col
    return None


def _standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names."""
    rename = {}
    ds_col = _detect_ds_col(df)
    if ds_col:
        rename[ds_col] = "ds"
    pred_col = _detect_pred_col(df)
    if pred_col:
        rename[pred_col] = "y_pred"
    true_col = _detect_true_col(df)
    if true_col:
        rename[true_col] = "y_true"
    model_col = _detect_model_col(df)
    if model_col:
        rename[model_col] = "model_name"
    if rename:
        df = df.rename(columns=rename)
    return df


def _parse_ds(df: pd.DataFrame) -> pd.DataFrame:
    if "ds" not in df.columns:
        return df
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    return df.dropna(subset=["ds"]).copy()


def scan_daily_runs(
    runs_root: Path,
    start_date: str,
    end_date: str,
    target: str,
    models: list[str] | None,
    fallback_root: Path | None = None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Scan runs_root directories and collect predictions.

    Returns (consolidated DataFrame, gap_report entries).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frames: list[pd.DataFrame] = []
    gaps: list[dict] = []

    # Generate list of dates in range
    current = start
    dates = []
    while current <= end:
        dates.append(current)
        current += timedelta(days=1)

    for d in dates:
        date_str = d.strftime("%Y-%m-%d")
        date_compact = d.strftime("%Y%m%d")

        # Try primary runs_root
        base = runs_root / date_str
        found_any = _scan_single_date(base, target, models, date_str, frames, gaps, "daily_runs")

        # Try fallback runs_root if primary found nothing
        if not found_any and fallback_root is not None:
            fallback_base = fallback_root / date_str
            found_any = _scan_single_date(fallback_base, target, models, date_str, frames, gaps, "outputs") or found_any

        if not found_any:
            # Try compact date format (YYYYMMDD)
            base_compact = runs_root / date_compact
            found_any = _scan_single_date(base_compact, target, models, date_str, frames, gaps, "daily_runs") or found_any
            if fallback_root is not None:
                fallback_compact = fallback_root / date_compact
                found_any = _scan_single_date(fallback_compact, target, models, date_str, frames, gaps, "outputs") or found_any

        # If still not found, log it as a gap
        if not found_any:
            gaps.append({
                "date": date_str,
                "reason": f"No prediction directories found in {runs_root}/{date_str} or {runs_root}/{date_compact}",
            })

    if not frames:
        return pd.DataFrame(), gaps

    combined = pd.concat(frames, ignore_index=True)
    combined = _parse_ds(combined)

    # Fill missing model info
    if "model_name" not in combined.columns:
        combined["model_name"] = "unknown"
    else:
        combined["model_name"] = combined["model_name"].fillna("unknown")

    # Ensure y_pred exists
    if "y_pred" not in combined.columns:
        combined["y_pred"] = float("nan")
    else:
        combined["y_pred"] = pd.to_numeric(combined["y_pred"], errors="coerce")

    logger.info(
        "Consolidated pack: %d rows, %d models, %s ~ %s",
        len(combined),
        combined["model_name"].nunique(),
        combined["ds"].min() if not combined.empty else "N/A",
        combined["ds"].max() if not combined.empty else "N/A",
    )
    return combined, gaps


def _scan_single_date(
    base: Path,
    target: str,
    models: list[str] | None,
    date_str: str,
    frames: list[pd.DataFrame],
    gaps: list[dict],
    source_label: str,
) -> bool:
    """Scan a single date directory for prediction files."""
    if not base.exists():
        return False

    found_any = False
    paths_to_try = []

    target_dir = base / target

    # 1. Model outputs
    model_outputs_dir = target_dir / "model_outputs"
    if model_outputs_dir.exists():
        if models:
            for model_name in models:
                model_dir = model_outputs_dir / model_name
                if model_dir.exists():
                    paths_to_try.extend([
                        (p, f"{source_label}/model_outputs/{model_name}")
                        for p in sorted(model_dir.glob("*.csv"))
                    ])
        else:
            for model_dir in sorted(model_outputs_dir.iterdir()):
                if model_dir.is_dir():
                    paths_to_try.extend([
                        (p, f"{source_label}/model_outputs/{model_dir.name}")
                        for p in sorted(model_dir.glob("*.csv"))
                    ])

    # 2. real/all_model_forecasts_long.csv
    real_dir = target_dir / "real"
    if real_dir.exists():
        fcast_path = real_dir / "all_model_forecasts_long.csv"
        if fcast_path.exists():
            paths_to_try.append((fcast_path, f"{source_label}/real"))

    # 3. fused/fused_predictions.csv
    fused_path = target_dir / "fused" / "fused_predictions.csv"
    if fused_path.exists():
        paths_to_try.append((fused_path, f"{source_label}/fused"))

    # 4. realtime/final/realtime_final_predictions.csv
    final_path_1 = target_dir / "final" / f"{target}_final_predictions.csv"
    if final_path_1.exists():
        paths_to_try.append((final_path_1, f"{source_label}/final"))

    # 5. final/realtime_final_predictions.csv
    final_path_2 = base / "final" / f"{target}_final_predictions.csv"
    if final_path_2.exists():
        paths_to_try.append((final_path_2, f"{source_label}/final_root"))

    # 6. final/realtime_final_predictions_corrected.csv
    final_corrected_path = base / "final" / f"{target}_final_predictions_corrected.csv"
    if final_corrected_path.exists():
        paths_to_try.append((final_corrected_path, f"{source_label}/final_corrected"))

    # 7. compat_fusion/realtime/fused_predictions_corrected.csv
    compat_path = base / "compat_fusion" / target / "fused_predictions_corrected.csv"
    if compat_path.exists():
        paths_to_try.append((compat_path, f"{source_label}/compat_fusion"))

    for csv_path, source in paths_to_try:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(csv_path, encoding="gbk")
            except Exception:
                logger.warning("Skipping unreadable: %s", csv_path)
                continue
        except Exception:
            logger.warning("Skipping unreadable: %s", csv_path)
            continue

        df = _standardise(df)
        df["_source"] = source
        df["_source_file"] = csv_path.name
        frames.append(df)
        found_any = True
        logger.info("  [%s] Found %d rows from %s", date_str, len(df), csv_path)

    return found_any


def build_prediction_pack(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Filter, deduplicate, and sort the consolidated pack."""
    if df.empty:
        return df

    # Filter date range
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    df = df[(df["ds"] >= start) & (df["ds"] <= end)].copy()

    # Sort
    sort_cols = ["ds"]
    if "model_name" in df.columns:
        sort_cols.append("model_name")
    df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def write_prediction_pack(df: pd.DataFrame, out_dir: Path, start_date: str, end_date: str):
    """Write consolidated prediction pack CSV."""
    start_compact = start_date.replace("-", "_")
    end_compact = end_date.replace("-", "_")
    filename = f"prediction_pack_realtime_{start_compact}_{end_compact}.csv"
    out_path = out_dir / filename
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        out_path.write_text("ds,model_name,y_pred,y_true\n", encoding="utf-8-sig")
        logger.info("Empty prediction pack written: %s", out_path)
        return

    # Drop internal columns before writing
    write_df = df.drop(columns=["_source", "_source_file"], errors="ignore")
    write_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Prediction pack written: %s (%d rows, %d cols)", out_path, len(write_df), len(write_df.columns))


def write_coverage_report(df: pd.DataFrame, out_dir: Path, gaps: list[dict]):
    """Write coverage and gap reports."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Coverage report
    if not df.empty and "ds" in df.columns:
        df["_date"] = df["ds"].dt.strftime("%Y-%m-%d")
        coverage = df.groupby("_date").agg(
            rows=("y_pred", "count"),
            models=("model_name", lambda x: x.nunique()),
            y_pred_available=("y_pred", lambda x: x.notna().sum()),
            start_ds=("ds", "min"),
            end_ds=("ds", "max"),
        ).reset_index().rename(columns={"_date": "date"})
        coverage.to_csv(out_dir / "coverage_report.csv", index=False, encoding="utf-8-sig")
        logger.info("Coverage report: %d dates", len(coverage))
    else:
        pd.DataFrame(columns=["date", "rows", "models", "y_pred_available", "start_ds", "end_ds"]).to_csv(
            out_dir / "coverage_report.csv", index=False, encoding="utf-8-sig")

    # Gap report
    gap_df = pd.DataFrame(gaps) if gaps else pd.DataFrame(columns=["date", "reason"])
    gap_df.to_csv(out_dir / "gap_report.csv", index=False, encoding="utf-8-sig")
    logger.info("Gap report: %d missing dates", len(gap_df))


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse model list
    models = None
    if args.models and args.models.lower() != "all":
        models = [m.strip() for m in args.models.split(",")]

    runs_root = Path(args.runs_root)
    fallback_root = Path(args.runs_root_fallback) if args.runs_root_fallback else None

    logger.info("Runs root: %s", runs_root)
    logger.info("Target: %s, Range: %s ~ %s", args.target, args.start_date, args.end_date)

    # Scan and build pack
    df, gaps = scan_daily_runs(
        runs_root, args.start_date, args.end_date,
        args.target, models, fallback_root,
    )
    pack_df = build_prediction_pack(df, args)

    # Write outputs
    write_prediction_pack(pack_df, out_dir, args.start_date, args.end_date)
    write_coverage_report(pack_df, out_dir, gaps)

    # Summary
    logger.info("=" * 60)
    logger.info("PREDICTION PACK BUILD SUMMARY")
    logger.info("=" * 60)
    logger.info("Total rows in pack: %d", len(pack_df))
    logger.info("Models found: %s", sorted(pack_df["model_name"].unique().tolist()) if not pack_df.empty and "model_name" in pack_df.columns else "N/A")
    logger.info("Date range: %s ~ %s",
                pack_df["ds"].min() if not pack_df.empty else "N/A",
                pack_df["ds"].max() if not pack_df.empty else "N/A")
    logger.info("Gaps (missing dates): %d", len(gaps))
    logger.info("Output directory: %s", out_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
