#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_p33_lgbm_weighting.py — P3.3 LightGBM Internal Sample Weighting experiments.

Runs a rolling daily walk-forward with different sample_weight profiles:
    - none
    - spike_weighted
    - severe_underestimate_weighted
    - period_spike_weighted

Usage:
    # Small window (2 weeks)
    python scripts/run_p33_lgbm_weighting.py \\
        --data-path data/shandong_pmos_hourly.csv \\
        --start-date 2025-11-01 --end-date 2025-11-15 \\
        --out-dir reports/local/p33_lgbm_internal_weighting

    # Full window (2 months)
    python scripts/run_p33_lgbm_weighting.py \\
        --data-path data/shandong_pmos_hourly.csv \\
        --start-date 2025-11-01 --end-date 2025-12-31 \\
        --out-dir reports/local/p33_lgbm_internal_weighting

Output:
    {profile}/predictions.csv    — daily predictions
    {profile}/metrics.json       — profile metrics
    comparison_summary.json      — all profiles compared
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lightGBM.main_fix import run_lgbm_pipeline

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Metrics ────────────────────────────────────────────────────────────

def compute_smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE with 50% floor, matching Phase2 evaluation."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_severe_underestimate_count(y_true: np.ndarray, y_pred: np.ndarray) -> int:
    """Count of hours where y_true - y_pred > 200."""
    return int((y_true - y_pred > 200).sum())


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all evaluation metrics from predictions DataFrame."""
    y_true = df["y"].values
    y_pred = df["pred_y"].values

    # Basic metrics
    mae = float(np.mean(np.abs(y_true - y_pred)))
    smape = compute_smape_floor50(y_true, y_pred)
    severe = compute_severe_underestimate_count(y_true, y_pred)

    # Hour-specific metrics
    hour = df["hour"].values if "hour" in df.columns else None

    # 9_16 sMAPE
    if hour is not None:
        is_9_16 = (hour >= 9) & (hour <= 16)
        if is_9_16.sum() > 0:
            smape_9_16 = compute_smape_floor50(y_true[is_9_16], y_pred[is_9_16])
        else:
            smape_9_16 = None
    else:
        smape_9_16 = None

    # Severe on 9_16 only
    if hour is not None and is_9_16.sum() > 0:
        severe_9_16 = compute_severe_underestimate_count(y_true[is_9_16], y_pred[is_9_16])
    else:
        severe_9_16 = None

    return {
        "mae": round(mae, 4),
        "smape_floor50": round(smape, 4),
        "severe_underestimate_count": severe,
        "smape_9_16_floor50": round(smape_9_16, 4) if smape_9_16 is not None else None,
        "severe_9_16": severe_9_16,
        "n_timestamps": len(df),
    }


# ── Runner ─────────────────────────────────────────────────────────────

PROFILES = ["reference", "none", "spike_weighted", "severe_underestimate_weighted", "period_spike_weighted"]


def run_profile(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    profile: str,
    out_dir: Path,
    training_months: int = 12,
    val_ratio: float = 0.2,
    use_predicted_temp: bool = True,
) -> dict[str, Any]:
    """Run LightGBM pipeline for a single weight profile.

    Args:
        profile: "reference" = existing code (no param), or a profile name.
    """
    print(f"\n  {'=' * 50}")
    print(f"  Profile: {profile}")
    print(f"  Date: {forecast_start} ~ {forecast_end}")
    print(f"  Training months: {training_months}")
    print(f"  {'=' * 50}")

    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if profile == "reference":
        result = run_lgbm_pipeline(
            data_path=data_path,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            target="实时电价",
            use_predicted_temp=use_predicted_temp,
            training_months=training_months,
            val_ratio=val_ratio,
        )
    else:
        result = run_lgbm_pipeline(
            data_path=data_path,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            target="实时电价",
            use_predicted_temp=use_predicted_temp,
            training_months=training_months,
            val_ratio=val_ratio,
            sample_weight_profile=profile,
        )

    runtime = time.time() - t0

    if result is None or result.empty:
        print(f"  [WARN] No predictions for {profile}")
        return {"profile": profile, "error": "No predictions", "runtime": runtime}

    # Save predictions
    pred_path = out_dir / "predictions.csv"
    result.to_csv(pred_path, index=False)
    print(f"  Predictions saved: {pred_path} ({len(result)} rows)")

    # Compute metrics
    # Use display_date or ds for grouping
    if "display_date" in result.columns:
        result["business_day"] = result["display_date"]
    elif "ds" in result.columns:
        result["business_day"] = pd.to_datetime(result["ds"]).dt.date

    metrics = compute_metrics(result)

    # Attach metadata
    metrics["profile"] = profile
    metrics["runtime_seconds"] = round(runtime, 1)
    metrics["n_days"] = result["business_day"].nunique() if "business_day" in result.columns else None

    # Save per-profile metrics
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"  MAE:      {metrics['mae']}")
    print(f"  sMAPE:    {metrics['smape_floor50']}")
    print(f"  Severe:   {metrics['severe_underestimate_count']}")
    print(f"  9-16 sMAPE: {metrics.get('smape_9_16_floor50', 'N/A')}")
    print(f"  Runtime:  {runtime:.0f}s")
    print(f"  Days:     {metrics.get('n_days', 'N/A')}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="P3.3 LightGBM Internal Sample Weighting experiments.",
    )
    parser.add_argument("--data-path", required=True,
                        help="Path to historical data CSV/Excel")
    parser.add_argument("--start-date", required=True,
                        help="Forecast start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True,
                        help="Forecast end date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p33_lgbm_internal_weighting",
                        help="Output directory")
    parser.add_argument("--training-months", type=int, default=12,
                        help="Training window in months (default: 12)")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio (default: 0.2)")
    parser.add_argument("--no-predicted-temp", action="store_true",
                        help="Disable predicted temperature")
    parser.add_argument("--profiles", nargs="+",
                        default=PROFILES,
                        choices=PROFILES,
                        help=f"Profiles to run (default: all {PROFILES})")
    args = parser.parse_args()

    data_path = args.data_path
    start_date = args.start_date
    end_date = args.end_date
    out_dir = Path(args.out_dir)
    use_predicted_temp = not args.no_predicted_temp

    # ── Run all profiles sequentially ─────────────────────────────────
    all_metrics: dict[str, dict[str, Any]] = {}
    t_start = time.time()

    for profile in args.profiles:
        profile_out = out_dir / profile
        try:
            metrics = run_profile(
                data_path=data_path,
                forecast_start=start_date,
                forecast_end=end_date,
                profile=profile,
                out_dir=profile_out,
                training_months=args.training_months,
                val_ratio=args.val_ratio,
                use_predicted_temp=use_predicted_temp,
            )
            all_metrics[profile] = metrics
        except Exception as e:
            print(f"  [ERROR] {profile} failed: {e}")
            all_metrics[profile] = {"profile": profile, "error": str(e)}

    total_time = time.time() - t_start

    # ── Comparison table ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  P3.3 LightGBM Internal Sample Weighting — Comparison")
    print(f"{'=' * 60}")

    # Header
    print(f"\n  {'Profile':<30} {'sMAPE':<10} {'Severe':<8} {'MAE':<10} {'9-16 sMAPE':<12} {'Runtime':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*8} {'-'*10} {'-'*12} {'-'*10}")

    for profile in args.profiles:
        m = all_metrics.get(profile, {})
        if "error" in m:
            print(f"  {profile:<30} ERROR: {m['error']}")
            continue
        smape = m.get("smape_floor50", "—")
        severe = m.get("severe_underestimate_count", "—")
        mae = m.get("mae", "—")
        smape916 = m.get("smape_9_16_floor50", "—")
        runtime = m.get("runtime_seconds", "—")
        if isinstance(smape, float):
            print(f"  {profile:<30} {smape:<10.2f} {severe:<8} {mae:<10.2f} {str(smape916):<12} {runtime:<10}")
        else:
            print(f"  {profile:<30} {smape:<10} {severe:<8} {mae:<10} {str(smape916):<12} {runtime:<10}")

    # ── GO check ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  GO Assessment")
    print(f"{'=' * 60}")

    # LightGBM reference = the 'reference' profile
    ref = all_metrics.get("reference", {})
    ref_smape = ref.get("smape_floor50")
    ref_severe = ref.get("severe_underestimate_count")

    print(f"\n  LightGBM reference: sMAPE={ref_smape}, severe={ref_severe}")

    # Internal GO: sMAPE < 22.02, severe < 80
    print(f"\n  Internal GO (sMAPE < 22.02, severe < 80):")
    for profile in args.profiles:
        m = all_metrics.get(profile, {})
        s = m.get("smape_floor50")
        v = m.get("severe_underestimate_count")
        if s is not None and v is not None:
            s_ok = s < 22.02
            v_ok = v < 80
            print(f"    {profile:<30} sMAPE={s:<8.2f} {'✅' if s_ok else '❌'}  severe={v:<5} {'✅' if v_ok else '❌'}")

    # Strong GO: sMAPE <= 20.86, severe <= 63
    print(f"\n  Strong GO (sMAPE ≤ 20.86, severe ≤ 63):")
    for profile in args.profiles:
        m = all_metrics.get(profile, {})
        s = m.get("smape_floor50")
        v = m.get("severe_underestimate_count")
        if s is not None and v is not None:
            s_ok = s <= 20.86
            v_ok = v <= 63
            print(f"    {profile:<30} sMAPE={s:<8.2f} {'✅' if s_ok else '❌'}  severe={v:<5} {'✅' if v_ok else '❌'}")

    # ── Summary ───────────────────────────────────────────────────────
    summary = {
        "script": "scripts/run_p33_lgbm_weighting.py",
        "data_path": data_path,
        "date_range": {"start": start_date, "end": end_date},
        "training_months": args.training_months,
        "val_ratio": args.val_ratio,
        "use_predicted_temp": use_predicted_temp,
        "total_runtime_seconds": round(total_time, 1),
        "profiles": all_metrics,
        "go_thresholds": {
            "internal_go": {"smape": 22.02, "severe": 80},
            "strong_go": {"smape": 20.86, "severe": 63},
        },
    }

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary: {summary_path}")
    print(f"  Total runtime: {total_time:.0f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
