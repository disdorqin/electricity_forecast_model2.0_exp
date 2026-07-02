#!/usr/bin/env python3
"""
evaluate_p5_negative_residual_module.py — Evaluate negative price / low valley correction.

Usage:
    python scripts/evaluate_p5_negative_residual_module.py \\
        --prediction-pack reports/local/prediction_pack.csv \\
        --history-days 120 \\
        --profile conservative \\
        --out-dir reports/local/p5_negative_residual

Output:
    - corrected_predictions.csv
    - metrics_report.json
    - correction_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme.negative_price.apply_negative_correction import (
    apply_negative_correction,
    compute_metrics,
    get_profile,
    PROFILES,
)
from extreme.negative_price.labels import add_all_labels
from extreme.negative_price.risk_model import NegativeRiskModel, NegativeRiskConfig


def evaluate_profiles(
    prediction_pack_path: str | Path,
    out_dir: str | Path,
    history_days: int = 120,
    profiles: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run negative correction for multiple profiles and collect metrics.

    Args:
        prediction_pack_path: Path to prediction pack CSV.
        out_dir: Output directory.
        history_days: Days of history for fitting.
        profiles: List of profile names (default: all).

    Returns:
        Dict of profile_name -> metrics dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if profiles is None:
        profiles = list(PROFILES.keys())

    all_metrics: dict[str, Any] = {}
    df_full = pd.read_csv(prediction_pack_path)

    # Use full dataset as history for fitting
    history_df = df_full.copy()

    for profile_name in profiles:
        print(f"\n  Evaluating profile: {profile_name}")
        profile = get_profile(profile_name)

        result_df = apply_negative_correction(
            prediction_pack_path=prediction_pack_path,
            history_df=history_df,
            profile=profile,
            pred_col="base_fused_pred",
        )

        # Compute metrics
        metrics = compute_metrics(result_df)
        metrics["profile"] = profile_name

        # Write corrected predictions
        csv_path = out_dir / f"corrected_{profile_name}.csv"
        result_df.to_csv(csv_path, index=False)
        metrics["predictions_csv"] = str(csv_path)

        all_metrics[profile_name] = metrics
        print(f"    overall_sMAPE: {metrics['overall_sMAPE_before']:.2f} → {metrics['overall_sMAPE_after']:.2f} (Δ={metrics['overall_sMAPE_delta']:+.2f})")
        print(f"    negative_MAE: {metrics['negative_MAE_before']:.2f} → {metrics['negative_MAE_after']:.2f}")
        print(f"    low_valley_MAE: {metrics['low_valley_MAE_before']:.2f} → {metrics['low_valley_MAE_after']:.2f}")
        print(f"    high_spike_MAE: {metrics['high_spike_MAE_before']:.2f} → {metrics['high_spike_MAE_after']:.2f}")
        print(f"    normal_degradation: {metrics['normal_degradation']:+.4f}")

    # Write metrics report
    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_pack": str(prediction_pack_path),
        "history_days": history_days,
        "results": all_metrics,
    }
    report_path = out_dir / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Determine GO/NO-GO for each profile
    print(f"\n  GO/NO-GO Assessment:")
    print(f"  {'Profile':<15} {'sMAPE Δ':<10} {'Neg MAE Δ':<12} {'HS Deg%':<10} {'Normal Deg':<12} {'GO?':<6}")
    print(f"  {'-'*65}")
    for profile_name in profiles:
        m = all_metrics[profile_name]
        smape_delta = m.get("overall_sMAPE_delta", 0)
        neg_improvement = m.get("negative_MAE_before", 0) - m.get("negative_MAE_after", 0)
        hs_degradation = m.get("high_spike_degradation", 0)
        normal_degradation = m.get("normal_degradation", 0)

        go = (
            neg_improvement >= 0  # negative MAE improves
            and abs(smape_delta) <= 0.3  # sMAPE not worsen > 0.3
            and abs(hs_degradation) <= 3.0  # high_spike not worsen > 3%
            and normal_degradation <= 0.5  # normal degradation <= 0.5
        )
        verdict = "GO ✅" if go else "NO-GO ❌"
        print(f"  {profile_name:<15} {smape_delta:+.2f}      {neg_improvement:+.2f}        {hs_degradation:+.1f}       {normal_degradation:+.2f}          {verdict}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate negative price / low valley correction")
    parser.add_argument("--prediction-pack", required=True, help="Path to prediction pack CSV")
    parser.add_argument("--history-days", type=int, default=120, help="History days for fitting")
    parser.add_argument("--profile", default=None, help="Profile name (default: all)")
    parser.add_argument("--out-dir", default="reports/local/p5_negative_residual", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small window)")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else None

    all_metrics = evaluate_profiles(
        prediction_pack_path=args.prediction_pack,
        out_dir=args.out_dir,
        history_days=args.history_days,
        profiles=profiles,
    )

    print(f"\n  All results written to: {args.out_dir}")


if __name__ == "__main__":
    main()
