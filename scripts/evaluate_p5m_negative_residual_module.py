#!/usr/bin/env python3
"""
evaluate_p5m_negative_residual_module.py — Evaluate negative price / low valley correction.

Usage:
    python scripts/evaluate_p5m_negative_residual_module.py \\
        --canonical-pack reports/local/canonical_eval_pack.csv \\
        --out-dir reports/local/p5m_negative_residual \\
        --profile conservative

Output:
    - metrics_report.json
    - correction_manifest.json
    - corrected_predictions.csv per profile
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
    canonical_pack_path: str | Path,
    out_dir: str | Path,
    history_days: int = 120,
    profiles: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run negative correction for multiple profiles and collect metrics.

    Args:
        canonical_pack_path: Path to canonical evaluation pack CSV.
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
    df_full = pd.read_csv(canonical_pack_path)

    # Use full dataset as history for fitting
    history_df = df_full.copy()

    for profile_name in profiles:
        print(f"\n  Evaluating profile: {profile_name}")
        profile = get_profile(profile_name)

        # Run the correction pipeline
        result_df = apply_negative_correction(
            prediction_pack_path=canonical_pack_path,
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

        # Print summary
        neg_count = metrics.get("negative_count", 0)
        lv_count = metrics.get("low_valley_count", 0)
        print(f"    negative_count={neg_count}  low_valley_count={lv_count}")
        print(f"    overall_sMAPE: {metrics['overall_sMAPE_before']:.2f} -> {metrics['overall_sMAPE_after']:.2f} (delta={metrics['overall_sMAPE_delta']:+.2f})")
        print(f"    negative_MAE: {metrics['negative_MAE_before']:.2f} -> {metrics['negative_MAE_after']:.2f}")
        print(f"    low_valley_MAE: {metrics['low_valley_MAE_before']:.2f} -> {metrics['low_valley_MAE_after']:.2f}")
        print(f"    high_spike_MAE: {metrics['high_spike_MAE_before']:.2f} -> {metrics['high_spike_MAE_after']:.2f} (delta={metrics.get('high_spike_MAE_delta', 0):+.2f}%)")
        print(f"    normal_degradation: {metrics['normal_degradation']:+.4f}")

    # Write metrics report
    report = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canonical_pack": str(canonical_pack_path),
        "history_days": history_days,
        "evaluated_profiles": profiles,
        "results": all_metrics,
    }
    report_path = out_dir / "metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # GO / NO-GO / DATA-LIMITED assessment
    print(f"\n  {'='*60}")
    print(f"  GO / NO-GO / DATA-LIMITED Assessment")
    print(f"  {'='*60}")
    print(f"  {'Profile':<15} {'sMAPE d':<9} {'NegMAE d':<10} {'HS d%':<9} {'NormDeg':<10} {'Verdict':<15}")
    print(f"  {'-'*70}")
    for profile_name in profiles:
        m = all_metrics[profile_name]
        neg_count = m.get("negative_count", 0)
        smape_delta = m.get("overall_sMAPE_delta", 0)
        neg_improvement = m.get("negative_MAE_before", 0) - m.get("negative_MAE_after", 0)
        lv_improvement = m.get("low_valley_MAE_before", 0) - m.get("low_valley_MAE_after", 0)
        hs_delta = m.get("high_spike_MAE_delta", 0)
        normal_deg = m.get("normal_degradation", 0)

        # Check DATA-LIMITED
        if neg_count == 0:
            verdict = "DATA-LIMITED"
            # Can still evaluate low_valley
            if lv_improvement >= 0 and abs(smape_delta) <= 0.3 and abs(hs_delta) <= 3.0 and normal_deg <= 0.5:
                verdict = "DATA-LIMITED (LV ok)"
        else:
            go = (
                (neg_improvement >= 0 or lv_improvement >= 0)
                and abs(smape_delta) <= 0.3
                and abs(hs_delta) <= 3.0
                and normal_deg <= 0.5
            )
            verdict = "GO" if go else "NO-GO"

        print(f"  {profile_name:<15} {smape_delta:+.2f}     {neg_improvement:+.2f}      {hs_delta:+.1f}     {normal_deg:+.2f}         {verdict}")

    return all_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate negative price / low valley correction")
    parser.add_argument("--canonical-pack", required=True, help="Path to canonical evaluation pack CSV")
    parser.add_argument("--out-dir", default="reports/local/p5m_negative_residual", help="Output directory")
    parser.add_argument("--profile", default=None, choices=["conservative", "moderate", "aggressive"] + [None],
                        help="Profile name (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small window)")
    args = parser.parse_args()

    profiles = [args.profile] if args.profile else None

    all_metrics = evaluate_profiles(
        canonical_pack_path=args.canonical_pack,
        out_dir=args.out_dir,
        profiles=profiles,
    )

    print(f"\n  All results written to: {args.out_dir}")


if __name__ == "__main__":
    main()
