#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3 SOTA Lab — LightGBM Profile Evaluation & Comparison

Compares all trained profiles against baseline,
outputs a markdown comparison table.

Usage:
  python scripts/evaluate_lightgbm_p3_sota.py \
    --results-dir reports/local/p3_sota_lab \
    --out-dir reports/local/p3_sota_lab
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pandas as pd

BASELINE_TARGET = {"smape": 22.02, "mae": 62.0, "severe": 80}
FUSION_TARGET = {"smape": 20.86, "mae": 58.0, "severe": 63}
STRONG_TARGET = {"smape": 20.86, "severe": 63}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3 SOTA Lab — Profile Evaluation")
    parser.add_argument("--results-dir", default="reports/local/p3_sota_lab", help="Directory with result JSONs")
    parser.add_argument("--out-dir", default="reports/local/p3_sota_lab", help="Output directory")
    return parser.parse_args(argv)


def load_all_results(results_dir: Path):
    results = []
    for f in sorted(results_dir.glob("lightgbm_*_results.json")):
        with open(f) as fh:
            data = json.load(fh)
            data["_file"] = f.name
            results.append(data)
    return results


def metric_emoji(val, target, higher_better=False):
    if val is None or val != val:
        return "⚪"
    if higher_better:
        return "✅" if val >= target else "⚠️"
    return "✅" if val <= target else "⚠️"


def build_report(results, targets):
    rows = []
    for r in results:
        profile = r["profile"]
        nl = r.get("no_leakage", False)
        label = f"{profile}{' (no-leakage)' if nl else ''}"

        smape = r.get("smape_calib", r.get("smape_raw"))
        mae = r.get("mae_calib", r.get("mae_raw"))
        severe = r.get("severe_calib", r.get("severe_raw"))
        features = r.get("features", "?")
        train_s = r.get("train_seconds", 0)
        n_val = r.get("val_rows", 0)
        calib = r.get("calibration", {})

        rows.append({
            "Profile": label,
            "sMAPE": f"{smape:.2f}" if smape else "N/A",
            "↓BL": f"{'✅' if smape and smape <= targets['baseline']['smape'] else '❌'}",
            "↓Fusion": f"{'✅' if smape and smape <= targets['fusion']['smape'] else '❌'}",
            "MAE": f"{mae:.2f}" if mae else "N/A",
            "Severe": str(severe) if severe is not None else "N/A",
            "Feat": str(features),
            "Train(s)": f"{train_s:.0f}",
            "Calib": f"{'Y' if calib else 'N'}" if calib else "N",
            "Best": "⭐" if smape and smape <= targets["strong"]["smape"] and severe is not None and severe <= targets["strong"]["severe"] else
                    "👍" if smape and smape <= targets["baseline"]["smape"] else ""
        })
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    targets = {
        "baseline": {"smape": 22.02, "severe": 80},
        "fusion": {"smape": 20.86, "severe": 63},
        "strong": {"smape": 20.86, "severe": 63},
    }

    df = build_report(results, targets)

    print("\n" + "=" * 80)
    print("P3 SOTA Lab — LightGBM Profile Comparison")
    print("=" * 80)
    print(df.to_string(index=False))
    print()

    # Determine best
    best = None
    best_score = float("inf")
    for r in results:
        smape = r.get("smape_calib", r.get("smape_raw"))
        if smape and smape < best_score:
            best_score = smape
            best = r

    if best:
        print(f"Best profile: {best['profile']} (sMAPE={best_score:.2f})")
        print(f"  Beats LightGBM baseline (22.02)? {'YES' if best_score <= 22.02 else 'NO'}")
        print(f"  Beats Phase2 fusion (20.86)? {'YES' if best_score <= 20.86 else 'NO'}")
        severe = best.get("severe_calib", best.get("severe_raw"))
        if severe is not None:
            print(f"  Severe underestimate: {severe} (target ≤ 80)")

    # Save report
    report_path = out_dir / "lightgbm_profile_comparison.csv"
    df.to_csv(report_path, index=False, encoding="utf-8-sig")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    main()
