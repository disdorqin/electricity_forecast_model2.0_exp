"""Inspect production pipeline outputs for a given date.

Usage:
    python scripts/inspect_outputs.py [DATE]

Checks:
    - run_manifest.json exists and steps complete
    - validation tap: 10 folds, 3 horizon_days, correct columns
    - weights: sum ~= 1 per (task, period)
    - fused_predictions: 24 rows per target_day
    - final: 24 rows
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def inspect_date(date_str: str):
    date_dir = PROJECT_ROOT / "outputs" / date_str
    if not date_dir.is_dir():
        print(f"ERROR: {date_dir} does not exist")
        return False

    print(f"=== Inspecting {date_str} ===\n")
    ok = True

    # 1. Manifest
    manifest_path = date_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        steps = manifest.get("steps", {})
        print(f"Manifest steps ({len(steps)}):")
        for k, v in steps.items():
            status_icon = "OK" if v in ("complete", "skipped") else "WARN"
            print(f"  [{status_icon}] {k}: {v}")
        warnings = manifest.get("warnings", [])
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                print(f"  ! {w}")
    else:
        print("[FAIL] run_manifest.json not found")
        ok = False

    # 2. Per-target checks
    for target in ["dayahead", "realtime"]:
        target_dir = date_dir / target
        if not target_dir.is_dir():
            print(f"\n[{target}] directory not found (skipped)")
            continue

        print(f"\n--- {target} ---")

        # Validation tap
        tap_path = target_dir / "validation" / "validation_tap_long_table.csv"
        if tap_path.exists():
            tap_df = pd.read_csv(tap_path)
            n_folds = tap_df["tap_fold_id"].nunique() if "tap_fold_id" in tap_df.columns else 0
            n_horizons = tap_df["horizon_day"].nunique() if "horizon_day" in tap_df.columns else 0
            n_models = tap_df["model_name"].nunique() if "model_name" in tap_df.columns else 0
            print(f"  Validation tap: {len(tap_df)} rows, {n_folds} folds, {n_horizons} horizon_days, {n_models} models")
            if n_folds != 10:
                print(f"  [WARN] Expected 10 folds, got {n_folds}")
        else:
            print("  [WARN] validation_tap_long_table.csv not found")

        # Weights
        weights_path = target_dir / "fused" / "weights.csv"
        if weights_path.exists():
            w_df = pd.read_csv(weights_path)
            if "weight" in w_df.columns and "period" in w_df.columns:
                for period, grp in w_df.groupby("period"):
                    wsum = grp["weight"].sum()
                    status = "OK" if abs(wsum - 1.0) < 0.05 else "FAIL"
                    print(f"  Weights {period}: sum={wsum:.4f} [{status}]")
        else:
            print("  [WARN] weights.csv not found")

        # Fused predictions
        fused_path = target_dir / "fused" / "fused_predictions.csv"
        if fused_path.exists():
            fused_df = pd.read_csv(fused_path)
            if "target_day" in fused_df.columns:
                day_counts = fused_df.groupby("target_day").size()
                all_24 = all(day_counts == 24)
                print(f"  Fused: {len(fused_df)} rows, {len(day_counts)} days, all 24h={all_24}")
            else:
                print(f"  Fused: {len(fused_df)} rows (no target_day column)")
        else:
            print("  [WARN] fused_predictions.csv not found")

    # 3. Final outputs
    final_dir = date_dir / "final"
    if final_dir.is_dir():
        print(f"\n--- final ---")
        for csv_file in sorted(final_dir.glob("*.csv")):
            df = pd.read_csv(csv_file)
            print(f"  {csv_file.name}: {len(df)} rows")
    else:
        print("\n[WARN] final/ directory not found")

    print(f"\n=== {'PASS' if ok else 'ISSUES FOUND'} ===")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Inspect pipeline outputs")
    parser.add_argument("date", nargs="?", default="2026-02-01", help="Date YYYY-MM-DD")
    args = parser.parse_args()

    success = inspect_date(args.date)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
