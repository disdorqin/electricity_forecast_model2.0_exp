"""Benchmark runtime for each model and pipeline step.

Usage:
    python scripts/benchmark_runtime.py [DATE]

Runs a single-day prediction and reports wall-clock time per model/step.
Reads timing from run_manifest.json if available, otherwise measures subprocess.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def benchmark_date(date_str: str, force: bool = False):
    import subprocess

    date_dir = PROJECT_ROOT / "outputs" / date_str
    cmd = [sys.executable, "main.py", date_str]
    if force:
        cmd.append("--force")

    print(f"=== Benchmark: {date_str} ===")
    print(f"Command: {' '.join(cmd)}")
    print()

    t0 = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    print(f"\nTotal wall-clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Try to read manifest for per-step timing
    manifest_path = date_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        timing = manifest.get("timing", {})
        if timing:
            print(f"\nPer-step timing from manifest:")
            print(f"{'Step':<40} {'Time':>10}")
            print("-" * 52)
            for step, t in sorted(timing.items(), key=lambda x: -x[1]):
                print(f"{step:<40} {t:>8.1f}s")

        steps = manifest.get("steps", {})
        print(f"\nStep status:")
        for k, v in sorted(steps.items()):
            print(f"  {k}: {v}")

    # Check for model-level timing in validation folds
    for target in ["dayahead", "realtime"]:
        folds_dir = date_dir / target / "validation" / "folds"
        if folds_dir.is_dir():
            print(f"\n{target} fold-level timing:")
            for fold_dir in sorted(folds_dir.iterdir()):
                if fold_dir.is_dir():
                    fold_manifest = fold_dir / "fold_manifest.json"
                    if fold_manifest.exists():
                        with open(fold_manifest, encoding="utf-8") as f:
                            fm = json.load(f)
                        ft = fm.get("timing", {})
                        total = sum(ft.values()) if ft else 0
                        print(f"  {fold_dir.name}: {total:.1f}s")

    return elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark pipeline runtime")
    parser.add_argument("date", nargs="?", default="2026-02-01", help="Date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Force rerun")
    args = parser.parse_args()

    benchmark_date(args.date, force=args.force)


if __name__ == "__main__":
    main()
