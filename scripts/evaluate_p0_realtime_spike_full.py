#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_p0_realtime_spike_full.py — Full P0 window evaluation orchestrator.

Runs the full P0 evaluation chain over a date window:
  1. Build prediction pack
  2. Diagnose extreme events
  3. Diagnose model regime
  4. Build spike dataset
  5. Train spike risk model
  6. Predict spike risk
  7. Evaluate spike correction

Unified CLI:
  --data-path, --runs-root, --prediction-pack, --target,
  --start-date, --end-date, --out-dir
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_p0_realtime_spike_full")

STEPS = [
    "build_pack", "diagnose_extreme", "diagnose_regime",
    "build_spike_dataset", "train_spike_model",
    "predict_spike_risk", "evaluate_correction",
]

STEP_SCRIPTS = {
    "build_pack": "scripts/build_backtest_prediction_pack.py",
    "diagnose_extreme": "scripts/diagnose_extreme_events.py",
    "diagnose_regime": "scripts/diagnose_model_regime.py",
    "build_spike_dataset": "scripts/build_realtime_spike_dataset.py",
    "train_spike_model": "scripts/train_realtime_spike_risk.py",
    "predict_spike_risk": "scripts/predict_realtime_spike_risk.py",
    "evaluate_correction": "scripts/evaluate_realtime_spike_correction.py",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P0 full-window evaluation orchestrator")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data")
    parser.add_argument("--runs-root", default="daily_runs", help="Prediction run root directory")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead", "both"], help="Market target")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run", help="Root output directory")
    parser.add_argument("--steps", default="all", help=f"Comma-separated steps: {','.join(STEPS)}, or 'all'")
    parser.add_argument("--skip-steps", default=None, help="Comma-separated steps to skip")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--models", default="all", help="Models to include (comma-separated or 'all')")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter")
    return parser.parse_args(argv)


def resolve_steps(args) -> list[str]:
    selected = STEPS[:] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    if args.skip_steps:
        skip = {s.strip() for s in args.skip_steps.split(",")}
        selected = [s for s in selected if s not in skip]
    return selected


def get_pack_path(args) -> str:
    if args.prediction_pack:
        return args.prediction_pack
    start_compact = args.start_date.replace("-", "_")
    end_compact = args.end_date.replace("-", "_")
    return str(Path(args.out_dir) / "prediction_pack" / f"prediction_pack_realtime_{start_compact}_{end_compact}.csv")


def run_step(step, script, args, extra=None):
    cmd = [
        args.python, script,
        "--data-path", args.data_path,
        "--runs-root", args.runs_root,
        "--target", args.target,
        "--start-date", args.start_date,
        "--end-date", args.end_date,
    ]
    pack_path = get_pack_path(args)

    if step == "build_pack":
        cmd += ["--out-dir", str(Path(args.out_dir) / "prediction_pack"), "--models", args.models]
    elif step == "diagnose_extreme":
        cmd += ["--out-dir", str(Path(args.out_dir) / "extreme_events")]
        if Path(pack_path).exists():
            cmd += ["--prediction-pack", pack_path]
    elif step == "diagnose_regime":
        cmd += ["--out-dir", str(Path(args.out_dir) / "regime")]
        if Path(pack_path).exists():
            cmd += ["--prediction-pack", pack_path]
    elif step == "build_spike_dataset":
        cmd += ["--out-dir", str(Path(args.out_dir) / "spike_dataset")]
        if Path(pack_path).exists():
            cmd += ["--prediction-pack", pack_path]
    elif step == "train_spike_model":
        cmd += ["--out-dir", str(Path(args.out_dir) / "spike_model")]
        dataset = Path(args.out_dir) / "spike_dataset" / "spike_training_dataset.csv"
        if dataset.exists():
            cmd += ["--dataset", str(dataset)]
    elif step == "predict_spike_risk":
        cmd += ["--out-dir", str(Path(args.out_dir) / "spike_prediction"),
                "--model-dir", str(Path(args.out_dir) / "spike_model")]
        if Path(pack_path).exists():
            cmd += ["--prediction-pack", pack_path]
    elif step == "evaluate_correction":
        cmd += ["--out-dir", str(Path(args.out_dir) / "correction_eval")]
        if Path(pack_path).exists():
            cmd += ["--prediction-pack", pack_path]

    if extra:
        cmd += extra

    cmd_str = " ".join(str(c) for c in cmd)
    if args.dry_run:
        logger.info("[DRY RUN] %s", cmd_str)
        return 0
    logger.info("Running '%s': %s", step, cmd_str)
    return subprocess.run(cmd).returncode


def write_summary(results, out_dir, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation": "P0 Full Window Evaluation",
        "agent": "p0-path-compat",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "step_results": results,
        "prediction_pack": get_pack_path(args),
    }
    (out_dir / "p0_evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# P0 Full Window Evaluation Summary",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Range**: {args.start_date} ~ {args.end_date}",
        "",
        "## Step Results",
        "",
        "| Step | Status |",
        "|------|--------|",
    ]
    for step in STEPS:
        lines.append(f"| {step} | {results.get(step, 'skipped')} |")
    (out_dir / "p0_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    logger.info("Args: %s", args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = resolve_steps(args)
    logger.info("Steps: %s", steps)

    results = {}
    for step in steps:
        script = STEP_SCRIPTS.get(step)
        if not script or not Path(script).exists():
            logger.warning("Script not found for step '%s'", step)
            results[step] = "script_not_found"
            continue
        rc = run_step(step, script, args)
        results[step] = "passed" if rc == 0 else f"failed (rc={rc})"

    write_summary(results, out_dir, args)

    passed = all(v == "passed" for v in results.values())
    logger.info("=" * 60)
    logger.info("P0 EVALUATION COMPLETE")
    for step, status in results.items():
        logger.info("  %s: %s", step, status)
    logger.info("Overall: %s", "ALL PASSED" if passed else "SOME FAILED")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
