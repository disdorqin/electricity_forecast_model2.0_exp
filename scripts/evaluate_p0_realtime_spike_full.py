#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p0_realtime_spike_full.py — Full P0 realtime spike correction evaluation.

Dual-mode operation:

Mode 1 — Profile-based Correction Evaluation (default):
    Runs the correction pipeline with tuning profiles.
    Requires --prediction-pack and --risk-predictions.

    CLI:
        python scripts/evaluate_p0_realtime_spike_full.py \\
            --prediction-pack outputs/prediction_pack.csv \\
            --risk-predictions outputs/risk_predictions.csv \\
            --history outputs/history.csv \\
            --profile conservative \\
            --profile-config config/p0_spike_correction_profiles.yaml \\
            --out-dir reports/local/p0_tuning

    Output per profile:
        - correction_result.csv       — full corrected DataFrame
        - correction_manifest.json    — profile params + metrics
        - metrics_summary.json        — all computed metrics

Mode 2 — Orchestrator (when --steps is provided or --run-all):
    Runs the full P0 evaluation chain over a date window:
      1. Build prediction pack
      2. Diagnose extreme events
      3. Diagnose model regime
      4. Build spike dataset
      5. Train spike risk model
      6. Predict spike risk
      7. Evaluate spike correction
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import (
    CorrectionProfile,
    get_profile,
    load_profile_config,
    run_correction,
    write_correction_manifest,
)
from extreme.realtime_high_spike.residual_lift import get_period


# ═════════════════════════════════════════════════════════════════════
# Mode 1 — Metrics & Correction Evaluation (SA3 profile-based)
# ═════════════════════════════════════════════════════════════════════

def compute_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute sMAPE with floor at 50."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_false_lift_rate(df: pd.DataFrame) -> float:
    """Fraction of non-high-spike hours where correction lifted the prediction."""
    if "high_spike" in df.columns:
        non_spike = df[df["high_spike"] == 0]
    else:
        non_spike = df

    if len(non_spike) == 0:
        return 0.0

    lifted = non_spike[
        (non_spike["final_pred"] > non_spike["base_fused_pred"])
        & (non_spike["lift_applied"] > 0)
    ]
    return float(len(lifted)) / float(len(non_spike))


def compute_normal_hours_degradation(df: pd.DataFrame) -> dict[str, float]:
    """Compute sMAPE delta on normal hours."""
    df = df.copy()
    if "period" not in df.columns:
        df["period"] = df["hour_business"].apply(get_period)

    normal = df[df["period"] != "9_16"]
    if len(normal) == 0:
        return {
            "normal_hours_before_smape_floor50": 0.0,
            "normal_hours_after_smape_floor50": 0.0,
            "normal_hours_degradation": 0.0,
        }

    before = compute_smape(normal["y_true"], normal["base_fused_pred"])
    after = compute_smape(normal["y_true"], normal["final_pred"])

    return {
        "normal_hours_before_smape_floor50": round(before, 4),
        "normal_hours_after_smape_floor50": round(after, 4),
        "normal_hours_degradation": round(after - before, 4),
    }


def compute_all_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all evaluation metrics."""
    metrics: dict[str, Any] = {}

    # Overall sMAPE
    metrics["realtime_overall_smape_floor50"] = round(
        compute_smape(df["y_true"], df["final_pred"]), 4
    )
    metrics["realtime_base_smape_floor50"] = round(
        compute_smape(df["y_true"], df["base_fused_pred"]), 4
    )

    # 9_16 period sMAPE
    df_p = df.copy()
    if "period" not in df_p.columns:
        df_p["period"] = df_p["hour_business"].apply(get_period)
    p9_16 = df_p[df_p["period"] == "9_16"]
    if len(p9_16) > 0:
        metrics["9_16_smape_floor50"] = round(
            compute_smape(p9_16["y_true"], p9_16["final_pred"]), 4
        )
        metrics["9_16_base_smape_floor50"] = round(
            compute_smape(p9_16["y_true"], p9_16["base_fused_pred"]), 4
        )
    else:
        metrics["9_16_smape_floor50"] = None

    # High spike metrics
    if "high_spike" in df.columns:
        spike = df[df["high_spike"] == 1]
        if len(spike) > 0:
            metrics["high_spike_mae"] = round(
                float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))), 4
            )
            metrics["high_spike_base_mae"] = round(
                float(np.mean(np.abs(spike["y_true"] - spike["base_fused_pred"]))), 4
            )
            metrics["high_spike_smape_floor50"] = round(
                compute_smape(spike["y_true"], spike["final_pred"]), 4
            )
        else:
            metrics["high_spike_mae"] = None
            metrics["high_spike_smape_floor50"] = None

    # Severe underestimate
    metrics["severe_underestimate_count"] = int(
        (df["y_true"] - df["final_pred"] > 200).sum()
    )
    metrics["severe_underestimate_base_count"] = int(
        (df["y_true"] - df["base_fused_pred"] > 200).sum()
    )

    # Normal hours degradation
    metrics.update(compute_normal_hours_degradation(df))

    # False lift rate
    metrics["false_lift_rate"] = round(compute_false_lift_rate(df), 4)

    # Count summary
    metrics["total_hours"] = len(df)
    metrics["lift_applied_count"] = int((df["lift_applied"] > 0).sum())
    metrics["lift_capped_count"] = int(
        (df["reason_code"] == "GUARDRAIL_CLIPPED").sum()
    )

    return metrics


def run_evaluation(
    prediction_pack_path: Path,
    risk_predictions_path: Path,
    history_path: Optional[Path],
    profile: CorrectionProfile,
    out_dir: Path,
) -> dict[str, Any]:
    """Run full correction + evaluation for one profile.

    Returns metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    history_df = None
    if history_path is not None and history_path.exists():
        history_df = pd.read_csv(history_path)

    result = run_correction(
        prediction_pack_path=str(prediction_pack_path),
        risk_predictions_path=str(risk_predictions_path),
        history_df=history_df,
        profile=profile,
    )

    result_csv = out_dir / "correction_result.csv"
    result.to_csv(result_csv, index=False)

    metrics = compute_all_metrics(result)

    # Attach profile info
    for key in (
        "profile_used", "spike_prob_threshold", "max_lift_ratio",
        "max_absolute_lift", "protect_normal_hours", "period_9_16_boost",
    ):
        val = getattr(profile, key.replace("profile_used", "name"), None)
        if val is not None:
            metrics[key] = val

    write_correction_manifest(out_dir, profile, metrics=metrics)

    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ═════════════════════════════════════════════════════════════════════
# Mode 2 — Orchestrator (SA2 pipeline runner)
# ═════════════════════════════════════════════════════════════════════

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


def resolve_steps(args: argparse.Namespace) -> list[str]:
    selected = STEPS[:] if args.steps == "all" else [s.strip() for s in args.steps.split(",")]
    if args.skip_steps:
        skip = {s.strip() for s in args.skip_steps.split(",")}
        selected = [s for s in selected if s not in skip]
    return selected


def get_pack_path(args: argparse.Namespace) -> str:
    if args.prediction_pack:
        return args.prediction_pack
    start_compact = args.start_date.replace("-", "_")
    end_compact = args.end_date.replace("-", "_")
    return str(Path(args.out_dir) / "prediction_pack" / f"prediction_pack_realtime_{start_compact}_{end_compact}.csv")


def run_step(step: str, script: str, args: argparse.Namespace, extra: Optional[list[str]] = None) -> int:
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


def write_summary(results: dict[str, str], out_dir: Path, args: argparse.Namespace) -> None:
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


# ═════════════════════════════════════════════════════════════════════
# Unified CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full P0 realtime spike correction evaluation. "
                    "Runs profile-based correction evaluation (default) "
                    "or full pipeline orchestration (with --steps).",
    )

    # --- Mode selection ---
    parser.add_argument("--steps", default=None,
                        help=f"Comma-separated steps: {','.join(STEPS)}, or 'all'. "
                             "If provided, runs orchestrator mode.")
    parser.add_argument("--skip-steps", default=None,
                        help="Comma-separated steps to skip (orchestrator mode).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing (orchestrator mode).")
    parser.add_argument("--models", default="all",
                        help="Models to include (comma-separated or 'all').")
    parser.add_argument("--python", default=sys.executable,
                        help="Python interpreter (orchestrator mode).")

    # --- Universal CLI flags ---
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx",
                        help="Path to raw data")
    parser.add_argument("--runs-root", default="daily_runs",
                        help="Prediction run root directory")
    parser.add_argument("--prediction-pack", default=None,
                        help="Pre-built prediction pack CSV")
    parser.add_argument("--target", default="realtime",
                        choices=["realtime", "dayahead", "both"],
                        help="Market target")
    parser.add_argument("--start-date", default="2025-11-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2025-12-31",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run",
                        help="Root output directory")

    # --- Profile arguments (correction evaluation mode) ---
    parser.add_argument("--risk-predictions", default=None,
                        help="Path to risk predictions CSV (correction evaluation mode)")
    parser.add_argument("--history", default=None,
                        help="Optional historical data CSV for lift quantile fitting")
    parser.add_argument("--profile", default="medium",
                        choices=["conservative", "medium", "aggressive", "all"],
                        help="Correction profile (default: medium). 'all' runs all three.")
    parser.add_argument("--profile-config",
                        default="config/p0_spike_correction_profiles.yaml",
                        help="Profile config file (YAML or JSON)")

    # --- Explicit overrides ---
    parser.add_argument("--spike-prob-threshold", type=float, default=None)
    parser.add_argument("--max-lift-ratio", type=float, default=None)
    parser.add_argument("--max-absolute-lift", type=float, default=None)
    parser.add_argument("--protect-normal-hours", type=str, default=None,
                        choices=["true", "false"])

    return parser.parse_args(argv)


# ═════════════════════════════════════════════════════════════════════
# Main dispatch
# ═════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # Determine mode: orchestrator if --steps is specified, else correction evaluation
    if args.steps is not None:
        # ── Orchestrator mode (SA2) ──
        logger.info("Orchestrator mode: steps=%s", args.steps)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        steps = resolve_steps(args)
        logger.info("Steps: %s", steps)

        results: dict[str, str] = {}
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

    else:
        # ── Correction evaluation mode (SA3) ──
        pp_path = Path(args.prediction_pack) if args.prediction_pack else None
        rp_path = Path(args.risk_predictions) if args.risk_predictions else None

        if pp_path is None or not pp_path.exists():
            sys.exit("Error: prediction pack not found. Provide --prediction-pack or use --steps for orchestrator mode.")
        if rp_path is None or not rp_path.exists():
            sys.exit("Error: risk predictions not found. Provide --risk-predictions or use --steps for orchestrator mode.")

        hist_path = Path(args.history) if args.history else None
        config_path = args.profile_config

        overrides: dict[str, Any] = {}
        if args.spike_prob_threshold is not None:
            overrides["spike_prob_threshold"] = args.spike_prob_threshold
        if args.max_lift_ratio is not None:
            overrides["max_lift_ratio"] = args.max_lift_ratio
        if args.max_absolute_lift is not None:
            overrides["max_absolute_lift"] = args.max_absolute_lift
        if args.protect_normal_hours is not None:
            overrides["protect_normal_hours"] = args.protect_normal_hours.lower() == "true"

        profiles: list[str]
        if args.profile == "all":
            profiles = ["conservative", "medium", "aggressive"]
        else:
            profiles = [args.profile]

        all_metrics: dict[str, Any] = {}

        for pname in profiles:
            print(f"\n{'='*60}")
            print(f"  Profile: {pname}")
            print(f"{'='*60}")

            profile = get_profile(pname, config_path=config_path, overrides=overrides)
            out_dir = Path(args.out_dir) / pname

            metrics = run_evaluation(
                prediction_pack_path=pp_path,
                risk_predictions_path=rp_path,
                history_path=hist_path,
                profile=profile,
                out_dir=out_dir,
            )

            all_metrics[pname] = metrics

            print(f"  overall sMAPE_floor50:    {metrics.get('realtime_overall_smape_floor50', 'N/A')}")
            print(f"  9_16 sMAPE_floor50:       {metrics.get('9_16_smape_floor50', 'N/A')}")
            print(f"  high_spike MAE:           {metrics.get('high_spike_mae', 'N/A')}")
            print(f"  false_lift_rate:          {metrics.get('false_lift_rate', 'N/A')}")
            print(f"  normal_hours_degradation: {metrics.get('normal_hours_degradation', 'N/A')}")

        if len(profiles) > 1:
            summary_path = Path(args.out_dir) / "all_profiles_summary.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(all_metrics, f, indent=2, ensure_ascii=False)
            print(f"\n  Combined: {summary_path}")

        print("\nDone.")


if __name__ == "__main__":
    main()
