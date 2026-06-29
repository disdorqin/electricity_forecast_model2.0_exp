#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_realtime_spike_correction.py — Evaluate P0 spike correction for a single profile.

CLI:
    python scripts/evaluate_realtime_spike_correction.py \\
        --prediction-pack outputs/prediction_pack.csv \\
        --risk-predictions outputs/risk_predictions.csv \\
        --history outputs/history.csv \\
        --profile conservative \\
        --profile-config config/p0_spike_correction_profiles.yaml \\
        --out-dir reports/local/p0_tuning

    # Explicit overrides (take precedence over profile)
    python scripts/evaluate_realtime_spike_correction.py \\
        --prediction-pack ... \\
        --spike-prob-threshold 0.70 \\
        --max-lift-ratio 0.25

    # Run all three profiles
    python scripts/evaluate_realtime_spike_correction.py \\
        --prediction-pack ... \\
        --profile all

Output:
    - correction_result.csv       — full DataFrame with corrected predictions
    - correction_manifest.json    — profile params + metrics
    - metrics_summary.json        — all computed metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
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


# ── Metrics ──────────────────────────────────────────────────────────

def compute_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute symmetric Mean Absolute Percentage Error (sMAPE), floored at 50."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    # Avoid division by zero: where denom is 0, sMAPE = 0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    # Floor at 50
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_false_lift_rate(df: pd.DataFrame) -> float:
    """Fraction of non-high-spike hours where final_pred > base_fused_pred and lift > 0.

    false_lift_rate = (non-spike hours with lift) / (total non-spike hours)

    Non-spike hours are defined as hours where the residual between base_fused_pred
    and y_true is not extreme — i.e., not a true high spike event.
    """
    if "high_spike" in df.columns:
        non_spike = df[df["high_spike"] == 0]
    else:
        # Fallback: if no label column, use all rows
        non_spike = df

    if len(non_spike) == 0:
        return 0.0

    lifted = non_spike[
        (non_spike["final_pred"] > non_spike["base_fused_pred"])
        & (non_spike["lift_applied"] > 0)
    ]
    return float(len(lifted)) / float(len(non_spike))


def compute_normal_hours_degradation(
    df: pd.DataFrame,
    period_col: str = "period",
) -> dict[str, float]:
    """Compute sMAPE delta on normal (non-9_16) hours before and after correction.

    normal_hours_degradation = normal_after_smape - normal_before_smape

    Returns dict with before/after sMAPE and the delta.
    """
    if period_col not in df.columns:
        df = df.copy()
        df[period_col] = df["hour_business"].apply(get_period)

    normal = df[df[period_col] != "9_16"].copy()

    if len(normal) == 0:
        return {
            "normal_hours_before_smape_floor50": 0.0,
            "normal_hours_after_smape_floor50": 0.0,
            "normal_hours_degradation": 0.0,
        }

    before_smape = compute_smape(
        normal["y_true"], normal["base_fused_pred"]
    )
    after_smape = compute_smape(
        normal["y_true"], normal["final_pred"]
    )

    return {
        "normal_hours_before_smape_floor50": round(before_smape, 4),
        "normal_hours_after_smape_floor50": round(after_smape, 4),
        "normal_hours_degradation": round(after_smape - before_smape, 4),
    }


def compute_all_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute all evaluation metrics from corrected DataFrame.

    Metrics:
        - realtime overall sMAPE_floor50
        - 9_16 sMAPE_floor50
        - high_spike MAE
        - high_spike sMAPE_floor50
        - severe_underestimate_count
        - normal_hours_degradation
        - false_lift_rate
    """
    metrics: dict[str, Any] = {}

    # overall sMAPE_floor50
    metrics["realtime_overall_smape_floor50"] = round(
        compute_smape(df["y_true"], df["final_pred"]), 4
    )

    # base sMAPE_floor50 (before correction)
    metrics["realtime_base_smape_floor50"] = round(
        compute_smape(df["y_true"], df["base_fused_pred"]), 4
    )

    # 9_16 sMAPE_floor50
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
            # MAE
            metrics["high_spike_mae"] = round(
                float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))), 4
            )
            metrics["high_spike_base_mae"] = round(
                float(np.mean(np.abs(spike["y_true"] - spike["base_fused_pred"]))), 4
            )
            # sMAPE_floor50
            metrics["high_spike_smape_floor50"] = round(
                compute_smape(spike["y_true"], spike["final_pred"]), 4
            )
            metrics["high_spike_base_smape_floor50"] = round(
                compute_smape(spike["y_true"], spike["base_fused_pred"]), 4
            )
        else:
            metrics["high_spike_mae"] = None
            metrics["high_spike_smape_floor50"] = None
    else:
        # Try to infer high_spike from residual
        residual = df["y_true"] - df["base_fused_pred"]
        threshold = residual.quantile(0.95)
        spike_mask = residual >= threshold
        spike = df[spike_mask]
        if len(spike) > 0:
            metrics["high_spike_mae"] = round(
                float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))), 4
            )
            metrics["high_spike_smape_floor50"] = round(
                compute_smape(spike["y_true"], spike["final_pred"]), 4
            )
        else:
            metrics["high_spike_mae"] = None
            metrics["high_spike_smape_floor50"] = None

    # Severe underestimate count: y_true - final_pred > 200
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
    metrics["lift_applied_count"] = int(
        (df["lift_applied"] > 0).sum()
    )
    metrics["lift_capped_count"] = int(
        (df["reason_code"] == "GUARDRAIL_CLIPPED").sum()
    )
    metrics["lift_rejected_low_prob"] = int(
        (df["reason_code"] == "NO_CORRECTION_LOW_PROB").sum()
    )
    metrics["lift_rejected_negative_base"] = int(
        (df["reason_code"] == "NO_CORRECTION_NEGATIVE_BASE").sum()
    )
    metrics["lift_rejected_normal_hour"] = int(
        (df["reason_code"] == "NO_CORRECTION_NORMAL_HOUR").sum()
    )

    return metrics


# ── Run single profile ───────────────────────────────────────────────

def run_evaluation(
    prediction_pack_path: Path,
    risk_predictions_path: Path,
    history_path: Optional[Path],
    profile: CorrectionProfile,
    out_dir: Path,
) -> dict[str, Any]:
    """Run evaluation for a single profile.

    Returns metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load history if provided
    history_df = None
    if history_path is not None and history_path.exists():
        history_df = pd.read_csv(history_path)

    # Run correction
    result = run_correction(
        prediction_pack_path=str(prediction_pack_path),
        risk_predictions_path=str(risk_predictions_path),
        history_df=history_df,
        profile=profile,
    )

    # Save full result
    result_csv = out_dir / "correction_result.csv"
    result.to_csv(result_csv, index=False)

    # Compute metrics
    metrics = compute_all_metrics(result)

    # Attach profile info
    metrics["profile_used"] = profile.name
    metrics["spike_prob_threshold"] = profile.spike_prob_threshold
    metrics["max_lift_ratio"] = profile.max_lift_ratio
    metrics["max_absolute_lift"] = profile.max_absolute_lift
    metrics["protect_normal_hours"] = profile.protect_normal_hours
    metrics["period_9_16_boost"] = profile.period_9_16_boost

    # Write manifest
    write_correction_manifest(out_dir, profile, metrics=metrics)

    # Write metrics summary
    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P0 realtime spike correction for a profile.",
    )
    # Standard P0 CLI flags (accepted but not all used by this script)
    parser.add_argument("--data-path", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--runs-root", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--target", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--start-date", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--end-date", default=None, help="Ignored, kept for orchestrator compatibility")

    parser.add_argument(
        "--prediction-pack", required=True,
        help="Path to prediction pack CSV (with base_fused_pred, y_true, ...)",
    )
    parser.add_argument(
        "--risk-predictions", required=True,
        help="Path to risk predictions CSV (with high_spike_prob, ...)",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional path to historical data CSV for fitting lift quantiles",
    )
    parser.add_argument(
        "--profile", default="medium",
        choices=["conservative", "medium", "aggressive", "all"],
        help="Correction profile to use (default: medium). 'all' runs all three.",
    )
    parser.add_argument(
        "--profile-config",
        default="config/p0_spike_correction_profiles.yaml",
        help="Path to profile configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/local/p0_tuning",
        help="Output directory for results",
    )

    # Explicit overrides (take precedence over profile)
    parser.add_argument("--spike-prob-threshold", type=float, default=None)
    parser.add_argument("--max-lift-ratio", type=float, default=None)
    parser.add_argument("--max-absolute-lift", type=float, default=None)
    parser.add_argument("--protect-normal-hours", type=str, default=None,
                        choices=["true", "false"])
    parser.add_argument("--period-aware", type=str, default=None,
                        choices=["true", "false"])

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    pp_path = Path(args.prediction_pack)
    rp_path = Path(args.risk_predictions)
    hist_path = Path(args.history) if args.history else None
    config_path = args.profile_config

    if not pp_path.exists():
        sys.exit(f"Error: prediction pack not found: {pp_path}")
    if not rp_path.exists():
        sys.exit(f"Error: risk predictions not found: {rp_path}")

    # Build overrides dict from explicit CLI args
    overrides: dict[str, Any] = {}
    if args.spike_prob_threshold is not None:
        overrides["spike_prob_threshold"] = args.spike_prob_threshold
    if args.max_lift_ratio is not None:
        overrides["max_lift_ratio"] = args.max_lift_ratio
    if args.max_absolute_lift is not None:
        overrides["max_absolute_lift"] = args.max_absolute_lift
    if args.protect_normal_hours is not None:
        overrides["protect_normal_hours"] = args.protect_normal_hours.lower() == "true"

    profiles_to_run: list[str]
    if args.profile == "all":
        profiles_to_run = ["conservative", "medium", "aggressive"]
    else:
        profiles_to_run = [args.profile]

    all_metrics: dict[str, Any] = {}

    for pname in profiles_to_run:
        print(f"\n{'='*60}")
        print(f"  Running profile: {pname}")
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

        # Print summary
        print(f"  overall sMAPE_floor50:    {metrics.get('realtime_overall_smape_floor50', 'N/A')}")
        print(f"  base sMAPE_floor50:       {metrics.get('realtime_base_smape_floor50', 'N/A')}")
        print(f"  9_16 sMAPE_floor50:       {metrics.get('9_16_smape_floor50', 'N/A')}")
        print(f"  high_spike MAE:           {metrics.get('high_spike_mae', 'N/A')}")
        print(f"  false_lift_rate:          {metrics.get('false_lift_rate', 'N/A')}")
        print(f"  normal_hours_degradation: {metrics.get('normal_hours_degradation', 'N/A')}")
        print(f"  severe_underestimate:     {metrics.get('severe_underestimate_count', 'N/A')}")
        print(f"  Output: {out_dir}")

    # Write combined summary if running 3 profiles
    if len(profiles_to_run) > 1:
        summary_path = Path(args.out_dir) / "all_profiles_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        print(f"\n  Combined summary: {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
