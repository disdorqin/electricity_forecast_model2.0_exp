#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p0_realtime_spike_full.py — Full P0 realtime spike correction evaluation.

Orchestrates correction + evaluation across one or more tuning profiles.
Designed for offline correction evaluation (not production_pipeline).

CLI:
    # Single profile
    python scripts/evaluate_p0_realtime_spike_full.py \\
        --prediction-pack outputs/prediction_pack.csv \\
        --risk-predictions outputs/risk_predictions.csv \\
        --history outputs/history.csv \\
        --profile conservative \\
        --profile-config config/p0_spike_correction_profiles.yaml \\
        --out-dir reports/local/p0_tuning

    # All three profiles
    python scripts/evaluate_p0_realtime_spike_full.py \\
        --prediction-pack outputs/prediction_pack.csv \\
        --profile all

    # With explicit overrides
    python scripts/evaluate_p0_realtime_spike_full.py \\
        --prediction-pack outputs/prediction_pack.csv \\
        --spike-prob-threshold 0.70 \\
        --max-lift-ratio 0.25

Output per profile:
    - correction_result.csv       — full corrected DataFrame
    - correction_manifest.json    — profile params + metrics
    - metrics_summary.json        — all computed metrics

Combined output (when --profile all):
    - reports/local/p0_tuning/all_profiles_summary.json
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


# ── Metrics (shared with evaluate_realtime_spike_correction) ─────────

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


# ── Run single profile ───────────────────────────────────────────────

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


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full P0 realtime spike correction evaluation",
    )
    # Standard P0 CLI flags (accepted but not all used by this script)
    parser.add_argument("--data-path", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--runs-root", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--target", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--start-date", default=None, help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--end-date", default=None, help="Ignored, kept for orchestrator compatibility")

    parser.add_argument(
        "--prediction-pack", required=True,
        help="Path to prediction pack CSV",
    )
    parser.add_argument(
        "--risk-predictions", required=True,
        help="Path to risk predictions CSV",
    )
    parser.add_argument(
        "--history", default=None,
        help="Optional historical data CSV for lift quantile fitting",
    )
    parser.add_argument(
        "--profile", default="medium",
        choices=["conservative", "medium", "aggressive", "all"],
        help="Correction profile (default: medium). 'all' runs all three.",
    )
    parser.add_argument(
        "--profile-config",
        default="config/p0_spike_correction_profiles.yaml",
        help="Profile config file (YAML or JSON)",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/local/p0_tuning",
        help="Output directory",
    )

    # Explicit overrides
    parser.add_argument("--spike-prob-threshold", type=float, default=None)
    parser.add_argument("--max-lift-ratio", type=float, default=None)
    parser.add_argument("--max-absolute-lift", type=float, default=None)
    parser.add_argument("--protect-normal-hours", type=str, default=None,
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
