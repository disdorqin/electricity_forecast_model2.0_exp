#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p32_rolling_base_correction.py — P3.2 Rolling Base + Spike Correction.

Combines P3.1 rolling severe_softmax base predictions with Phase2 correction
pipeline. The goal is to retain rolling's sMAPE improvement while reducing
severe underestimates to <= 63 (Phase2 best).

Pipeline:
    1. Read P3.1 rolling predictions (severe_softmax) as base_fused_pred
    2. Build prediction pack with rolling predictions as the base
    3. Feed through Phase2 correction (medium / conservative / aggressive profiles)
    4. Evaluate combined metrics vs Phase2 and P3.1 baselines
    5. Produce comparison table and GO/NO-GO verdict

CLI:
    python scripts/evaluate_p32_rolling_base_correction.py \\
        --rolling-predictions reports/local/p31_severe_aware_rolling/severe_softmax/rolling_predictions.csv \\
        --risk-predictions reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/risk_predictions_multicandidate.csv \\
        --profile-config config/p0_spike_correction_profiles.yaml \\
        --out-dir reports/local/p32_rolling_base_correction

Output:
    - p32_prediction_pack.csv       — prediction pack with rolling fused pred
    - {profile}/correction_result.csv  — corrected predictions per profile
    - {profile}/correction_manifest.json
    - comparison_summary.json       — all metrics across profiles + baselines
    - comparison_table.md           — human-readable comparison

GO Rules (P3 Phase 3 combined):
    | Criterion | Threshold | Source |
    |-----------|-----------|--------|
    | sMAPE <= 19.50 | <= 19.50 | P3 rolling sMAPE target |
    | Severe underestimates <= 63 | <= 63 | Phase2 best |
    | False lift rate <= 10% | <= 10% | P0 guardrail |
    | Normal hours degradation <= 0.5 | <= 0.5 | P0 production constraint |

Author: SA4 (automated)
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import (
    CorrectionMode,
    CorrectionProfile,
    get_profile,
    load_profile_config,
    write_correction_manifest,
)
from scripts.evaluate_realtime_spike_correction import (
    compute_all_metrics,
    compute_smape,
)


# ── Baselines (hardcoded from Phase2 + P3.1 evaluations) ───────────────

PHASE2_BEST_METRICS = {
    "realtime_overall_smape_floor50": 20.86,
    "severe_underestimate_count": 63,
    "9_16_smape_floor50": 25.46,
    "total_hours": 2880,
}

P31_BEST_METRICS = {
    "realtime_overall_smape_floor50": 19.10,
    "severe_underestimate_count": 80,
    "9_16_smape_floor50": 26.02,
    "total_hours": 2880,
}

GO_THRESHOLDS = {
    "smape": 19.50,
    "severe": 63,
    "false_lift_rate": 0.10,
    "normal_hours_degradation": 0.50,
}


# ── Prediction pack builder ────────────────────────────────────────────

def build_prediction_pack(
    rolling_predictions_path: Path,
    out_dir: Path,
) -> Path:
    """Build a prediction pack CSV from P3.1 rolling predictions.

    The rolling predictions already have base_fused_pred and y_true.
    We output a minimal prediction pack with 1 row per timestamp.

    Returns:
        Path to the generated prediction pack CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    roll = pd.read_csv(rolling_predictions_path)

    required = {"business_day", "hour_business", "base_fused_pred", "y_true"}
    missing = required - set(roll.columns)
    if missing:
        raise ValueError(f"Rolling predictions missing columns: {missing}")

    # Build minimal prediction pack (1 row per timestamp)
    pack = roll[["business_day", "hour_business", "base_fused_pred", "y_true"]].copy()

    # Add timestamp if available from rolling predictions
    if "timestamp" in roll.columns:
        pack["timestamp"] = roll["timestamp"]

    out_path = out_dir / "p32_prediction_pack.csv"
    pack.to_csv(out_path, index=False)
    print(f"  [INFO] Prediction pack written: {out_path} ({len(pack)} rows)")
    return out_path


# ── Run single profile ─────────────────────────────────────────────────

def run_profile_evaluation(
    prediction_pack_path: Path,
    risk_predictions_path: Path,
    profile: CorrectionProfile,
    out_dir: Path,
    history_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run correction + evaluation for a single profile.

    Returns metrics dict.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load history if provided
    history_df = None
    if history_path is not None and history_path.exists():
        history_df = pd.read_csv(history_path)

    # Import and run correction
    from extreme.realtime_high_spike.apply_correction import run_correction

    result = run_correction(
        prediction_pack_path=str(prediction_pack_path),
        risk_predictions_path=str(risk_predictions_path),
        history_df=history_df,
        profile=profile,
    )

    # Save full corrected result
    result_csv = out_dir / "correction_result.csv"
    result.to_csv(result_csv, index=False)
    print(f"  [INFO] Corrected result: {result_csv} ({len(result)} rows)")

    # Ensure timestamp-level dedup (rolling predictions are already 1-per-timestamp,
    # but correction merge might create duplicates)
    ts_key = None
    for k in ("business_day", "ds_date"):
        if k in result.columns:
            ts_key = k
            break
    hb_key = "hour_business" if "hour_business" in result.columns else "hour"
    if ts_key and hb_key in result.columns:
        n_before = len(result)
        result = result.drop_duplicates(subset=[ts_key, hb_key]).copy()
        n_after = len(result)
        if n_after < n_before:
            print(f"  [INFO] Timestamp-level dedup: {n_before} -> {n_after} rows "
                  f"({(1 - n_after / n_before) * 100:.1f}% reduction)")

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


# ── Comparison ─────────────────────────────────────────────────────────

def build_comparison_table(
    profile_metrics: dict[str, dict[str, Any]],
) -> str:
    """Build a markdown comparison table.

    Args:
        profile_metrics: dict of profile_name -> metrics dict
    """
    rows = []

    # Header
    rows.append("| Metric | Phase2 Best | P3.1 Rolling | P3.2 Medium | P3.2 Conservative | P3.2 Aggressive | GO? |")
    rows.append("|--------|-------------|--------------|-------------|-------------------|-----------------|-----|")

    metric_defs = [
        ("realtime_overall_smape_floor50", "sMAPE (floor50)", "↓", 19.50),
        ("severe_underestimate_count", "Severe underestimates", "↓", 63),
        ("9_16_smape_floor50", "9_16 sMAPE (floor50)", "↓", None),
        ("high_spike_mae", "High-spike MAE", "↓", None),
        ("false_lift_rate", "False lift rate", "↓", 0.10),
        ("normal_hours_degradation", "Normal hours degradation", "↓", 0.50),
        ("lift_applied_count", "Lift applied count", "—", None),
        ("total_hours", "Total hours", "—", None),
    ]

    for key, label, direction, go_threshold in metric_defs:
        row = f"| {label} |"

        # Phase2
        p2_val = PHASE2_BEST_METRICS.get(key, "—")
        row += f" {_fmt(p2_val)} |"

        # P3.1
        p31_val = P31_BEST_METRICS.get(key, "—")
        row += f" {_fmt(p31_val)} |"

        # P3.2 profiles
        for pname in ["medium", "conservative", "aggressive"]:
            if pname in profile_metrics and key in profile_metrics[pname]:
                val = profile_metrics[pname][key]
                row += f" {_fmt(val)} |"
            else:
                row += f" — |"

        # GO check
        if go_threshold is not None and key in profile_metrics.get("medium", {}):
            best_val = profile_metrics["medium"][key]
            if direction == "↓":
                passed = best_val <= go_threshold
            else:
                passed = best_val >= go_threshold
            row += f" {'✅' if passed else '❌'} |"
        else:
            row += f" — |"

        rows.append(row)

    return "\n".join(rows)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    if v is None or v == "—":
        return "—"
    return str(v)


def assess_go(metrics: dict[str, Any]) -> dict[str, Any]:
    """Assess GO/NO-GO for a single profile's metrics.

    Returns verdict dict with per-criterion results.
    """
    criteria = {
        "sMAPE": {
            "threshold": GO_THRESHOLDS["smape"],
            "actual": metrics.get("realtime_overall_smape_floor50"),
            "direction": "≤",
            "met": False,
        },
        "Severe underestimates": {
            "threshold": GO_THRESHOLDS["severe"],
            "actual": metrics.get("severe_underestimate_count"),
            "direction": "≤",
            "met": False,
        },
        "False lift rate": {
            "threshold": GO_THRESHOLDS["false_lift_rate"],
            "actual": metrics.get("false_lift_rate"),
            "direction": "≤",
            "met": False,
        },
        "Normal hours degradation": {
            "threshold": GO_THRESHOLDS["normal_hours_degradation"],
            "actual": metrics.get("normal_hours_degradation"),
            "direction": "≤",
            "met": False,
        },
    }

    for name, c in criteria.items():
        if c["actual"] is not None:
            c["met"] = c["actual"] <= c["threshold"]

    all_met = all(c["met"] for c in criteria.values() if c["actual"] is not None)

    return {
        "verdict": "GO" if all_met else "NO-GO",
        "criteria": criteria,
        "all_criteria_met": all_met,
    }


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3.2 Rolling Base + Spike Correction evaluation.",
    )
    parser.add_argument(
        "--rolling-predictions", required=True,
        help="Path to P3.1 rolling predictions CSV (with base_fused_pred, y_true)",
    )
    parser.add_argument(
        "--risk-predictions", required=True,
        help="Path to Phase2 risk predictions CSV (with high_spike_prob)",
    )
    parser.add_argument(
        "--profile-config",
        default="config/p0_spike_correction_profiles.yaml",
        help="Path to profile configuration file",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/local/p32_rolling_base_correction",
        help="Output directory for results",
    )
    parser.add_argument(
        "--correction-mode", default="normal",
        choices=["normal", "relaxed"],
        help="Correction strictness (default: normal)",
    )
    parser.add_argument(
        "--history",
        default=None,
        help="Optional path to historical data CSV for fitting lift quantiles",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    roll_path = Path(args.rolling_predictions)
    risk_path = Path(args.risk_predictions)
    hist_path = Path(args.history) if args.history else None
    config_path = args.profile_config

    if not roll_path.exists():
        sys.exit(f"Error: rolling predictions not found: {roll_path}")
    if not risk_path.exists():
        sys.exit(f"Error: risk predictions not found: {risk_path}")

    correction_mode = CorrectionMode(args.correction_mode)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build prediction pack from rolling predictions ────────
    print("\n" + "=" * 60)
    print("  P3.2 Rolling Base + Spike Correction")
    print("=" * 60)
    print(f"\n  Rolling predictions: {roll_path}")
    print(f"  Risk predictions:    {risk_path}")
    print(f"  Correction mode:     {correction_mode.value}")

    pack_path = build_prediction_pack(roll_path, out_dir)

    # ── Step 2: Run correction for each profile ───────────────────────
    profiles_to_run = ["medium", "conservative", "aggressive"]
    all_metrics: dict[str, dict[str, Any]] = {}

    for pname in profiles_to_run:
        print(f"\n  {'─' * 50}")
        print(f"  Profile: {pname}")
        print(f"  {'─' * 50}")

        profile = get_profile(pname, config_path=config_path, mode=correction_mode)

        profile_out_dir = out_dir / pname
        metrics = run_profile_evaluation(
            prediction_pack_path=pack_path,
            risk_predictions_path=risk_path,
            profile=profile,
            out_dir=profile_out_dir,
            history_path=hist_path,
        )
        all_metrics[pname] = metrics

        # Print summary
        print(f"    overall sMAPE_floor50:     {metrics.get('realtime_overall_smape_floor50', 'N/A')}")
        print(f"    base sMAPE_floor50:        {metrics.get('realtime_base_smape_floor50', 'N/A')}")
        print(f"    9_16 sMAPE_floor50:        {metrics.get('9_16_smape_floor50', 'N/A')}")
        print(f"    high_spike MAE:            {metrics.get('high_spike_mae', 'N/A')}")
        print(f"    high_spike_base_MAE:       {metrics.get('high_spike_base_mae', 'N/A')}")
        print(f"    false_lift_rate:           {metrics.get('false_lift_rate', 'N/A')}")
        print(f"    normal_hours_degradation:  {metrics.get('normal_hours_degradation', 'N/A')}")
        print(f"    severe_underestimate:      {metrics.get('severe_underestimate_count', 'N/A')}")
        print(f"    severe_underestimate_base: {metrics.get('severe_underestimate_base_count', 'N/A')}")
        print(f"    lift_applied_count:        {metrics.get('lift_applied_count', 'N/A')}")
        print(f"    lift_capped_count:         {metrics.get('lift_capped_count', 'N/A')}")
        print(f"    Output: {profile_out_dir}")

    # ── Step 3: Comparison table ──────────────────────────────────────
    print(f"\n  {'=' * 60}")
    print("  Comparison vs Baselines")
    print(f"  {'=' * 60}")

    comparison_table = build_comparison_table(all_metrics)
    print("\n" + comparison_table)

    # Save comparison table
    table_path = out_dir / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# P3.2 Rolling Base + Spike Correction — Comparison\n\n")
        f.write(comparison_table)
        f.write("\n")
    print(f"\n  Comparison table: {table_path}")

    # ── Step 4: GO/NO-GO assessment ───────────────────────────────────
    print(f"\n  {'=' * 60}")
    print("  GO / NO-GO Assessment")
    print(f"  {'=' * 60}")

    verdicts = {}
    for pname in profiles_to_run:
        verdict = assess_go(all_metrics[pname])
        verdicts[pname] = verdict
        met_count = sum(1 for c in verdict["criteria"].values() if c["met"])
        total = sum(1 for c in verdict["criteria"].values() if c["actual"] is not None)
        print(f"\n  {pname}: {verdict['verdict']} ({met_count}/{total} criteria met)")
        for cname, c in verdict["criteria"].items():
            if c["actual"] is not None:
                mark = "✅" if c["met"] else "❌"
                print(f"    {mark} {cname}: {c['actual']} {c['direction']} {c['threshold']}")

    # ── Step 5: Summary JSON ──────────────────────────────────────────
    summary = {
        "script": "scripts/evaluate_p32_rolling_base_correction.py",
        "rolling_predictions": str(roll_path),
        "risk_predictions": str(risk_path),
        "correction_mode": correction_mode.value,
        "baselines": {
            "phase2_best": PHASE2_BEST_METRICS,
            "p31_best": P31_BEST_METRICS,
        },
        "go_thresholds": GO_THRESHOLDS,
        "profiles": all_metrics,
        "verdicts": verdicts,
    }

    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary: {summary_path}")

    # Print final verdict
    best_profile = min(
        verdicts.items(),
        key=lambda x: (
            0 if x[1]["verdict"] == "GO" else 1,
            all_metrics[x[0]].get("severe_underestimate_count", 999),
        ),
    )[0]

    print(f"\n  {'=' * 60}")
    print(f"  BEST PROFILE: {best_profile}")
    print(f"  VERDICT: {verdicts[best_profile]['verdict']}")
    print(f"  {'=' * 60}")
    print("\nDone.")


if __name__ == "__main__":
    main()
