#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_p33_spike_gated_uplift.py — Evaluate P3.3 spike-gated uplift correction.

Reuses the existing correction pipeline infrastructure, replacing the
generic spike_risk_score with gate-enhanced probabilities from the
P3.3 trained uplift gate.

Pipeline:
  1. Load prediction pack + gate risk predictions
  2. Apply correction with medium profile (or custom overrides)
  3. Compute evaluation metrics
  4. Compare with Phase 2 baseline (from execution board)
  5. Generate summary report

Usage:
    python scripts/evaluate_p33_spike_gated_uplift.py
        --prediction-pack <multicandidate_pack.csv>
        --gate-risk-predictions <gate_risk_predictions.csv>
        --out-dir reports/local/p33_spike_gated_uplift/evaluation

Output:
    <out-dir>/
      - correction_result.csv
      - correction_manifest.json
      - metrics_summary.json
      - comparison_with_phase2.json
    docs/reports/P33_spike_gated_uplift_report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import (
    CorrectionProfile,
    CorrectionMode,
    get_profile,
    run_correction,
    write_correction_manifest,
    diagnose_zero_lift,
)
from extreme.realtime_high_spike.residual_lift import get_period


# ── Metrics ──────────────────────────────────────────────────────────

PHASE2_BASELINE = {
    "medium": {
        "smape_floor50": 25.67,
        "severe_underestimate": 125,
        "false_lift_rate": 0.0812,
        "normal_hours_degradation": None,
    },
    "base_before_correction": {
        "smape_floor50": 21.20,
        "severe_underestimate": 81,
    },
}


def compute_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """sMAPE with 50 floor."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_false_lift_rate(df: pd.DataFrame) -> float:
    """Fraction of non-high-spike hours where lift > 0."""
    non_spike = df[df.get("high_spike", pd.Series(0, index=df.index)) == 0]
    if len(non_spike) == 0:
        return 0.0
    lifted = non_spike[
        (non_spike["final_pred"] > non_spike["base_fused_pred"])
        & (non_spike["lift_applied"] > 0)
    ]
    return float(len(lifted)) / float(len(non_spike))


def compute_normal_hours_degradation(df: pd.DataFrame) -> dict[str, float]:
    """sMAPE delta on non-9_16 hours."""
    df = df.copy()
    if "period" not in df.columns:
        df["period"] = df["hour_business"].apply(get_period)
    normal = df[df["period"] != "9_16"]
    if len(normal) == 0:
        return {"normal_hours_degradation": 0.0}
    before = compute_smape(normal["y_true"], normal["base_fused_pred"])
    after = compute_smape(normal["y_true"], normal["final_pred"])
    return {
        "normal_hours_before": round(before, 4),
        "normal_hours_after": round(after, 4),
        "normal_hours_degradation": round(after - before, 4),
    }


def compute_all_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Full metrics computation (timestamp-level)."""
    metrics: dict[str, Any] = {}

    # Deduplicate to timestamp level
    ts_df = df.drop_duplicates(subset=["business_day", "hour_business"]).copy()
    metrics["n_timestamps"] = len(ts_df)

    # Overall sMAPE
    metrics["overall_smape_floor50"] = round(
        compute_smape(ts_df["y_true"], ts_df["final_pred"]), 4
    )
    metrics["base_smape_floor50"] = round(
        compute_smape(ts_df["y_true"], ts_df["base_fused_pred"]), 4
    )

    # 9_16 sMAPE
    p9_16 = ts_df[ts_df["period"] == "9_16"]
    if len(p9_16) > 0:
        metrics["9_16_smape_floor50"] = round(
            compute_smape(p9_16["y_true"], p9_16["final_pred"]), 4
        )

    # High spike metrics
    spike = ts_df[ts_df.get("high_spike", pd.Series(0, index=ts_df.index)) == 1]
    if len(spike) > 0:
        metrics["high_spike_mae"] = round(
            float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))), 4
        )
        metrics["high_spike_smape_floor50"] = round(
            compute_smape(spike["y_true"], spike["final_pred"]), 4
        )

    # Severe underestimate
    metrics["severe_underestimate"] = int(
        (ts_df["y_true"] - ts_df["final_pred"] > 200).sum()
    )
    metrics["severe_underestimate_base"] = int(
        (ts_df["y_true"] - ts_df["base_fused_pred"] > 200).sum()
    )

    # Normal hours degradation
    metrics.update(compute_normal_hours_degradation(ts_df))

    # False lift rate
    metrics["false_lift_rate"] = round(compute_false_lift_rate(ts_df), 4)

    # Lift counts
    metrics["lift_applied_count"] = int((ts_df["lift_applied"] > 0).sum())
    metrics["lift_capped_count"] = int(
        (ts_df["reason_code"] == "GUARDRAIL_CLIPPED").sum()
    )
    metrics["lift_rejected_low_prob"] = int(
        (ts_df["reason_code"] == "NO_CORRECTION_LOW_PROB").sum()
    )
    metrics["lift_rejected_negative_base"] = int(
        (ts_df["reason_code"] == "NO_CORRECTION_NEGATIVE_BASE").sum()
    )
    metrics["lift_rejected_normal_hour"] = int(
        (ts_df["reason_code"] == "NO_CORRECTION_NORMAL_HOUR").sum()
    )

    return metrics


def compare_with_phase2(
    p33_metrics: dict[str, Any],
    profile_name: str = "medium",
) -> dict[str, Any]:
    """Compare P3.3 results with Phase 2 baseline."""
    phase2 = PHASE2_BASELINE.get(profile_name, PHASE2_BASELINE["medium"])
    base = PHASE2_BASELINE["base_before_correction"]

    comparison: dict[str, Any] = {
        "phase2_baseline": phase2,
        "phase2_base_before_correction": base,
        "p33_results": {
            "smape_floor50": p33_metrics.get("overall_smape_floor50"),
            "severe_underestimate": p33_metrics.get("severe_underestimate"),
            "false_lift_rate": p33_metrics.get("false_lift_rate"),
            "normal_hours_degradation": p33_metrics.get("normal_hours_degradation"),
        },
        "delta_vs_phase2": {},
        "meets_target": {},
        "targets": {
            "smape_floor50_max": 20.50,
            "severe_underestimate_max": 63,
            "false_lift_rate_max": 0.10,
        },
    }

    # Compute deltas
    p33_smape = p33_metrics.get("overall_smape_floor50", 999)
    p33_severe = p33_metrics.get("severe_underestimate", 999)
    p33_false_lift = p33_metrics.get("false_lift_rate", 999)

    comparison["delta_vs_phase2"] = {
        "smape_floor50": round(p33_smape - phase2["smape_floor50"], 4) if p33_smape else None,
        "severe_underestimate": int(p33_severe - phase2["severe_underestimate"]) if p33_severe else None,
        "false_lift_rate": round(p33_false_lift - phase2["false_lift_rate"], 4) if p33_false_lift else None,
    }

    # Check targets
    targets = comparison["targets"]
    comparison["meets_target"] = {
        "smape_floor50": p33_smape <= targets["smape_floor50_max"] if p33_smape else False,
        "severe_underestimate": p33_severe <= targets["severe_underestimate_max"] if p33_severe else False,
        "false_lift_rate": p33_false_lift <= targets["false_lift_rate_max"] if p33_false_lift else False,
    }

    comparison["all_targets_met"] = all(comparison["meets_target"].values())
    comparison["beats_phase2"] = (
        p33_severe < phase2["severe_underestimate"]
        if (p33_severe is not None and phase2["severe_underestimate"] is not None)
        else None
    )

    return comparison


def generate_report(
    metrics: dict[str, Any],
    comparison: dict[str, Any],
    profile_name: str,
    args: argparse.Namespace,
) -> str:
    """Generate the P3.3 spike-gated uplift report markdown."""
    p33 = comparison["p33_results"]
    phase2 = comparison["phase2_baseline"]
    targets = comparison["targets"]
    meets = comparison["meets_target"]
    beats = comparison.get("beats_phase2", False)
    all_met = comparison["all_targets_met"]

    # Assessment
    if all_met and beats:
        assessment = "GO"
        assessment_reason = "All targets met AND beats Phase 2 severe underestimate count."
    elif all_met:
        assessment = "GO"
        assessment_reason = "All targets met."
    elif beats:
        assessment = "CONDITIONAL"
        assessment_reason = "Beats Phase 2 severe count but does not meet all targets."
    else:
        assessment = "NO-GO"
        assessment_reason = "Does not meet targets and does not improve over Phase 2."

    lines = []
    L = lines.append

    L(f"# P3.3 Spike-Gated Uplift — Evaluation Report")
    L(f"")
    L(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    L(f"**Profile**: {profile_name}")
    L(f"**Model**: {args.model_type}")
    L(f"**Threshold**: {args.threshold}")
    L(f"**Window**: {args.start_date} ~ {args.end_date}")
    L(f"")
    L(f"## Summary")
    L(f"")
    L(f"| Metric | P3.3 Result | Phase 2 Baseline | Target | Meets? |")
    L(f"|--------|------------|-----------------|--------|--------|")
    L(f"| sMAPE (floor50) | {p33.get('smape_floor50', 'N/A')} | {phase2.get('smape_floor50', 'N/A')} | ≤{targets['smape_floor50_max']} | {'✅' if meets['smape_floor50'] else '❌'} |")
    L(f"| Severe Underestimates | {p33.get('severe_underestimate', 'N/A')} | {phase2.get('severe_underestimate', 'N/A')} | ≤{targets['severe_underestimate_max']} | {'✅' if meets['severe_underestimate'] else '❌'} |")
    L(f"| False Lift Rate | {p33.get('false_lift_rate', 'N/A')} | {phase2.get('false_lift_rate', 'N/A')} | ≤{targets['false_lift_rate_max']} | {'✅' if meets['false_lift_rate'] else '❌'} |")
    L(f"| Normal Hours Degradation | {p33.get('normal_hours_degradation', 'N/A')} | {phase2.get('normal_hours_degradation', 'N/A')} | — | — |")
    L(f"")
    L(f"**Assessment**: **{assessment}** — {assessment_reason}")
    L(f"")
    L(f"## Detailed Metrics")
    L(f"")
    L(f"| Metric | Value |")
    L(f"|--------|-------|")
    for key, val in sorted(metrics.items()):
        L(f"| {key} | {val} |")
    L(f"")
    L(f"## Comparison with Phase 2")
    L(f"")
    L(f"| Metric | P3.3 | Phase 2 | Delta |")
    L(f"|--------|------|---------|-------|")
    delta = comparison["delta_vs_phase2"]
    for metric_key in ["smape_floor50", "severe_underestimate", "false_lift_rate"]:
        p33v = p33.get(metric_key, "N/A")
        p2v = phase2.get(metric_key, "N/A")
        dv = delta.get(metric_key, "N/A")
        arrow = "↓" if dv is not None and dv < 0 else ("↑" if dv is not None and dv > 0 else "→")
        L(f"| {metric_key} | {p33v} | {p2v} | {arrow} {dv} |")
    L(f"")
    L(f"## Configuration")
    L(f"")
    L(f"| Parameter | Value |")
    L(f"|-----------|-------|")
    L(f"| Model | {args.model_type} |")
    L(f"| Threshold | {args.threshold} |")
    L(f"| Lift quantile | {args.lift_quantile} |")
    L(f"| Eval window | {args.start_date} ~ {args.end_date} |")
    L(f"| Pack | {Path(args.prediction_pack).name if args.prediction_pack else 'default'} |")
    L(f"")
    L(f"## Feature Importance (Top 10)")
    L(f"")
    L(f"See `feature_importance.csv` in the training output directory for full list.")
    L(f"")
    L(f"## Known Limitations")
    L(f"")
    L(f"- Rolling window requires 30-day warmup; first 30 days of P0 window excluded from evaluation.")
    L(f"- Gate only controls *when* to lift, not *how much*; lift amount uses historical quantiles.")
    L(f"- Feature set limited to prediction-time safe columns only.")
    L(f"- Simple classifier may not capture all severe event patterns.")
    L(f"")
    L(f"---")
    L(f"*Generated by `evaluate_p33_spike_gated_uplift.py` at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines)


# ── CLI ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate P3.3 spike-gated uplift correction.",
    )
    parser.add_argument("--prediction-pack", default=None,
                        help="Path to multi-candidate prediction pack CSV")
    parser.add_argument("--gate-risk-predictions", default=None,
                        help="Path to gate risk predictions CSV (from train_p33)")
    parser.add_argument("--out-dir", default="reports/local/p33_spike_gated_uplift/evaluation",
                        help="Output directory for evaluation results")
    parser.add_argument("--start-date", default="2025-11-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-02-28",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--profile", default="medium",
                        choices=["conservative", "medium", "aggressive"],
                        help="Correction profile")
    parser.add_argument("--correction-mode", default="normal",
                        choices=["normal", "relaxed"],
                        help="Correction mode")
    parser.add_argument("--profile-config",
                        default="config/p0_spike_correction_profiles.yaml",
                        help="Profile config file")
    parser.add_argument("--model-type", default="rf",
                        help="Model type used (for report)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Gate threshold used (for report)")
    parser.add_argument("--lift-quantile", type=float, default=0.90,
                        help="Lift quantile used (for report)")
    return parser.parse_args(argv)


def resolve_default_pack_path() -> Path:
    return (
        _PROJECT_ROOT
        / "reports/local/p0_full_run/prediction_pack_multicandidate"
        / "prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv"
    )


def resolve_default_gate_risk_path() -> Path:
    return (
        _PROJECT_ROOT
        / "reports/local/p33_spike_gated_uplift"
        / "gate_risk_predictions.csv"
    )


def main() -> None:
    args = parse_args()

    # Resolve paths
    pp_path = Path(args.prediction_pack) if args.prediction_pack else resolve_default_pack_path()
    grp_path = Path(args.gate_risk_predictions) if args.gate_risk_predictions else resolve_default_gate_risk_path()

    if not pp_path.exists():
        print(f"  [ERR] Prediction pack not found: {pp_path}")
        sys.exit(1)
    if not grp_path.exists():
        print(f"  [ERR] Gate risk predictions not found: {grp_path}")
        print(f"        Run train_p33_spike_gated_uplift.py first.")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Correction mode
    try:
        correction_mode = CorrectionMode(args.correction_mode)
    except ValueError:
        sys.exit(f"Error: invalid --correction-mode '{args.correction_mode}'")

    print("=" * 60)
    print("  P3.3 — Evaluate Spike-Gated Uplift")
    print("=" * 60)
    print(f"  Profile:       {args.profile}")
    print(f"  Correction:    {correction_mode.value}")
    print(f"  Pack:          {pp_path.name}")
    print(f"  Gate risk:     {grp_path.name}")
    print()

    # ── 1. Resolve profile ───────────────────────────────────────────
    profile = get_profile(
        args.profile,
        config_path=args.profile_config,
        mode=correction_mode,
    )
    eff = profile.to_dict_effective()
    print(f"  Effective thresholds:")
    print(f"    spike_prob_threshold → {eff.get('effective_spike_prob_threshold', '?')}")
    print(f"    lift_floor           → {eff.get('lift_floor_applied', 0)}")

    # ── 2. Run correction with gate risk predictions ──────────────────
    print(f"\n  Running correction...")
    result = run_correction(
        prediction_pack_path=str(pp_path),
        risk_predictions_path=str(grp_path),
        history_df=None,
        profile=profile,
    )
    print(f"  -> {len(result)} rows after correction")

    # Save full result
    result_csv = out_dir / "correction_result.csv"
    result.to_csv(result_csv, index=False)
    print(f"  [OK] Correction result: {result_csv}")

    # ── 3. Compute metrics (timestamp-level) ─────────────────────────
    metrics = compute_all_metrics(result)
    metrics["profile_used"] = args.profile
    metrics["correction_mode"] = args.correction_mode

    print(f"\n  ── P3.3 Correction Metrics ──")
    print(f"    overall sMAPE_floor50:    {metrics.get('overall_smape_floor50', 'N/A')}")
    print(f"    base sMAPE_floor50:       {metrics.get('base_smape_floor50', 'N/A')}")
    print(f"    9_16 sMAPE_floor50:       {metrics.get('9_16_smape_floor50', 'N/A')}")
    print(f"    high_spike MAE:           {metrics.get('high_spike_mae', 'N/A')}")
    print(f"    severe_underestimate:     {metrics.get('severe_underestimate', 'N/A')} "
          f"(base: {metrics.get('severe_underestimate_base', 'N/A')})")
    print(f"    false_lift_rate:          {metrics.get('false_lift_rate', 'N/A')}")
    print(f"    normal_hours_degradation: {metrics.get('normal_hours_degradation', 'N/A')}")
    print(f"    lift_applied_count:       {metrics.get('lift_applied_count', 'N/A')}")

    # Diagnose zero-lift
    if (out_dir / "correction_result.csv").exists():
        result_df = pd.read_csv(result_csv)
        diagnose_zero_lift(result_df, top_n=10)

    # Write manifest
    write_correction_manifest(out_dir, profile, metrics=metrics)

    # Write metrics summary
    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Metrics: {metrics_path}")

    # ── 4. Compare with Phase 2 ──────────────────────────────────────
    comparison = compare_with_phase2(metrics, profile_name=args.profile)

    comparison_path = out_dir / "comparison_with_phase2.json"
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Phase 2 comparison: {comparison_path}")

    # Print comparison
    phase2 = comparison["phase2_baseline"]
    p33 = comparison["p33_results"]
    meets = comparison["meets_target"]
    beats = comparison.get("beats_phase2", False)
    all_met = comparison["all_targets_met"]

    print(f"\n  ── Phase 2 Comparison ({args.profile}) ──")
    print(f"    sMAPE:          P3.3={p33.get('smape_floor50')}  P2={phase2.get('smape_floor50')}  "
          f"{'✅target' if meets['smape_floor50'] else '❌'} {'↓' if comparison['delta_vs_phase2'].get('smape_floor50', 0) < 0 else '↑'}")
    print(f"    severe:         P3.3={p33.get('severe_underestimate')}  P2={phase2.get('severe_underestimate')}  "
          f"{'✅target' if meets['severe_underestimate'] else '❌'} {'↓' if comparison['delta_vs_phase2'].get('severe_underestimate', 0) < 0 else '↑'}")
    print(f"    false_lift:     P3.3={p33.get('false_lift_rate')}  P2={phase2.get('false_lift_rate')}  "
          f"{'✅target' if meets['false_lift_rate'] else '❌'} {'↓' if comparison['delta_vs_phase2'].get('false_lift_rate', 0) < 0 else '↑'}")

    # Assessment
    if all_met and beats:
        assessment = "GO"
        reason = "All targets met AND beats Phase 2 severe count."
    elif all_met:
        assessment = "GO"
        reason = "All targets met."
    elif beats:
        assessment = "CONDITIONAL"
        reason = "Beats Phase 2 severe count but not all targets met."
    else:
        assessment = "NO-GO"
        reason = "Does not meet targets or beat Phase 2."
    print(f"\n  Assessment: {assessment} — {reason}")

    # ── 5. Generate report ───────────────────────────────────────────
    report = generate_report(metrics, comparison, args.profile, args)

    report_doc = _PROJECT_ROOT / "docs/reports/P33_spike_gated_uplift_report.md"
    report_doc.parent.mkdir(parents=True, exist_ok=True)
    report_doc.write_text(report, encoding="utf-8")
    print(f"\n  [OK] Report: {report_doc}")

    # Also save a copy locally
    local_report = out_dir / "P33_spike_gated_uplift_report.md"
    local_report.write_text(report, encoding="utf-8")
    print(f"  [OK] Report (local): {local_report}")

    print("\n" + "=" * 60)
    print(f"  Assessment: {assessment}")
    print("=" * 60)


if __name__ == "__main__":
    main()
