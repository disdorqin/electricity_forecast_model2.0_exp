#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search_p33_severe_constrained_correction.py — P3.3 Grid search over correction params.

Constrains: correction_mode=normal only.
Eliminates: false_lift > 12%, normal_hours_degradation > 0.5.

Output:
    - grid_results.csv              — all combos with computed metrics
    - best_candidates.json          — best per profile family + overall
    - top_candidates_converged.csv  — combos meeting both GO constraints
    - grid_manifest.json            — run config

Usage:
    python scripts/search_p33_severe_constrained_correction.py
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
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
)
from extreme.realtime_high_spike.residual_lift import (
    PERIOD_DEFS,
    ResidualLiftConfig,
    ResidualLiftCorrector,
    get_period,
)
from extreme.realtime_high_spike.guardrail import GuardrailConfig, SpikeGuardrail
from scripts.evaluate_realtime_spike_correction import (
    compute_all_metrics,
    compute_smape,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ────────────────────────────────────────────────────────────────

ROLLING_PREDICTIONS = "reports/local/p31_severe_aware_rolling/severe_softmax/rolling_predictions.csv"
RISK_PREDICTIONS = "reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/risk_predictions_multicandidate.csv"
OUT_DIR = "reports/local/p33_severe_constrained_correction"

# ── Baselines ────────────────────────────────────────────────────────────

PHASE2_BEST = {"sMAPE": 20.86, "severe": 63, "false_lift": 0.0}
P32_MEDIUM = {"sMAPE": 20.74, "severe": 73, "false_lift": 0.0343}  # from P3.2 medium

# ── Constraint thresholds ────────────────────────────────────────────────

DEPLOY_GO = {"sMAPE": 20.50, "severe": 63, "false_lift": 0.10}
RESEARCH_GO = {"sMAPE": 20.00, "severe": 70, "false_lift": 0.12}
HARD_ELIM_FALSE_LIFT = 0.12     # false_lift > 12% → eliminated
HARD_ELIM_NORMAL_DEGRAD = 0.5   # normal_hours_degradation > 0.5 → eliminated

# ── Profile families ─────────────────────────────────────────────────────

# Each profile family is a list of param dicts to iterate.
# Entries with a single item are fixed; lists are searched.

FAMILIES: dict[str, list[dict[str, Any]]] = {
    "medium_plus": [
        {
            "spike_prob_threshold": [0.45, 0.50, 0.55, 0.60],
            "max_lift_ratio": [0.25, 0.35, 0.45],
            "max_absolute_lift": [250, 350, 500],
            "period_9_16_boost": [1.0, 1.15, 1.30, 1.50],
            "protect_normal_hours": [True],
        },
    ],
    "medium_spike_only": [
        {
            "spike_prob_threshold": [0.45, 0.50, 0.55, 0.60],
            "max_lift_ratio": [0.35],
            "max_absolute_lift": [350],
            "period_9_16_boost": [1.15],
            "protect_normal_hours": [True],
        },
    ],
    "medium_916_boost": [
        {
            "spike_prob_threshold": [0.60],
            "max_lift_ratio": [0.35],
            "max_absolute_lift": [350],
            "period_9_16_boost": [1.0, 1.15, 1.30, 1.50],
            "protect_normal_hours": [True],
        },
    ],
    "high_risk_only": [
        {
            "spike_prob_threshold": [0.55, 0.60],
            "max_lift_ratio": [0.35, 0.45],
            "max_absolute_lift": [350, 500],
            "period_9_16_boost": [1.0, 1.15, 1.30],
            "protect_normal_hours": [True],
        },
    ],
    "asymmetric_lift": [
        {
            "spike_prob_threshold": [0.45, 0.50],
            "max_lift_ratio": [0.25, 0.35],
            "max_absolute_lift": [250, 350],
            "period_9_16_boost": [1.0, 1.30],
            "protect_normal_hours": [True],
        },
    ],
}


def build_param_grid(family_config: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build flat list of param combos from a family config."""
    combos: list[dict[str, Any]] = []
    for spec in family_config:
        keys = list(spec.keys())
        values = [spec[k] for k in keys]
        for vals in itertools.product(*values):
            combos.append(dict(zip(keys, vals)))
    return combos


def run_correction_on_df(
    merged: pd.DataFrame,
    params: dict[str, Any],
) -> pd.DataFrame:
    """Run correction pipeline on a pre-merged DataFrame.

    Args:
        merged: DataFrame with business_day, hour_business, base_fused_pred,
                y_true, high_spike_prob columns.
        params: Correction parameters (spike_prob_threshold, max_lift_ratio,
                max_absolute_lift, period_9_16_boost, protect_normal_hours).

    Returns:
        DataFrame with final_pred, lift_applied, reason_code columns added.
    """
    result = merged.copy()

    # Build configs from params
    spike_prob_threshold = params["spike_prob_threshold"]
    max_lift_ratio = params["max_lift_ratio"]
    max_absolute_lift = params["max_absolute_lift"]
    period_9_16_boost = params["period_9_16_boost"]
    protect_normal_hours = params["protect_normal_hours"]

    # Residual lift config
    lift_cfg = ResidualLiftConfig(
        spike_prob_threshold=spike_prob_threshold,
        max_lift_ratio=max_lift_ratio,
        max_absolute_lift=max_absolute_lift,
        protect_normal_hours=protect_normal_hours,
        period_9_16_boost=period_9_16_boost,
        lift_quantile=0.90,
        period_aware=True,
        normal_hour_prob_cap=spike_prob_threshold * 1.1,
        mode=CorrectionMode.NORMAL,
    )

    # Guardrail config
    guard_cfg = GuardrailConfig(
        min_prob_for_lift=spike_prob_threshold,
        protect_normal_hours=protect_normal_hours,
        normal_hour_prob_cap=spike_prob_threshold * 1.1,
        max_lift_ratio_9_16=max_lift_ratio,
        max_absolute_lift_9_16=max_absolute_lift,
        max_lift_ratio_1_8=max_lift_ratio * 0.7,
        max_absolute_lift_1_8=max_absolute_lift * 0.6,
        max_lift_ratio_17_24=max_lift_ratio * 0.7,
        max_absolute_lift_17_24=max_absolute_lift * 0.6,
        mode=CorrectionMode.NORMAL,
    )

    # Default lift candidates (same as run_correction without history)
    corrector = ResidualLiftCorrector(lift_cfg)
    corrector.set_lift_candidates({p: 50.0 for p in PERIOD_DEFS})
    current_boost = corrector._lift_candidates["9_16"]
    corrector._lift_candidates["9_16"] = current_boost * period_9_16_boost

    guardrail = SpikeGuardrail(guard_cfg)

    # Apply row-by-row
    spike_corrected_list: list[float] = []
    final_pred_list: list[float] = []
    reason_code_list: list[str] = []
    lift_applied_list: list[float] = []

    for _, row in result.iterrows():
        base_pred = row.get("base_fused_pred", 0.0)
        spike_prob = row.get("high_spike_prob", 0.0)
        hour_business = row.get("hour_business", 12)

        if pd.isna(base_pred):
            base_pred = 0.0
        if pd.isna(spike_prob):
            spike_prob = 0.0

        # Compute lift
        lift_result = corrector.compute_lift(
            base_pred=float(base_pred),
            spike_prob=float(spike_prob),
            hour_business=int(hour_business),
        )
        corrected = lift_result.corrected_pred

        # Guardrail
        guard_result = guardrail.evaluate(
            base_pred=float(base_pred),
            spike_prob=float(spike_prob),
            corrected_pred=corrected,
            hour_business=int(hour_business),
        )

        spike_corrected_list.append(corrected)
        final_pred_list.append(guard_result.final_pred)
        reason_code_list.append(guard_result.reason_code)
        lift_applied_list.append(guard_result.final_pred - float(base_pred))

    result["spike_corrected_pred"] = spike_corrected_list
    result["final_pred"] = final_pred_list
    result["reason_code"] = reason_code_list
    result["lift_applied"] = lift_applied_list

    return result


def evaluate_combo(
    merged: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Run correction + metrics for one param combo.

    Returns dict with all metrics plus params.
    """
    try:
        result = run_correction_on_df(merged, params)

        # Timestamp-level dedup
        result = result.drop_duplicates(subset=["business_day", "hour_business"]).copy()

        metrics = compute_all_metrics(result)

        # Embed params
        metrics["spike_prob_threshold"] = params["spike_prob_threshold"]
        metrics["max_lift_ratio"] = params["max_lift_ratio"]
        metrics["max_absolute_lift"] = params["max_absolute_lift"]
        metrics["period_9_16_boost"] = params["period_9_16_boost"]
        metrics["protect_normal_hours"] = params["protect_normal_hours"]

        # Denylist check
        metrics["_eliminated_by_false_lift"] = (
            metrics.get("false_lift_rate", 999) > HARD_ELIM_FALSE_LIFT
        )
        metrics["_eliminated_by_normal_degrad"] = (
            metrics.get("normal_hours_degradation", 999) > HARD_ELIM_NORMAL_DEGRAD
        )
        metrics["_eligible"] = not (
            metrics["_eliminated_by_false_lift"]
            or metrics["_eliminated_by_normal_degrad"]
        )

        return metrics

    except Exception as e:
        return {
            "spike_prob_threshold": params.get("spike_prob_threshold"),
            "max_lift_ratio": params.get("max_lift_ratio"),
            "max_absolute_lift": params.get("max_absolute_lift"),
            "period_9_16_boost": params.get("period_9_16_boost"),
            "protect_normal_hours": params.get("protect_normal_hours"),
            "realtime_overall_smape_floor50": None,
            "severe_underestimate_count": None,
            "false_lift_rate": None,
            "normal_hours_degradation": None,
            "_error": str(e),
            "_eligible": False,
        }


def load_and_premerge(
    rolling_path: str,
    risk_path: str,
) -> pd.DataFrame:
    """Load rolling predictions and risk predictions, merge on (business_day, hour_business)."""
    roll = pd.read_csv(rolling_path)
    risk = pd.read_csv(risk_path)

    # Ensure string keys
    roll["business_day"] = roll["business_day"].astype(str)
    risk["business_day"] = risk["business_day"].astype(str)

    merged = pd.merge(
        roll,
        risk[["business_day", "hour_business", "high_spike_prob"]],
        on=["business_day", "hour_business"],
        how="left",
    )
    print(f"  Merged: {len(merged)} rows ({roll.columns.tolist()})")
    print(f"  Risk cols: {risk.columns.tolist()}")
    print(f"  Merged columns: {merged.columns.tolist()}")
    return merged


def score_candidate(
    metrics: dict[str, Any],
    deploy_go: dict[str, float] | None = None,
    research_go: dict[str, float] | None = None,
) -> float:
    """Score a candidate: lower is better.

    Ranks by severe count first, then sMAPE.
    Penalises candidates that miss one constraint badly.
    """
    if deploy_go is None:
        deploy_go = DEPLOY_GO
    if research_go is None:
        research_go = RESEARCH_GO

    severe = metrics.get("severe_underestimate_count", 999) or 999
    smape = metrics.get("realtime_overall_smape_floor50", 999) or 999
    false_lift = metrics.get("false_lift_rate", 0) or 0
    normal_degrad = metrics.get("normal_hours_degradation", 0) or 0

    # Primary: severe count (lower = better)
    # Secondary: sMAPE (lower = better)
    # Tertiary: false_lift (lower = better)
    return severe * 1000 + smape * 10 + false_lift * 100 + normal_degrad * 50


def main():
    print("=" * 60)
    print("  P3.3 Severe-Constrained Correction Grid Search")
    print("=" * 60)

    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────
    print(f"\n  Loading rolling predictions: {ROLLING_PREDICTIONS}")
    print(f"  Loading risk predictions:    {RISK_PREDICTIONS}")
    merged = load_and_premerge(ROLLING_PREDICTIONS, RISK_PREDICTIONS)

    if "base_fused_pred" not in merged.columns:
        sys.exit("ERROR: 'base_fused_pred' not found in merged data")
    if "high_spike_prob" not in merged.columns:
        sys.exit("ERROR: 'high_spike_prob' not found in merged data")
    print(f"  Validated: base_fused_pred + high_spike_prob present")

    # ── Compute base metrics (no correction) ────────────────────────────
    base = merged.copy()
    base["final_pred"] = base["base_fused_pred"]
    base["lift_applied"] = 0.0
    base["reason_code"] = "NO_CORRECTION"
    base_metrics = compute_all_metrics(base)
    print(f"\n  Base (no correction):")
    print(f"    sMAPE:      {base_metrics.get('realtime_overall_smape_floor50', 'N/A')}")
    print(f"    Severe:     {base_metrics.get('severe_underestimate_count', 'N/A')}")
    print(f"    False lift: {base_metrics.get('false_lift_rate', 'N/A')}")

    # ── Grid search ────────────────────────────────────────────────────
    all_rows: list[dict[str, Any]] = []
    best_by_family: dict[str, dict[str, Any]] = {}
    overall_best: dict[str, Any] | None = None

    total_combos = sum(len(build_param_grid(cfg)) for cfg in FAMILIES.values())
    print(f"\n  Total parameter combos: {total_combos}")
    print(f"  {'=' * 50}")

    run_idx = 0
    for family_name, family_config in FAMILIES.items():
        combos = build_param_grid(family_config)
        family_best: dict[str, Any] | None = None
        family_eligible: int = 0
        family_total: int = len(combos)

        print(f"\n  Profile family: {family_name} ({family_total} combos)")
        print(f"  {'─' * 40}")

        for params in combos:
            run_idx += 1

            metrics = evaluate_combo(merged, params)
            metrics["family"] = family_name
            metrics["combo_id"] = run_idx
            all_rows.append(metrics)

            eligible = metrics.get("_eligible", False)
            smape_val = metrics.get("realtime_overall_smape_floor50")
            severe_val = metrics.get("severe_underestimate_count")

            status = "."
            if eligible:
                family_eligible += 1
                # Score and track best in family
                score = score_candidate(metrics)
                metrics["_score"] = score
                if family_best is None or score < family_best.get("_score", 999999):
                    family_best = metrics
                # Track overall best
                if overall_best is None or score < overall_best.get("_score", 999999):
                    overall_best = metrics
                status = "+"

            # Progress indicator
            if run_idx % 20 == 0 or run_idx == total_combos:
                progress = f"  [{run_idx}/{total_combos}] {family_name} combo {run_idx}"
                if smape_val is not None:
                    progress += f" → sMAPE={smape_val:.2f} severe={severe_val} eligible={eligible}{status}"
                print(progress)

        best_by_family[family_name] = family_best
        print(f"  {family_name}: {family_eligible}/{family_total} eligible, "
              f"best severe={family_best.get('severe_underestimate_count') if family_best else 'N/A'} "
              f"sMAPE={family_best.get('realtime_overall_smape_floor50') if family_best else 'N/A'}")

    # ── Build results DataFrame ─────────────────────────────────────────
    result_cols = [
        "combo_id", "family",
        "spike_prob_threshold", "max_lift_ratio", "max_absolute_lift",
        "period_9_16_boost", "protect_normal_hours",
        "realtime_overall_smape_floor50", "realtime_base_smape_floor50",
        "severe_underestimate_count", "severe_underestimate_base_count",
        "false_lift_rate", "normal_hours_degradation",
        "high_spike_mae", "high_spike_base_mae",
        "9_16_smape_floor50", "lift_applied_count",
        "_eligible", "_eliminated_by_false_lift", "_eliminated_by_normal_degrad",
        "_score", "_error",
    ]
    existing_cols = [c for c in result_cols if c in all_rows[0]]
    df_results = pd.DataFrame(all_rows)[existing_cols]

    # Save full grid
    grid_path = out_dir / "grid_results.csv"
    df_results.to_csv(grid_path, index=False)
    print(f"\n  Full grid: {grid_path} ({len(df_results)} rows)")

    # ── Top candidates ─────────────────────────────────────────────────
    eligible = df_results[df_results["_eligible"] == True].copy()
    print(f"\n  Eligible candidates: {len(eligible)} / {len(df_results)}")

    # Eliminated breakdown
    elim_false_lift = df_results[df_results["_eliminated_by_false_lift"] == True]
    elim_normal = df_results[df_results["_eliminated_by_normal_degrad"] == True]
    if len(elim_false_lift) > 0:
        print(f"  Eliminated by false_lift > 12%: {len(elim_false_lift)}")
    if len(elim_normal) > 0:
        print(f"  Eliminated by normal_degrad > 0.5: {len(elim_normal)}")

    if len(eligible) > 0:
        # Sort by severe, then sMAPE, then false_lift
        eligible_sorted = eligible.sort_values(
            by=["severe_underestimate_count", "realtime_overall_smape_floor50", "false_lift_rate"],
            ascending=[True, True, True],
        )

        # Top 10 for deploy GO (severe <= 63, sMAPE <= 20.50)
        deploy = eligible_sorted[
            (eligible_sorted["severe_underestimate_count"] <= DEPLOY_GO["severe"])
            & (eligible_sorted["realtime_overall_smape_floor50"] <= DEPLOY_GO["sMAPE"])
        ].copy()
        print(f"\n  DEPLOY GO (sMAPE<={DEPLOY_GO['sMAPE']}, severe<={DEPLOY_GO['severe']}): {len(deploy)} candidates")

        # Top 10 for research GO (sMAPE <= 20.00, severe <= 70)
        research = eligible_sorted[
            (eligible_sorted["severe_underestimate_count"] <= RESEARCH_GO["severe"])
            & (eligible_sorted["realtime_overall_smape_floor50"] <= RESEARCH_GO["sMAPE"])
        ].copy()
        print(f"  RESEARCH GO (sMAPE<={RESEARCH_GO['sMAPE']}, severe<={RESEARCH_GO['severe']}): {len(research)} candidates")

        # Write top candidates
        top10_path = out_dir / "top_candidates_converged.csv"
        eligible_sorted.head(30).to_csv(top10_path, index=False)
        print(f"  Top 30 candidates: {top10_path}")
    else:
        # No eligible candidates — show nearest misses
        print("\n  No eligible candidates found. Showing nearest misses:")
        df_results["_severe_penalty"] = (
            df_results["severe_underestimate_count"].fillna(999)
            + df_results["realtime_overall_smape_floor50"].fillna(999) * 0.5
        )
        nearest = df_results.sort_values("_severe_penalty").head(20)
        near_path = out_dir / "nearest_misses.csv"
        nearest.to_csv(near_path, index=False)
        print(f"  Nearest misses (20): {near_path}")

    # ── Best candidates JSON ────────────────────────────────────────────
    def make_candidate_entry(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "combo_id": metrics.get("combo_id"),
            "family": metrics.get("family"),
            "params": {
                "spike_prob_threshold": metrics.get("spike_prob_threshold"),
                "max_lift_ratio": metrics.get("max_lift_ratio"),
                "max_absolute_lift": metrics.get("max_absolute_lift"),
                "period_9_16_boost": metrics.get("period_9_16_boost"),
                "protect_normal_hours": metrics.get("protect_normal_hours"),
            },
            "metrics": {
                "smape": metrics.get("realtime_overall_smape_floor50"),
                "base_smape": metrics.get("realtime_base_smape_floor50"),
                "severe": metrics.get("severe_underestimate_count"),
                "base_severe": metrics.get("severe_underestimate_base_count"),
                "false_lift": metrics.get("false_lift_rate"),
                "normal_degradation": metrics.get("normal_hours_degradation"),
                "9_16_smape": metrics.get("9_16_smape_floor50"),
                "high_spike_mae": metrics.get("high_spike_mae"),
                "lift_applied": metrics.get("lift_applied_count"),
            },
        }

    best_candidates: dict[str, Any] = {
        "script": "scripts/search_p33_severe_constrained_correction.py",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baselines": {
            "phase2_best": PHASE2_BEST,
            "p32_medium": P32_MEDIUM,
        },
        "constraints": {
            "correction_mode": "normal",
            "eliminate_false_lift_above": HARD_ELIM_FALSE_LIFT,
            "eliminate_normal_degrad_above": HARD_ELIM_NORMAL_DEGRAD,
            "deploy_go": DEPLOY_GO,
            "research_go": RESEARCH_GO,
        },
        "base_metrics": {
            "smape": base_metrics.get("realtime_overall_smape_floor50"),
            "severe": base_metrics.get("severe_underestimate_count"),
        },
        "total_combos": total_combos,
        "eligible_count": int(eligible.shape[0]) if len(eligible) > 0 else 0,
        "best_by_family": {
            name: make_candidate_entry(metrics) if metrics else None
            for name, metrics in best_by_family.items()
        },
        "overall_best": make_candidate_entry(overall_best) if overall_best else None,
    }

    # Add deploy top 3 if any
    if len(eligible) > 0:
        top_deploy = deploy.head(3) if len(deploy) > 0 else eligible_sorted.head(3)
        best_candidates["top_candidates"] = []
        for _, row in top_deploy.iterrows():
            best_candidates["top_candidates"].append(make_candidate_entry(row.to_dict()))

    best_path = out_dir / "best_candidates.json"
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_candidates, f, indent=2, ensure_ascii=False)
    print(f"\n  Best candidates: {best_path}")

    # ── Manifest ────────────────────────────────────────────────────────
    manifest = {
        "script": "scripts/search_p33_severe_constrained_correction.py",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rolling_predictions": ROLLING_PREDICTIONS,
        "risk_predictions": RISK_PREDICTIONS,
        "profile_families": list(FAMILIES.keys()),
        "total_combos": total_combos,
        "eligible_count": int(eligible.shape[0]) if len(eligible) > 0 else 0,
        "base_metrics": {
            "smape": base_metrics.get("realtime_overall_smape_floor50"),
            "severe": base_metrics.get("severe_underestimate_count"),
        },
    }
    manifest_path = out_dir / "grid_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  Manifest: {manifest_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Grid Search Complete")
    print("=" * 60)
    print(f"  Total combos:   {total_combos}")
    print(f"  Eligible:        {best_candidates['eligible_count']}")
    if overall_best:
        print(f"  Overall best:")
        print(f"    sMAPE:        {overall_best.get('realtime_overall_smape_floor50'):.4f}")
        print(f"    Severe:       {overall_best.get('severe_underestimate_count')}")
        print(f"    False lift:   {overall_best.get('false_lift_rate'):.4f}")
        print(f"    Params:       spike_thr={overall_best.get('spike_prob_threshold')}, "
              f"lift_ratio={overall_best.get('max_lift_ratio')}, "
              f"abs_lift={overall_best.get('max_absolute_lift')}, "
              f"9_16_boost={overall_best.get('period_9_16_boost')}")
    print()


if __name__ == "__main__":
    main()
