#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p4_final_fusion_correction.py — P4 Final Fusion + Correction.

Combines Window 2 (best LightGBM candidate) and Window 3 (best hybrid gate)
with Phase2 fusion + correction pipeline. Evaluates 4 combinations against
DEPLOY GO (sMAPE <= 20.50, severe <= 63, false_lift <= 10%).

Pipeline:
    1. Load Phase2 canonical prediction pack (always available)
    2. Optionally merge Window 2 LightGBM predictions (replacing lightgbm y_pred)
    3. Optionally merge Window 3 hybrid gate predictions (replacing base_fused_pred)
    4. Build 4 combination base predictions:
       a) Phase2 champion baseline
       b) W2 best LGBM + Phase2 fusion
       c) Phase2 base + W3 hybrid gate
       d) W2 best LGBM + W3 hybrid gate
    5. Run Phase2 medium correction (normal mode only) on each
    6. Compute metrics + GO/NO-GO assessment

Usage:
    # All inputs available
    python scripts/evaluate_p4_final_fusion_correction.py \\
        --canonical-pack <path> \\
        --window2-csv <path> \\
        --window3-csv <path> \\
        --risk-predictions <path> \\
        --out-dir reports/local/p4_final_fusion_correction

    # Phase2 baseline only (W2/W3 not yet ready)
    python scripts/evaluate_p4_final_fusion_correction.py \\
        --canonical-pack <path> \\
        --risk-predictions <path>

Output:
    {combo}/
        predictions.csv
        metrics.json
    comparison_table.md
    comparison_summary.json
    docs/reports/P4_final_fusion_correction_report.md (auto-generated)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
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
    run_correction,
    write_correction_manifest,
)
from scripts.evaluate_realtime_spike_correction import (
    compute_all_metrics,
    compute_smape,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ── GO Thresholds ──────────────────────────────────────────────────────

PHASE2_CHAMPION = {"smape": 20.86, "severe": 63}

DEPLOY_GO = {
    "smape": 20.50,
    "severe": 63,
    "false_lift_rate": 0.10,
    "normal_hours_degradation": 0.50,
}


# ── Combination definitions ────────────────────────────────────────────

COMBOS = {
    "phase2_baseline": {
        "label": "Phase2 Champion Baseline",
        "use_w2": False,
        "use_w3": False,
    },
    "w2_only": {
        "label": "W2 Best LGBM + Phase2 Fusion",
        "use_w2": True,
        "use_w3": False,
    },
    "w3_only": {
        "label": "Phase2 Base + W3 Hybrid Gate",
        "use_w2": False,
        "use_w3": True,
    },
    "w2_plus_w3": {
        "label": "W2 Best LGBM + W3 Hybrid Gate",
        "use_w2": True,
        "use_w3": True,
    },
}


# ── Prediction pack builder ────────────────────────────────────────────

def build_single_timestamp_pack(
    base_fused_pred: pd.Series,
    y_true: pd.Series,
    keys: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> Path:
    """Build a 1-row-per-timestamp prediction pack from base_fused values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = keys[["business_day", "hour_business"]].copy()
    pack["base_fused_pred"] = base_fused_pred.values
    pack["y_true"] = y_true.values
    out_path = out_dir / f"prediction_pack_{label}.csv"
    pack.to_csv(out_path, index=False)
    print(f"  [INFO] Prediction pack: {out_path} ({len(pack)} rows)")
    return out_path


def build_phase2_pack(
    canonical_pack: pd.DataFrame,
    out_dir: Path,
) -> Path:
    """Extract base_fused_pred from canonical Phase2 pack (1 row per timestamp).

    The canonical pack has 4 rows per timestamp (one per model), each with
    the same base_fused_pred. We deduplicate on (business_day, hour_business).
    """
    pack = canonical_pack[["business_day", "hour_business", "base_fused_pred", "y_true"]]
    pack = pack.drop_duplicates(subset=["business_day", "hour_business"]).copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase2_baseline_pack.csv"
    pack.to_csv(out_path, index=False)
    print(f"  [INFO] Phase2 baseline pack: {out_path} ({len(pack)} rows)")
    return out_path


# ── Build base_fused_pred per combo ────────────────────────────────────

def build_combo_base_pred(
    combo_key: str,
    canonical_pack: pd.DataFrame,
    w2_df: Optional[pd.DataFrame],
    w3_df: Optional[pd.DataFrame],
    keys: pd.DataFrame,
) -> pd.Series:
    """Build base_fused_pred for a given combo.

    Args:
        combo_key: One of phase2_baseline, w2_only, w3_only, w2_plus_w3.
        canonical_pack: Full Phase2 canonical pack (4 rows per timestamp).
        w2_df: Window 2 DataFrame with lightgbm predictions.
        w3_df: Window 3 DataFrame with hybrid gate predictions.
        keys: Unique (business_day, hour_business) key DataFrame.

    Returns:
        pd.Series aligned to keys with base_fused_pred values.
    """
    combo = COMBOS[combo_key]

    if not combo["use_w2"] and not combo["use_w3"]:
        # Phase2 champion baseline: use existing base_fused_pred
        deduped = canonical_pack[["business_day", "hour_business", "base_fused_pred"]].drop_duplicates(
            subset=["business_day", "hour_business"]
        )
        merged = keys.merge(deduped, on=["business_day", "hour_business"], how="left")
        return merged["base_fused_pred"]

    if combo["use_w2"] and not combo["use_w3"]:
        # W2 best LGBM + Phase2 fusion: replace lightgbm y_pred with W2,
        # then recompute anchor fusion (same weights as Phase2).
        return _compute_w2_fusion_pred(canonical_pack, w2_df, keys)

    if not combo["use_w2"] and combo["use_w3"]:
        # Phase2 base + W3 hybrid gate: use W3 predictions as base_fused_pred
        return _merge_w3_pred(w3_df, keys)

    # w2_plus_w3: W2 replaces lightgbm, W3 replaces base_fused_pred
    return _merge_w3_pred(w3_df, keys)


def _compute_w2_fusion_pred(
    canonical_pack: pd.DataFrame,
    w2_df: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.Series:
    """Recompute Phase2 anchor fusion with W2 replacing lightgbm's y_pred.

    Phase2 anchor_90 weights: 0.9 lightgbm + 0.03333 each of 3 baselines.
    """
    # Get baseline models from canonical pack (non-lightgbm)
    baselines = canonical_pack[canonical_pack["model_name"] != "lightgbm"].copy()
    baseline_pred = baselines.groupby(["business_day", "hour_business"])["y_pred"].mean()
    baseline_pred = baseline_pred.reset_index()

    # Get W2 predictions
    w2_renamed = w2_df[["business_day", "hour_business", "y_pred"]].copy()
    w2_renamed = w2_renamed.rename(columns={"y_pred": "w2_pred"})

    # Merge all
    merged = keys.merge(baseline_pred, on=["business_day", "hour_business"], how="left")
    merged = merged.merge(w2_renamed, on=["business_day", "hour_business"], how="left")

    # Phase2 anchor_90: 0.9 * W2 + (0.1/3) * each baseline ≈ 0.9 * W2 + 0.1 * baseline_mean
    # Since baseline_pred is already the mean of 3 baselines,
    # base_fused = 0.9 * w2_pred + 0.1 * baseline_mean
    merged["base_fused_pred"] = 0.9 * merged["w2_pred"] + 0.1 * merged["y_pred"]
    merged["base_fused_pred"] = merged["base_fused_pred"].fillna(
        merged["w2_pred"]  # fallback if baselines missing
    )
    return merged["base_fused_pred"]


def _merge_w3_pred(
    w3_df: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.Series:
    """Use W3 hybrid gate predictions as base_fused_pred."""
    w3 = w3_df[["business_day", "hour_business", "y_pred"]].copy()
    w3 = w3.rename(columns={"y_pred": "base_fused_pred"})
    merged = keys.merge(w3, on=["business_day", "hour_business"], how="left")
    if merged["base_fused_pred"].isna().any():
        print(f"  [WARN] W3 missing {merged['base_fused_pred'].isna().sum()} timestamps")
    return merged["base_fused_pred"]


# ── Run single combo ───────────────────────────────────────────────────

def run_combo(
    combo_key: str,
    base_fused: pd.Series,
    y_true: pd.Series,
    keys: pd.DataFrame,
    risk_predictions_path: Path,
    profile: CorrectionProfile,
    out_dir: Path,
) -> dict[str, Any]:
    """Run correction + evaluation for a single combo. Returns metrics dict."""
    combo_label = COMBOS[combo_key]["label"]
    config_out = out_dir / combo_key
    config_out.mkdir(parents=True, exist_ok=True)

    pack_path = build_single_timestamp_pack(
        base_fused, y_true, keys, config_out, combo_key
    )

    result = run_correction(
        prediction_pack_path=str(pack_path),
        risk_predictions_path=str(risk_predictions_path),
        profile=profile,
    )

    result_csv = config_out / "predictions.csv"
    result.to_csv(result_csv, index=False)

    # Deduplicate if needed
    if "hour_business" in result.columns:
        n_before = len(result)
        result = result.drop_duplicates(subset=["business_day", "hour_business"]).copy()
        if len(result) < n_before:
            print(f"  [INFO] Dedup: {n_before} -> {len(result)} rows")

    # Metrics
    metrics = compute_all_metrics(result)
    metrics["combo"] = combo_key
    metrics["combo_label"] = combo_label
    metrics["n_timestamps"] = len(result)

    metrics_path = config_out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  sMAPE:           {metrics.get('realtime_overall_smape_floor50', 'N/A')}")
    print(f"  Base sMAPE:      {metrics.get('realtime_base_smape_floor50', 'N/A')}")
    print(f"  Severe:          {metrics.get('severe_underestimate_count', 'N/A')}")
    print(f"  High-spike MAE:  {metrics.get('high_spike_mae', 'N/A')}")
    print(f"  False lift:      {metrics.get('false_lift_rate', 'N/A')}")
    print(f"  Normal degrad:   {metrics.get('normal_hours_degradation', 'N/A')}")

    return metrics


# ── GO assessment ──────────────────────────────────────────────────────

def assess_go(metrics: dict[str, Any]) -> dict[str, Any]:
    """Assess DEPLOY GO / PAPER GO / NO-GO for a single combo."""
    deploy_criteria = {
        "sMAPE <= 20.50": {
            "threshold": DEPLOY_GO["smape"],
            "actual": metrics.get("realtime_overall_smape_floor50"),
            "met": False,
        },
        "Severe <= 63": {
            "threshold": DEPLOY_GO["severe"],
            "actual": metrics.get("severe_underestimate_count"),
            "met": False,
        },
        "False lift <= 10%": {
            "threshold": DEPLOY_GO["false_lift_rate"],
            "actual": metrics.get("false_lift_rate"),
            "met": False,
        },
        "Normal degradation <= 0.5": {
            "threshold": DEPLOY_GO["normal_hours_degradation"],
            "actual": metrics.get("normal_hours_degradation"),
            "met": False,
        },
    }

    for _, c in deploy_criteria.items():
        if c["actual"] is not None:
            c["met"] = c["actual"] <= c["threshold"]

    all_deploy_met = all(c["met"] for c in deploy_criteria.values() if c["actual"] is not None)

    # PAPER GO: clear improvement over Phase2 champion
    smape_improvement = (
        PHASE2_CHAMPION["smape"] - metrics.get("realtime_overall_smape_floor50", 999)
        if metrics.get("realtime_overall_smape_floor50") is not None
        else -999
    )
    severe_improvement = (
        PHASE2_CHAMPION["severe"] - metrics.get("severe_underestimate_count", 999)
        if metrics.get("severe_underestimate_count") is not None
        else -999
    )
    paper_go = smape_improvement > 0 and severe_improvement >= 0

    if all_deploy_met:
        verdict = "DEPLOY GO"
    elif paper_go:
        verdict = "PAPER GO"
    else:
        verdict = "NO-GO"

    return {
        "verdict": verdict,
        "all_deploy_criteria_met": all_deploy_met,
        "paper_go": paper_go,
        "deploy_criteria": deploy_criteria,
        "smape_improvement_vs_phase2": round(smape_improvement, 2),
        "severe_improvement_vs_phase2": severe_improvement,
    }


# ── Comparison table ────────────────────────────────────────────────────

def build_comparison_table(all_metrics: dict[str, dict]) -> str:
    """Build markdown comparison table."""
    rows = []
    header = "| Combo | Label | sMAPE | Base sMAPE | Severe | High-spike MAE | False Lift | Normal Degrad | Verdict |"
    sep = "|------|-------|:-----:|:----------:|:------:|:--------------:|:----------:|:-------------:|:-------:|"
    rows.append(header)
    rows.append(sep)

    for combo_key in COMBOS:
        if combo_key not in all_metrics:
            continue
        m = all_metrics[combo_key]
        label = COMBOS[combo_key]["label"]
        smape = _fmt(m.get("realtime_overall_smape_floor50"))
        base_smape = _fmt(m.get("realtime_base_smape_floor50"))
        severe = m.get("severe_underestimate_count", "—")
        hspike = _fmt(m.get("high_spike_mae"))
        flift = _fmt(m.get("false_lift_rate"))
        ndeg = _fmt(m.get("normal_hours_degradation"))
        verdict = m.get("_verdict", {}).get("verdict", "—")
        rows.append(f"| {combo_key} | {label} | {smape} | {base_smape} | {severe} | {hspike} | {flift} | {ndeg} | {verdict} |")

    # Phase2 champion reference row
    rows.append(f"| phase2_champion | Phase2 Champion (ref) | {PHASE2_CHAMPION['smape']} | — | {PHASE2_CHAMPION['severe']} | — | — | — | — |")

    return "\n".join(rows)


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v) if v is not None else "—"


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P4 Final Fusion + Correction evaluation.",
    )
    parser.add_argument("--canonical-pack", required=True,
                        help="Path to Phase2 canonical prediction pack CSV")
    parser.add_argument("--window2-csv", default=None,
                        help="Path to Window 2 best LightGBM candidate CSV")
    parser.add_argument("--window3-csv", default=None,
                        help="Path to Window 3 best hybrid gate CSV")
    parser.add_argument("--risk-predictions", required=True,
                        help="Path to risk predictions CSV (with high_spike_prob)")
    parser.add_argument("--out-dir",
                        default="reports/local/p4_final_fusion_correction")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    canonical_path = Path(args.canonical_pack)
    w2_path = Path(args.window2_csv) if args.window2_csv else None
    w3_path = Path(args.window3_csv) if args.window3_csv else None
    rp_path = Path(args.risk_predictions)

    for p in [canonical_path, rp_path]:
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    # Determine which combos are feasible
    available_combos = ["phase2_baseline"]  # always available
    if w2_path and w2_path.exists():
        available_combos.append("w2_only")
        if w3_path and w3_path.exists():
            available_combos.append("w2_plus_w3")
    if w3_path and w3_path.exists():
        available_combos.append("w3_only")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  P4 Final Fusion + Correction")
    print("=" * 60)

    canonical = pd.read_csv(canonical_path)
    w2 = pd.read_csv(w2_path) if w2_path and w2_path.exists() else None
    w3 = pd.read_csv(w3_path) if w3_path and w3_path.exists() else None

    # Unique timestamps from canonical pack
    keys = canonical[["business_day", "hour_business"]].drop_duplicates().reset_index(drop=True)
    y_true = canonical[["business_day", "hour_business", "y_true"]].drop_duplicates(
        subset=["business_day", "hour_business"]
    )["y_true"]

    print(f"  Canonical timestamps: {len(keys)}")
    print(f"  W2 available: {w2 is not None}")
    print(f"  W3 available: {w3 is not None}")
    print(f"  Combos to evaluate: {available_combos}")
    print(f"  Correction: medium (normal mode only)")

    # ── Correction profile (medium, normal mode) ─────────────────────
    profile = CorrectionProfile(
        name="medium",
        spike_prob_threshold=0.60,
        max_lift_ratio=0.35,
        max_absolute_lift=350.0,
        protect_normal_hours=True,
        period_9_16_boost=1.15,
    )

    # ── Run each combo ───────────────────────────────────────────────
    all_metrics: dict[str, dict[str, Any]] = {}
    t_start = time.time()

    for combo_key in available_combos:
        print(f"\n  {'─' * 50}")
        print(f"  Combo: {combo_key} — {COMBOS[combo_key]['label']}")
        print(f"  {'─' * 50}")

        base_pred = build_combo_base_pred(combo_key, canonical, w2, w3, keys)

        metrics = run_combo(
            combo_key=combo_key,
            base_fused=base_pred,
            y_true=y_true,
            keys=keys,
            risk_predictions_path=rp_path,
            profile=profile,
            out_dir=out_dir,
        )

        verdict = assess_go(metrics)
        metrics["_verdict"] = verdict
        all_metrics[combo_key] = metrics

        print(f"  >> VERDICT: {verdict['verdict']}")

    total_time = time.time() - t_start

    # ── Print comparison ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}")

    table = build_comparison_table(all_metrics)
    print(f"\n{table}")

    table_path = out_dir / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# P4 Final Fusion + Correction — Comparison\n\n")
        f.write(f"DEPLOY GO: sMAPE ≤ {DEPLOY_GO['smape']}, severe ≤ {DEPLOY_GO['severe']}, "
                f"false_lift ≤ {DEPLOY_GO['false_lift_rate']}, normal_degradation ≤ {DEPLOY_GO['normal_hours_degradation']}\n\n")
        f.write(table)
        f.write("\n")

    # ── Summary JSON ─────────────────────────────────────────────────
    summary = {
        "script": "scripts/evaluate_p4_final_fusion_correction.py",
        "canonical_pack": str(canonical_path),
        "window2_csv": str(w2_path) if w2_path else None,
        "window3_csv": str(w3_path) if w3_path else None,
        "profile": {"name": "medium", "mode": "normal"},
        "deploy_go_thresholds": DEPLOY_GO,
        "phase2_champion": PHASE2_CHAMPION,
        "combos": {
            k: {
                "metrics": all_metrics[k],
                "verdict": all_metrics[k].get("_verdict"),
            }
            for k in all_metrics
        },
        "total_runtime_seconds": round(total_time, 1),
    }
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  Summary: {summary_path}")
    print(f"  Total runtime: {total_time:.0f}s")
    print("\nDone.")


if __name__ == "__main__":
    main()
