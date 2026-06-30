#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_phase2_anchored_results.py — Aggregate Phase 2 results.

Reads all correction metrics from reports/local/p0_phase2_anchored/,
recomputes timestamp-level (deduplicated) metrics, and produces
a unified ranking table.

Usage:
    python scripts/evaluate_phase2_anchored_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

PHASE2_ROOT = _PROJECT_ROOT / "reports/local/p0_phase2_anchored"
PACKS_DIR = PHASE2_ROOT / "packs"
CORRECTION_DIR = PHASE2_ROOT / "correction"
SUMMARY_DIR = PHASE2_ROOT / "summary"

FUSION_MODES = ["mean", "lightgbm_anchor_90", "lightgbm_anchor_80", "candidate_reference_only"]
CORRECTION_MODES = ["normal", "relaxed"]
PROFILES = ["conservative", "medium", "aggressive"]

BASELINE_LGBM = {"smape": 22.02, "severe": 80}
BASELINE_MEAN = {"smape": 24.46, "severe": 150}


def compute_smape_floor50(
    y_true: pd.Series, y_pred: pd.Series
) -> np.ndarray:
    yt = np.maximum(np.abs(y_true.values), 50.0)
    yp = np.maximum(np.abs(y_pred.values), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
    return np.minimum(smape, 50.0)


def compute_9_16_smape(ts_df: pd.DataFrame) -> float:
    mask = ts_df["period"] == "9_16"
    if mask.sum() == 0:
        return float("nan")
    smapes = compute_smape_floor50(
        ts_df.loc[mask, "y_true"], ts_df.loc[mask, "final_pred"]
    )
    return float(np.nanmean(smapes))


def compute_metrics(ts_df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Compute timestamp-level metrics from deduplicated DataFrame."""
    smape = float(np.nanmean(compute_smape_floor50(ts_df["y_true"], ts_df["final_pred"])))
    base_smape = float(np.nanmean(compute_smape_floor50(ts_df["y_true"], ts_df["base_fused_pred"])))

    severe = int((ts_df["y_true"] - ts_df["final_pred"] > 200).sum())
    severe_base = int((ts_df["y_true"] - ts_df["base_fused_pred"] > 200).sum())

    smape_9_16 = compute_9_16_smape(ts_df)
    try:
        mask_9 = ts_df["period"] == "9_16"
        base_smape_9_16 = float(np.nanmean(compute_smape_floor50(
            ts_df.loc[mask_9, "y_true"], ts_df.loc[mask_9, "base_fused_pred"]
        )))
    except Exception:
        base_smape_9_16 = float("nan")

    # High spike MAE
    spike_mask = ts_df["high_spike_flag"] == 1
    if spike_mask.sum() > 0:
        spike_mae = float(ts_df.loc[spike_mask, "abs_error"].mean())
        spike_smape = float(np.nanmean(compute_smape_floor50(
            ts_df.loc[spike_mask, "y_true"], ts_df.loc[spike_mask, "final_pred"]
        )))
    else:
        spike_mae = float("nan")
        spike_smape = float("nan")

    # Normal hours
    normal_mask = ts_df["high_spike_flag"] == 0
    if normal_mask.sum() > 0:
        normal_before = float(np.nanmean(compute_smape_floor50(
            ts_df.loc[normal_mask, "y_true"], ts_df.loc[normal_mask, "base_fused_pred"]
        )))
        normal_after = float(np.nanmean(compute_smape_floor50(
            ts_df.loc[normal_mask, "y_true"], ts_df.loc[normal_mask, "final_pred"]
        )))
        normal_degradation = round(normal_after - normal_before, 4)
    else:
        normal_before = float("nan")
        normal_after = float("nan")
        normal_degradation = float("nan")

    # False lift rate (non-spike timestamps where final_pred > base_fused_pred)
    false_lift_mask = (normal_mask) & (ts_df["final_pred"] > ts_df["base_fused_pred"])
    false_lift_rate = false_lift_mask.sum() / max(normal_mask.sum(), 1)

    # Lift applied count
    lift_applied = int((ts_df["final_pred"] != ts_df["base_fused_pred"]).sum())

    return {
        "label": label,
        "n_timestamps": len(ts_df),
        "smape": round(smape, 4),
        "base_smape": round(base_smape, 4),
        "smape_9_16": round(smape_9_16, 4),
        "base_smape_9_16": round(base_smape_9_16, 4),
        "severe_underestimate": severe,
        "severe_underestimate_base": severe_base,
        "severe_delta": severe - severe_base,
        "high_spike_mae": round(spike_mae, 4),
        "high_spike_smape": round(spike_smape, 4),
        "normal_hours_before": round(normal_before, 4),
        "normal_hours_after": round(normal_after, 4),
        "normal_hours_degradation": normal_degradation,
        "false_lift_rate": round(false_lift_rate, 4),
        "lift_applied_count": lift_applied,
    }


def go_nogo(row: dict, is_relaxed: bool) -> str:
    """Determine GO / CONDITIONAL / NO-GO."""
    smape = row["smape"]
    severe = row["severe_underestimate"]
    false_lift = row["false_lift_rate"]
    degradation = row["normal_hours_degradation"]

    if is_relaxed:
        # Relaxed mode can never be GO
        if (
            smape <= 22.50
            and severe < 80
            and false_lift <= 0.20
            and degradation <= 1.0
        ):
            return "CONDITIONAL (relaxed)"
        return "NO-GO"

    if (
        smape <= 22.02
        and severe < 80
        and false_lift <= 0.15
        and degradation <= 0.5
    ):
        return "GO"

    if (
        smape <= 22.50
        and severe < 80
        and false_lift <= 0.20
        and degradation <= 1.0
    ):
        return "CONDITIONAL"

    return "NO-GO"


def load_pack(pack_dir: Path) -> pd.DataFrame | None:
    """Load the prediction pack CSV from a pack directory."""
    csvs = list(pack_dir.glob("prediction_pack_realtime_multicandidate_*.csv"))
    if not csvs:
        return None
    return pd.read_csv(csvs[0])


def load_correction_result(pack: str, mode: str, profile: str) -> pd.DataFrame | None:
    """Load correction_result.csv for a specific eval."""
    path = CORRECTION_DIR / pack / mode / profile / "correction_result.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def load_build_manifest(pack_dir: Path) -> dict:
    manifest_path = pack_dir / "build_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    return {}


def main():
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    print("=" * 70)
    print("  Phase 2 Anchored Fusion — Timestamp-Level Metrics Aggregation")
    print("=" * 70)

    for fusion_mode in FUSION_MODES:
        pack_dir = PACKS_DIR / fusion_mode
        manifest = load_build_manifest(pack_dir)
        fusion_base_smape = manifest.get("timestamp_level_base_smape", "?")
        fusion_base_severe = manifest.get("timestamp_level_severe_underestimate", "?")

        print(f"\n  [{fusion_mode}] pre-correction: sMAPE={fusion_base_smape}, "
              f"severe={fusion_base_severe}")

        for corr_mode in CORRECTION_MODES:
            for profile in PROFILES:
                label = f"{fusion_mode}/{corr_mode}/{profile}"
                result_df = load_correction_result(fusion_mode, corr_mode, profile)
                if result_df is None or result_df.empty:
                    print(f"    {label}: NO DATA")
                    continue

                # Deduplicate to timestamp level (1 row per business_day + hour_business)
                ts_df = result_df.drop_duplicates(
                    subset=["business_day", "hour_business"]
                ).copy()

                # Ensure metric columns exist
                if "abs_error" not in ts_df.columns:
                    ts_df["abs_error"] = (ts_df["y_true"] - ts_df["final_pred"]).abs()
                if "high_spike_flag" not in ts_df.columns:
                    ts_df["high_spike_flag"] = (
                        (ts_df["y_true"] - ts_df["base_fused_pred"]).abs() > 200
                    ).astype(int)
                if "period" not in ts_df.columns:
                    ts_df["period"] = "?"

                metrics = compute_metrics(ts_df, label)
                verdict = go_nogo(metrics, is_relaxed=(corr_mode == "relaxed"))
                metrics["verdict"] = verdict
                all_results.append(metrics)

                print(f"    {label}: sMAPE={metrics['smape']}, "
                      f"severe={metrics['severe_underestimate']}, "
                      f"lift={metrics['lift_applied_count']}, "
                      f"false_lift={metrics['false_lift_rate']:.1%}, "
                      f"degradation={metrics['normal_hours_degradation']:+.2f} | "
                      f"{verdict}")

    # Build DataFrame for CSV output
    rows_out = []
    for r in all_results:
        rows_out.append({
            "fusion_mode": r["label"].split("/")[0],
            "correction_mode": r["label"].split("/")[1],
            "profile": r["label"].split("/")[2],
            "n_timestamps": r["n_timestamps"],
            "smape_floor50": r["smape"],
            "base_smape_floor50": r["base_smape"],
            "smape_9_16": r["smape_9_16"],
            "severe_underestimate": r["severe_underestimate"],
            "severe_underestimate_base": r["severe_underestimate_base"],
            "severe_delta": r["severe_delta"],
            "high_spike_mae": r["high_spike_mae"],
            "high_spike_smape": r["high_spike_smape"],
            "normal_hours_degradation": r["normal_hours_degradation"],
            "false_lift_rate": r["false_lift_rate"],
            "lift_applied_count": r["lift_applied_count"],
            "verdict": r["verdict"],
        })

    # CSV output
    csv_path = SUMMARY_DIR / "phase2_anchored_metrics.csv"
    df_out = pd.DataFrame(rows_out)
    df_out.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n[OK] CSV: {csv_path}")

    # Markdown summary
    lines = [
        "# Phase 2 Anchored Fusion — Correction Results",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Baselines",
        "",
        f"| Baseline | sMAPE (floor50) | Severe Underestimates |",
        f"|----------|-----------------|----------------------|",
        f"| LightGBM-only | {BASELINE_LGBM['smape']} | {BASELINE_LGBM['severe']} |",
        f"| Mean multi-candidate | {BASELINE_MEAN['smape']} | {BASELINE_MEAN['severe']} |",
        "",
        "## All Results (timestamp-level)",
        "",
        "| # | Fusion | Mode | Profile | sMAPE | base_sMAPE | 9_16 | Severe | ΔSevere | Spike MAE | False Lift | Degrad | Lift | Verdict |",
        "|---|--------|------|---------|-------|-----------|------|--------|---------|-----------|------------|--------|------|---------|",
    ]

    # Sort: normal mode first then relaxed, then by sMAPE ascending
    def sort_key(r):
        mode_order = 0 if r["label"].split("/")[1] == "normal" else 1
        return (mode_order, r["smape"])

    all_results.sort(key=sort_key)

    for i, r in enumerate(all_results, 1):
        parts = r["label"].split("/")
        lines.append(
            f"| {i} | {parts[0]} | {parts[1]} | {parts[2]} "
            f"| {r['smape']} | {r['base_smape']} | {r['smape_9_16']} "
            f"| {r['severe_underestimate']} | {r['severe_delta']:+d} "
            f"| {r['high_spike_mae']} | {r['false_lift_rate']:.1%} "
            f"| {r['normal_hours_degradation']:+.2f} "
            f"| {r['lift_applied_count']} | {r['verdict']} |"
        )

    # GO / CONDITIONAL / NO-GO summary
    lines += [
        "",
        "## GO / CONDITIONAL / NO-GO Summary",
        "",
    ]
    verdicts = {}
    for r in all_results:
        v = r["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    for v, count in sorted(verdicts.items()):
        lines.append(f"- **{v}**: {count} configurations")

    # Best normal-mode candidate
    normal_results = [r for r in all_results if r["label"].split("/")[1] == "normal"]
    if normal_results:
        best_normal = min(normal_results, key=lambda r: (
            r["severe_underestimate"], r["smape"]
        ))
        lines += [
            "",
            "## Best Normal-Mode Candidate",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Fusion | {best_normal['label']} |",
            f"| sMAPE | {best_normal['smape']} |",
            f"| Severe | {best_normal['severe_underestimate']} |",
            f"| False Lift | {best_normal['false_lift_rate']:.1%} |",
            f"| Degradation | {best_normal['normal_hours_degradation']:+.2f} |",
            f"| Lift Applied | {best_normal['lift_applied_count']} |",
            f"| Verdict | {best_normal['verdict']} |",
        ]

    # Best relaxed-mode candidate
    relaxed_results = [r for r in all_results if r["label"].split("/")[1] == "relaxed"]
    if relaxed_results:
        best_relaxed = min(relaxed_results, key=lambda r: (
            r["severe_underestimate"], r["smape"]
        ))
        lines += [
            "",
            "## Best Relaxed-Mode Candidate (offline diagnostic only)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Fusion | {best_relaxed['label']} |",
            f"| sMAPE | {best_relaxed['smape']} |",
            f"| Severe | {best_relaxed['severe_underestimate']} |",
            f"| False Lift | {best_relaxed['false_lift_rate']:.1%} |",
            f"| Degradation | {best_relaxed['normal_hours_degradation']:+.2f} |",
            f"| Lift Applied | {best_relaxed['lift_applied_count']} |",
            f"| Verdict | {best_relaxed['verdict']} |",
        ]

    # Recommendations
    lines += [
        "",
        "## Recommendations",
        "",
    ]

    # Check if any normal mode achieved GO
    go_normal = [r for r in normal_results if r["verdict"] == "GO"]
    conditional_normal = [r for r in normal_results if r["verdict"] == "CONDITIONAL"]

    if go_normal:
        lines.append("**GO achieved in normal mode** — candidate ready for further tuning.")
    elif conditional_normal:
        lines.append("**CONDITIONAL in normal mode** — best candidate needs improvement.")
    else:
        lines.append("**NO-GO in normal mode** — correction not viable for production.")

    lines += [
        "",
        "### Next Steps",
        "",
    ]

    # Find top blocker
    if normal_results:
        best = min(normal_results, key=lambda r: r["severe_underestimate"])
        lines.append(f"1. Best normal candidate: {best['label']} "
                     f"(sMAPE {best['smape']}, severe {best['severe_underestimate']})")
        if best["severe_underestimate"] >= 80:
            lines.append("2. Severe underestimates still >= 80 — correction not reducing spike errors enough")
        if best["false_lift_rate"] > 0.15:
            lines.append(f"3. False lift rate {best['false_lift_rate']:.1%} exceeds 15% threshold")
        lines.append("4. RT916 selective inference recommended for top spike days")
        lines.append("5. Multi-model real predictions (TimesFM/SGDFNet) needed for P3")

    md_path = SUMMARY_DIR / "phase2_anchored_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[OK] MD: {md_path}")

    print("\n" + "=" * 70)
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
