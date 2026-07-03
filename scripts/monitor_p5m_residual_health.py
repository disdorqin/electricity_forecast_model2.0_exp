#!/usr/bin/env python3
"""
monitor_p5m_residual_health.py — Monitor negative price module residual health.

Outputs:
    reports/local/p5m_monitor/
        residual_health.json
        residual_health.md

Monitors:
    negative_count, low_valley_count,
    negative_trigger_rate, low_valley_trigger_rate,
    high_spike_overlap_count, downward_correction_count,
    negative_MAE_improvement, low_valley_MAE_improvement,
    overall_sMAPE_improvement, high_spike_MAE_improvement,
    normal_degradation, DATA_LIMITED flag

Delta convention: improvement = before - after, positive = better.
normal_degradation: positive = worse (degradation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme.negative_price.apply_negative_correction import (
    apply_negative_correction,
    compute_metrics,
    get_profile,
    PROFILES,
)
from extreme.negative_price.labels import add_all_labels
from extreme.negative_price.risk_model import compute_heuristic_v2_risk


def monitor_health(
    canonical_pack_path: str | Path,
    out_dir: str | Path,
    profile_name: str = "conservative",
    pred_col: str = "base_fused_pred",
    risk_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run health monitoring on the negative correction module.

    Args:
        canonical_pack_path: Path to canonical evaluation pack CSV.
        out_dir: Output directory.
        profile_name: Correction profile to use.
        pred_col: Column name for predictions.
        risk_path: Optional pre-computed risk CSV (overrides heuristic_v2).

    Returns:
        Health report dict.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(canonical_pack_path)
    df = add_all_labels(df, y_pred_col=pred_col)

    health: dict[str, Any] = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "canonical_pack": str(canonical_pack_path),
        "profile": profile_name,
        "total_rows": len(df),
        "business_days": int(df["business_day"].nunique()) if "business_day" in df.columns else 0,
    }

    # ── Event counts ─────────────────────────────────────────────────
    health["negative_count"] = int((df.get("label_negative_price", 0) == 1).sum())
    health["low_valley_count"] = int((df.get("label_low_valley", 0) == 1).sum())

    # ── DATA_LIMITED check ───────────────────────────────────────────
    health["DATA_LIMITED"] = health["negative_count"] == 0

    # ── Risk scores (from CSV or heuristic_v2) ──────────────────────
    if risk_path is not None:
        risk_df = pd.read_csv(risk_path)
        neg_prob = risk_df.get("negative_prob", pd.Series(0.0))
        lv_prob = risk_df.get("low_valley_prob", pd.Series(0.0))
        health["risk_source"] = "from_csv"
    else:
        heur = compute_heuristic_v2_risk(df, history_df=df, pred_col=pred_col)
        neg_prob = heur.get("negative_prob", pd.Series(0.0))
        lv_prob = heur.get("low_valley_prob", pd.Series(0.0))
        health["risk_source"] = "heuristic_v2"

    for thresh in [0.2, 0.3, 0.4, 0.5]:
        health[f"negative_trigger_rate_{thresh}"] = round(float((neg_prob > thresh).mean()), 4)
        health[f"low_valley_trigger_rate_{thresh}"] = round(float((lv_prob > thresh).mean()), 4)

    health["negative_trigger_rate"] = health.get("negative_trigger_rate_0.3", 0.0)
    health["low_valley_trigger_rate"] = health.get("low_valley_trigger_rate_0.3", 0.0)

    # ── High-spike overlap ───────────────────────────────────────────
    high_spike_cols = ["high_spike_prob", "high_spike_flag", "label_high_spike", "spike_label"]
    spike_col = next((c for c in high_spike_cols if c in df.columns), None)
    if spike_col is not None:
        spike_active = df[spike_col].fillna(0).astype(float) > 0.5
        low_valley_active = (df.get("label_low_valley", 0) == 1)
        health["high_spike_overlap_count"] = int((spike_active & low_valley_active).sum())
    else:
        health["high_spike_overlap_count"] = 0

    # ── Run correction and get metrics ───────────────────────────────
    profile = get_profile(profile_name)
    result_df = apply_negative_correction(
        prediction_pack_path=canonical_pack_path,
        history_df=df,
        profile=profile,
        pred_col=pred_col,
    )

    metrics = compute_metrics(result_df)

    health["downward_correction_count"] = int(
        (result_df.get("downward_amount", pd.Series(0.0)) < -1e-6).sum()
    )
    health["negative_MAE_improvement"] = metrics.get("negative_MAE_before", 0) - metrics.get("negative_MAE_after", 0)
    health["low_valley_MAE_improvement"] = metrics.get("low_valley_MAE_before", 0) - metrics.get("low_valley_MAE_after", 0)
    health["overall_sMAPE_improvement"] = metrics.get("overall_sMAPE_improvement", 0)
    health["high_spike_MAE_improvement"] = metrics.get("high_spike_MAE_improvement", 0)
    health["normal_degradation"] = metrics.get("normal_degradation", 0)

    # ── GO / NO-GO / DATA-LIMITED ──────────────────────────────────
    if health["DATA_LIMITED"]:
        health["verdict"] = "DATA-LIMITED"
        if health["low_valley_MAE_improvement"] >= 0 and health["overall_sMAPE_improvement"] >= -0.3:
            health["verdict"] = "DATA-LIMITED (LV ok)"
    else:
        neg_ok = health["negative_MAE_improvement"] >= 0 or health["low_valley_MAE_improvement"] >= 0
        smape_ok = health["overall_sMAPE_improvement"] >= -0.3
        hs_ok = health["high_spike_MAE_improvement"] >= -3.0
        norm_ok = health["normal_degradation"] <= 0.5
        health["verdict"] = "GO" if (neg_ok and smape_ok and hs_ok and norm_ok) else "NO-GO"

    # ── Write outputs ────────────────────────────────────────────────
    json_path = out_dir / "residual_health.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(health, f, indent=2, ensure_ascii=False)

    md = _format_markdown(health)
    md_path = out_dir / "residual_health.md"
    md_path.write_text(md, encoding="utf-8")

    # ── Print ────────────────────────────────────────────────────────
    _print_health(health)

    return health


def _format_markdown(health: dict[str, Any]) -> str:
    lines = [
        "# P5M Residual Health Monitor",
        "",
        f"- **Generated:** {health.get('generated_at', 'N/A')}",
        f"- **Profile:** {health.get('profile', 'N/A')}",
        f"- **Canonical pack:** {health.get('canonical_pack', 'N/A')}",
        "",
        "## Event Counts",
        f"- **Total rows:** {health.get('total_rows', 'N/A')}",
        f"- **Business days:** {health.get('business_days', 'N/A')}",
        f"- **Negative count:** {health.get('negative_count', 'N/A')}",
        f"- **Low valley count:** {health.get('low_valley_count', 'N/A')}",
        "",
        "## Trigger Rates (prob > 0.3)",
        f"- **Negative trigger rate:** {health.get('negative_trigger_rate', 'N/A')}",
        f"- **Low valley trigger rate:** {health.get('low_valley_trigger_rate', 'N/A')}",
        "",
        "## Overlap",
        f"- **High-spike / low-valley overlap:** {health.get('high_spike_overlap_count', 'N/A')}",
        f"- **Downward corrections applied:** {health.get('downward_correction_count', 'N/A')}",
        "",
        "## Correction Impact",
        f"- **Negative MAE improvement:** {health.get('negative_MAE_improvement', 'N/A'):+.2f}",
        f"- **Low valley MAE improvement:** {health.get('low_valley_MAE_improvement', 'N/A'):+.2f}",
        f"- **Overall sMAPE improvement:** {health.get('overall_sMAPE_improvement', 'N/A'):+.4f}",
        f"- **High spike MAE improvement:** {health.get('high_spike_MAE_improvement', 'N/A'):+.2f}%",
        f"- **Normal degradation:** {health.get('normal_degradation', 'N/A'):+.4f}",
        "",
        "## Verdict",
        f"**{health.get('verdict', 'UNKNOWN')}**",
        "",
    ]
    return "\n".join(lines)


def _print_health(health: dict[str, Any]) -> None:
    print(f"\n  {'='*55}")
    print(f"  P5M Residual Health — {health.get('profile', 'N/A')}")
    print(f"  {'='*55}")
    print(f"  Total rows:           {health.get('total_rows', 'N/A')}")
    print(f"  Negative count:       {health.get('negative_count', 'N/A')}")
    print(f"  Low valley count:     {health.get('low_valley_count', 'N/A')}")
    print(f"  DATA_LIMITED:         {health.get('DATA_LIMITED', 'N/A')}")
    print(f"  {'─'*55}")
    print(f"  Neg trigger rate:     {health.get('negative_trigger_rate', 0):.4f}")
    print(f"  LV trigger rate:      {health.get('low_valley_trigger_rate', 0):.4f}")
    print(f"  HS overlap:           {health.get('high_spike_overlap_count', 'N/A')}")
    print(f"  Downward corrs:       {health.get('downward_correction_count', 'N/A')}")
    print(f"  {'─'*55}")
    print(f"  Neg MAE improvement:  {health.get('negative_MAE_improvement', 0):+.2f}")
    print(f"  LV MAE improvement:   {health.get('low_valley_MAE_improvement', 0):+.2f}")
    print(f"  sMAPE improvement:    {health.get('overall_sMAPE_improvement', 0):+.4f}")
    print(f"  HS MAE improvement:   {health.get('high_spike_MAE_improvement', 0):+.2f}%")
    print(f"  Normal degradation:   {health.get('normal_degradation', 0):+.4f}")
    print(f"  {'─'*55}")
    print(f"  Verdict:              {health.get('verdict', 'UNKNOWN')}")
    print(f"  {'='*55}")
    print(f"  Output: residual_health.json / .md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor P5M negative price module residual health")
    parser.add_argument("--canonical-pack", required=True, help="Path to canonical evaluation pack CSV")
    parser.add_argument("--out-dir", default="reports/local/p5m_monitor", help="Output directory")
    parser.add_argument("--profile", default="conservative", choices=["conservative", "moderate", "aggressive"],
                        help="Correction profile")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small window)")
    parser.add_argument("--pred-col", default="base_fused_pred", help="Prediction column name")
    parser.add_argument("--risk-path", default=None, help="Path to pre-computed risk CSV")
    args = parser.parse_args()

    monitor_health(
        canonical_pack_path=args.canonical_pack,
        out_dir=args.out_dir,
        profile_name=args.profile,
        pred_col=args.pred_col,
        risk_path=args.risk_path,
    )


if __name__ == "__main__":
    main()
