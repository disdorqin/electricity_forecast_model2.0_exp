#!/usr/bin/env python3
"""
calibrate_p5m_negative_risk.py — Calibrate negative/low-valley risk scoring.

Runs heuristic_v2 and rolling_ml_low_valley scorers on the canonical pack,
outputs risk CSVs, and summarizes trigger rates.

Usage:
    python scripts/calibrate_p5m_negative_risk.py \\
        --canonical-pack reports/local/canonical_eval_pack.csv \\
        --out-dir reports/local/p5m_calibration

Output:
    - risk_heuristic_v2.csv
    - risk_rolling_ml.csv (if ML training succeeds)
    - negative_risk_predictions.csv (consolidated for residual stack)
    - calibration_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme.negative_price.labels import add_all_labels
from extreme.negative_price.risk_model import (
    compute_heuristic_v2_risk,
    RollingLowValleyScorer,
    RollingMLConfig,
)


def run_calibration(
    canonical_pack_path: str | Path,
    out_dir: str | Path,
    max_days: int = 0,
    pred_col: str = "base_fused_pred",
) -> dict[str, Any]:
    """Run calibration on the canonical pack.

    Args:
        canonical_pack_path: Path to canonical evaluation pack CSV.
        out_dir: Output directory.
        max_days: Max days to process (0 = all).
        pred_col: Column name for predictions (used for labels).

    Returns:
        Summary dict with trigger rates and file paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(canonical_pack_path)
    if max_days > 0 and "business_day" in df.columns:
        days = sorted(df["business_day"].unique())[:max_days]
        df = df[df["business_day"].isin(days)].copy()

    df = add_all_labels(df, y_pred_col=pred_col)

    summary: dict[str, Any] = {
        "canonical_pack": str(canonical_pack_path),
        "total_rows": len(df),
        "business_days": int(df["business_day"].nunique()) if "business_day" in df.columns else 0,
        "negative_count": int((df.get("label_negative_price", 0) == 1).sum()),
        "low_valley_count": int((df.get("label_low_valley", 0) == 1).sum()),
        "scorers": {},
    }

    # ── Heuristic V2 ─────────────────────────────────────────────────
    print("\n  Running heuristic_v2...")
    heur = compute_heuristic_v2_risk(df, history_df=df, pred_col=pred_col)
    heur.to_csv(out_dir / "risk_heuristic_v2.csv", index=False)
    summary["scorers"]["heuristic_v2"] = _summarize_risk(heur, "heuristic_v2")

    # ── Rolling ML ───────────────────────────────────────────────────
    print("  Running rolling_ml_low_valley...")
    try:
        config = RollingMLConfig(
            train_window_days=30,
            target_label="combined",
            min_train_samples=50,
        )
        scorer = RollingLowValleyScorer(config)
        ml_result = scorer.fit_predict(df, pred_col=pred_col)
        ml_result.to_csv(out_dir / "risk_rolling_ml.csv", index=False)
        summary["scorers"]["rolling_ml"] = _summarize_risk(ml_result, "rolling_ml")
    except Exception as e:
        print(f"  WARNING: Rolling ML failed: {e}")
        summary["scorers"]["rolling_ml"] = {"status": f"FAILED: {e}"}

    # ── Consolidated risk CSV for residual stack ──────────────────────
    risk_out = heur[["business_day", "hour_business", "timestamp",
                     "negative_prob", "low_valley_prob"]].copy()
    risk_out["overestimate_low_prob"] = 0.0
    # Prefer rolling_ml scores when available
    if "rolling_ml" in summary["scorers"] and "status" not in summary["scorers"]["rolling_ml"]:
        try:
            merge_cols = ["business_day", "hour_business"]
            ml_merge = ml_result[merge_cols + ["negative_prob", "low_valley_prob",
                                                "overestimate_low_prob"]].copy()
            ml_merge.columns = merge_cols + ["neg_ml", "lv_ml", "over_ml"]
            risk_out = risk_out.merge(ml_merge, on=merge_cols, how="left")
            ml_mask = risk_out["neg_ml"].notna()
            risk_out.loc[ml_mask, "negative_prob"] = risk_out.loc[ml_mask, "neg_ml"]
            risk_out.loc[ml_mask, "low_valley_prob"] = risk_out.loc[ml_mask, "lv_ml"]
            risk_out.loc[ml_mask, "overestimate_low_prob"] = risk_out.loc[ml_mask, "over_ml"]
            risk_out = risk_out.drop(columns=["neg_ml", "lv_ml", "over_ml"], errors="ignore")
        except Exception as e:
            print(f"  WARNING: Risk consolidation failed: {e}")
    risk_out["risk_source"] = "calibrated_prob"
    risk_out["leakage_safe"] = True
    risk_out.to_csv(out_dir / "negative_risk_predictions.csv", index=False)

    # ── Write summary ────────────────────────────────────────────────
    report_path = out_dir / "calibration_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ── Print table ──────────────────────────────────────────────────
    print(f"\n  {'='*55}")
    print(f"  Calibration Summary")
    print(f"  {'='*55}")
    print(f"  Total rows:          {summary['total_rows']}")
    print(f"  Business days:       {summary['business_days']}")
    print(f"  Negative price:      {summary['negative_count']}")
    print(f"  Low valley:          {summary['low_valley_count']}")
    print(f"  {'─'*55}")
    for name, s in summary["scorers"].items():
        if "status" in s and s["status"].startswith("FAILED"):
            print(f"  {name:<20} FAILED")
            continue
        print(f"  {name:<20} neg_prob_mean={s.get('neg_prob_mean', 'N/A')}")
        print(f"  {'':<20} lv_prob_mean={s.get('lv_prob_mean', 'N/A')}")
        print(f"  {'':<20} trigger_neg={s.get('trigger_rate_neg_0.3', 'N/A')} (at p>0.3)")
        print(f"  {'':<20} trigger_lv={s.get('trigger_rate_lv_0.3', 'N/A')} (at p>0.3)")

    print(f"\n  Output: {out_dir}/")
    return summary


def _summarize_risk(df: pd.DataFrame, source: str) -> dict[str, Any]:
    """Compute trigger rate statistics for a risk DataFrame."""
    result: dict[str, Any] = {"source": source}
    neg_prob = df.get("negative_prob", pd.Series(0.0))
    lv_prob = df.get("low_valley_prob", pd.Series(0.0))

    result["neg_prob_mean"] = round(float(neg_prob.mean()), 4)
    result["neg_prob_std"] = round(float(neg_prob.std()), 4)
    result["neg_prob_p95"] = round(float(neg_prob.quantile(0.95)), 4)
    result["lv_prob_mean"] = round(float(lv_prob.mean()), 4)
    result["lv_prob_std"] = round(float(lv_prob.std()), 4)
    result["lv_prob_p95"] = round(float(lv_prob.quantile(0.95)), 4)

    # Trigger rates at different thresholds
    for thresh in [0.2, 0.3, 0.4, 0.5]:
        result[f"trigger_rate_neg_{thresh}"] = round(float((neg_prob > thresh).mean()), 4)
        result[f"trigger_rate_lv_{thresh}"] = round(float((lv_prob > thresh).mean()), 4)

    # Count active hours
    result["active_hours_neg"] = int((neg_prob > 0.3).sum())
    result["active_hours_lv"] = int((lv_prob > 0.3).sum())
    result["total_hours"] = len(df)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate negative/low-valley risk scoring")
    parser.add_argument("--canonical-pack", required=True, help="Path to canonical evaluation pack CSV")
    parser.add_argument("--out-dir", default="reports/local/p5m_calibration", help="Output directory")
    parser.add_argument("--max-days", type=int, default=0, help="Max days to process (0 = all)")
    parser.add_argument("--pred-col", default="base_fused_pred", help="Prediction column name")
    args = parser.parse_args()

    run_calibration(
        canonical_pack_path=args.canonical_pack,
        out_dir=args.out_dir,
        max_days=args.max_days,
        pred_col=args.pred_col,
    )


if __name__ == "__main__":
    main()
