#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p34_timesfm_diversity_smoke.py — P3.4 TimesFM Diversity Smoke Test.

Tests whether TimesFM predictions, added as a small-weight diversity signal,
improves high-spike hour predictions vs LightGBM-only and Phase2 anchor.

Pipeline:
    1. Read TimesFM top-10 spike days predictions
    2. Create fused base predictions for each weight config:
       - LightGBM-only (no fusion, raw LightGBM pred)
       - Phase2 anchor (0.9 lightgbm + 0.033 each of 3 baselines)
       - Phase2 anchor + TimesFM @ 0.05, 0.10, 0.15
    3. Run Phase2 medium correction on each config
    4. Compare sMAPE, severe, high_spike MAE, false_lift

Usage:
    python scripts/evaluate_p34_timesfm_diversity_smoke.py \\
        --timesfm-csv reports/local/p33_extra_signal/timesfm_top10_spike_days.csv \\
        --prediction-pack reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv \\
        --risk-predictions reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/risk_predictions_multicandidate.csv \\
        --profile-config config/p0_spike_correction_profiles.yaml \\
        --out-dir reports/local/p34_timesfm_diversity_smoke

Output:
    {config}/predictions.csv           — corrected predictions per config
    {config}/metrics.json              — metrics per config
    comparison_table.md                — markdown comparison table
    comparison_summary.json            — all results
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
from scripts.evaluate_realtime_spike_correction import compute_all_metrics

warnings.filterwarnings("ignore", category=FutureWarning)


# ── Metrics (subset for spike-day focus) ──────────────────────────────

def compute_smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_spike_day_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute metrics focused on spike-day evaluation."""
    y_true = df["y_true"].values
    y_pred = df["final_pred"].values

    smape = compute_smape_floor50(y_true, y_pred)
    severe = int((y_true - y_pred > 200).sum())

    # High-spike MAE (hours where actual > p90 across these spike days)
    p90 = float(np.percentile(y_true, 90))
    spike_mask = y_true > p90
    if spike_mask.sum() > 0:
        high_spike_mae = float(np.mean(np.abs(y_true[spike_mask] - y_pred[spike_mask])))
        high_spike_smape = compute_smape_floor50(y_true[spike_mask], y_pred[spike_mask])
    else:
        high_spike_mae = None
        high_spike_smape = None

    base_pred = df["base_fused_pred"].values if "base_fused_pred" in df.columns else y_pred
    severe_base = int((y_true - base_pred > 200).sum())

    return {
        "smape_floor50": round(smape, 4),
        "severe_underestimate": severe,
        "severe_underestimate_base": severe_base,
        "high_spike_mae": round(high_spike_mae, 4) if high_spike_mae is not None else None,
        "high_spike_smape": round(high_spike_smape, 4) if high_spike_smape is not None else None,
        "n_timestamps": len(df),
        "n_spike_hours": int(spike_mask.sum()) if spike_mask.sum() > 0 else 0,
    }


# ── Prediction pack builder ───────────────────────────────────────────

def build_prediction_pack(
    base_fused: pd.Series,
    y_true: pd.Series,
    keys: pd.DataFrame,
    out_dir: Path,
    label: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pack = keys[["business_day", "hour_business"]].copy()
    pack["base_fused_pred"] = base_fused.values
    pack["y_true"] = y_true.values

    out_path = out_dir / f"prediction_pack_{label}.csv"
    pack.to_csv(out_path, index=False)
    print(f"  [INFO] Prediction pack: {out_path} ({len(pack)} rows)")
    return out_path


# ── Run config ────────────────────────────────────────────────────────

def run_config(
    config_name: str,
    base_fused: pd.Series,
    y_true: pd.Series,
    keys: pd.DataFrame,
    risk_predictions_path: Path,
    profile: CorrectionProfile,
    out_dir: Path,
) -> dict[str, Any]:
    config_out = out_dir / config_name
    config_out.mkdir(parents=True, exist_ok=True)

    pack_path = build_prediction_pack(base_fused, y_true, keys, config_out, config_name)

    result = run_correction(
        prediction_pack_path=str(pack_path),
        risk_predictions_path=str(risk_predictions_path),
        profile=profile,
    )

    result_csv = config_out / "predictions.csv"
    result.to_csv(result_csv, index=False)

    if "hour_business" in result.columns:
        n_before = len(result)
        result = result.drop_duplicates(subset=["business_day", "hour_business"]).copy()
        if len(result) < n_before:
            print(f"  [INFO] Dedup: {n_before} -> {len(result)} rows")

    metrics = compute_spike_day_metrics(result)
    try:
        full_metrics = compute_all_metrics(result)
        metrics["false_lift_rate"] = full_metrics.get("false_lift_rate")
        metrics["lift_applied_count"] = full_metrics.get("lift_applied_count")
        metrics["normal_hours_degradation"] = full_metrics.get("normal_hours_degradation")
    except Exception:
        metrics["false_lift_rate"] = None

    metrics["config"] = config_name
    metrics["n_timestamps"] = len(result)

    metrics_path = config_out / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"  sMAPE:           {metrics.get('smape_floor50', 'N/A')}")
    print(f"  Severe:          {metrics.get('severe_underestimate', 'N/A')}")
    print(f"  High-spike MAE:  {metrics.get('high_spike_mae', 'N/A')}")
    print(f"  False lift:      {metrics.get('false_lift_rate', 'N/A')}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P3.4 TimesFM Diversity Smoke Test.",
    )
    parser.add_argument("--timesfm-csv", required=True,
                        help="Path to TimesFM spike days CSV")
    parser.add_argument("--prediction-pack", required=True,
                        help="Path to Phase2 multi-candidate prediction pack")
    parser.add_argument("--risk-predictions", required=True,
                        help="Path to risk predictions CSV")
    parser.add_argument("--profile-config",
                        default="config/p0_spike_correction_profiles.yaml")
    parser.add_argument("--out-dir",
                        default="reports/local/p34_timesfm_diversity_smoke")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    tf_path = Path(args.timesfm_csv)
    pp_path = Path(args.prediction_pack)
    rp_path = Path(args.risk_predictions)

    for p in [tf_path, pp_path, rp_path]:
        if not p.exists():
            sys.exit(f"Error: {p} not found")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  P3.4 TimesFM Diversity Smoke Test")
    print("=" * 60)

    tf = pd.read_csv(tf_path)
    pack = pd.read_csv(pp_path)

    spike_days = sorted(tf["business_day"].unique())
    print(f"\n  Spike days ({len(spike_days)}): {spike_days}")

    pack_spike = pack[pack["business_day"].isin(spike_days)].copy()
    lgb_spike = pack_spike[pack_spike["model_name"] == "lightgbm"].copy()

    tf_aligned = tf[["business_day", "hour_business", "y_pred"]].copy()
    tf_aligned = tf_aligned.rename(columns={"y_pred": "timesfm_pred"})

    # Key dataframe (unique timestamps)
    keys = lgb_spike[["business_day", "hour_business"]].drop_duplicates().reset_index(drop=True)

    merged = keys.copy()
    lgb_vals = lgb_spike[["business_day", "hour_business", "y_pred", "base_fused_pred", "y_true"]].copy()
    merged = merged.merge(lgb_vals, on=["business_day", "hour_business"], how="left")
    merged = merged.merge(tf_aligned, on=["business_day", "hour_business"], how="left")
    merged["timesfm_pred"] = merged["timesfm_pred"].fillna(merged["base_fused_pred"])

    print(f"  Aligned timestamps: {len(merged)}")

    # ── Correction profile ───────────────────────────────────────────
    profile = CorrectionProfile(
        name="medium",
        spike_prob_threshold=0.60,
        max_lift_ratio=0.35,
        max_absolute_lift=350.0,
        protect_normal_hours=True,
        period_9_16_boost=1.15,
    )

    # ── Configurations ──────────────────────────────────────────────
    configs = {
        "lightgbm_only": merged["y_pred"],
        "phase2_anchor": merged["base_fused_pred"],
    }
    for tf_w in [0.05, 0.10, 0.15]:
        name = f"anchor_plus_timesfm_{tf_w:.2f}"
        configs[name] = (1.0 - tf_w) * merged["base_fused_pred"] + tf_w * merged["timesfm_pred"]

    y_true = merged["y_true"]

    # ── Run each config ──────────────────────────────────────────────
    all_metrics: dict[str, dict[str, Any]] = {}
    t_start = time.time()

    for config_name, base_pred in configs.items():
        print(f"\n  {'─' * 50}")
        print(f"  Config: {config_name}")
        print(f"  {'─' * 50}")

        metrics = run_config(
            config_name=config_name,
            base_fused=base_pred,
            y_true=y_true,
            keys=keys,
            risk_predictions_path=rp_path,
            profile=profile,
            out_dir=out_dir,
        )
        all_metrics[config_name] = metrics

    total_time = time.time() - t_start

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Comparison")
    print(f"{'=' * 60}")

    header = "| Config | sMAPE | Severe | High-spike MAE | False Lift | Lift Count |"
    sep = "|-------|:-----:|:------:|:--------------:|:----------:|:----------:|"
    print(f"\n{header}\n{sep}")
    rows = []

    for cname in configs:
        m = all_metrics.get(cname, {})
        s = m.get("smape_floor50", "—")
        v = m.get("severe_underestimate", "—")
        h = m.get("high_spike_mae", "—")
        f = m.get("false_lift_rate", "—")
        lc = m.get("lift_applied_count", "—")
        row = f"| {cname} | {_fmt(s)} | {v} | {_fmt(h)} | {_fmt(f)} | {lc} |"
        rows.append(row)
        print(row)

    table_md = f"# P3.4 TimesFM Diversity Smoke — Comparison\n\n{header}\n{sep}\n" + "\n".join(rows) + "\n"
    table_path = out_dir / "comparison_table.md"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table_md)

    # ── Summary JSON ─────────────────────────────────────────────────
    summary = {
        "script": "scripts/evaluate_p34_timesfm_diversity_smoke.py",
        "timesfm_csv": str(tf_path),
        "prediction_pack": str(pp_path),
        "spike_days": spike_days,
        "profile": {"name": "medium", "mode": "normal"},
        "configs": all_metrics,
        "total_runtime_seconds": round(total_time, 1),
    }
    summary_path = out_dir / "comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary: {summary_path}")
    print(f"  Total runtime: {total_time:.0f}s")
    print("\nDone.")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v) if v is not None else "—"


if __name__ == "__main__":
    main()
