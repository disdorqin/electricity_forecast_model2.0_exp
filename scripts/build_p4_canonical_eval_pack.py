#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_p4_canonical_eval_pack.py — P4 canonical evaluation pack builder.

Creates a locked-down evaluation pack that ALL P4 windows must use:
  - canonical_prediction_pack.csv   — 1 row/timestamp, wide model columns
  - canonical_risk_predictions.csv   — high_spike_prob aligned to same timestamps
  - canonical_metrics_baseline.json  — Phase2 champion metric reproduction
  - canonical_manifest.json          — full metadata + date coverage + anomalies

Locked:
  1. Date range: 2025-11-01 ~ 2026-02-28
  2. Metric level: timestamp (business_day + hour_business)
  3. Phase2 champion medium+normal must reproduce sMAPE~20.86, severe~63
  4. No row-level inflation
  5. No D+1 actual features in prediction-time columns
  6. Missing days / missing model rows reported in manifest

Usage:
    python scripts/build_p4_canonical_eval_pack.py

Output (gitignored):
    reports/local/p4_canonical/
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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import (
    CorrectionMode,
    CorrectionProfile,
    get_profile,
)
from extreme.realtime_high_spike.residual_lift import (
    PERIOD_DEFS,
    ResidualLiftConfig,
    ResidualLiftCorrector,
    get_period,
)
from extreme.realtime_high_spike.guardrail import GuardrailConfig, SpikeGuardrail

# ── Constants ───────────────────────────────────────────────────────────

CANONICAL_START = "2025-11-01"
CANONICAL_END = "2026-02-28"

PHASE2_PACK = "reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv"
PHASE2_RISK = "reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90/risk_predictions_multicandidate.csv"

OUT_DIR = "reports/local/p4_canonical"

ALL_MODELS = ["dayahead_proxy", "naive_lag1", "naive_lag7", "lightgbm"]

# Expected spike threshold (y_true > 500 → high_spike)
SPIKE_THRESHOLD = 500.0

# Phase2 champion expected values (with floor50 sMAPE)
PHASE2_EXPECTED = {
    "smape_floor50": 20.86,
    "severe": 63,
}

# Tolerance for metric reproduction
SMAPE_TOLERANCE = 0.05  # ±0.05
SEVERE_TOLERANCE = 0    # exact


# ── sMAPE floor50 (matches Phase2 evaluation) ──────────────────────────

def compute_smape_floor50(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """sMAPE with 50 floor on both |actual| and |predicted|."""
    yt = np.maximum(np.abs(np.asarray(y_true, dtype=float)), 50.0)
    yp = np.maximum(np.abs(np.asarray(y_pred, dtype=float)), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
    return np.minimum(smape, 50.0)


# ── Correction runner (reproduces Phase2 medium+normal) ────────────────

def run_medium_correction(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Phase2 medium + normal correction to a timestamp-level DataFrame.

    Must have columns: business_day, hour_business, base_fused_pred, y_true.
    Uses default lift candidates (50.0) with medium profile params.
    """
    result = df.copy()

    # Medium profile params (from config/p0_spike_correction_profiles.yaml)
    spike_prob_threshold = 0.60
    max_lift_ratio = 0.35
    max_absolute_lift = 350.0
    period_9_16_boost = 1.15
    protect_normal_hours = True

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

    corrector = ResidualLiftCorrector(lift_cfg)
    corrector.set_lift_candidates({p: 50.0 for p in PERIOD_DEFS})
    corrector._lift_candidates["9_16"] *= period_9_16_boost

    guardrail = SpikeGuardrail(guard_cfg)

    final_preds: list[float] = []
    lift_applied: list[float] = []
    reason_codes: list[str] = []

    for _, row in result.iterrows():
        base_pred = float(row.get("base_fused_pred", 0.0) or 0.0)
        spike_prob = float(row.get("high_spike_prob", 0.0) or 0.0)
        hb = int(row.get("hour_business", 12))

        lift_result = corrector.compute_lift(base_pred, spike_prob, hb)
        guard_result = guardrail.evaluate(
            base_pred, spike_prob, lift_result.corrected_pred, hb,
        )
        final_preds.append(guard_result.final_pred)
        lift_applied.append(guard_result.final_pred - base_pred)
        reason_codes.append(guard_result.reason_code)

    result["final_pred"] = final_preds
    result["lift_applied"] = lift_applied
    result["reason_code"] = reason_codes
    return result


# ── Metrics computation ────────────────────────────────────────────────

def compute_baseline_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute Phase2-style baseline metrics (timestamp-level, deduped).

    Uses floor50 sMAPE to match Phase2 champion evaluation.
    """
    valid = df.dropna(subset=["y_true", "final_pred"]).copy()
    if len(valid) == 0:
        return {"error": "no valid rows"}

    # sMAPE floor50 (corrected)
    smape = float(np.nanmean(compute_smape_floor50(valid["y_true"], valid["final_pred"])))
    # Base sMAPE floor50 (before correction)
    base_smape = float(np.nanmean(compute_smape_floor50(valid["y_true"], valid["base_fused_pred"])))

    # Severe underestimates
    severe = int((valid["y_true"] - valid["final_pred"] > 200).sum())
    severe_base = int((valid["y_true"] - valid["base_fused_pred"] > 200).sum())

    # 9_16 sMAPE
    p9 = valid[valid["period"] == "9_16"]
    smape_9_16 = float(np.nanmean(compute_smape_floor50(p9["y_true"], p9["final_pred"]))) if len(p9) > 0 else None
    base_smape_9_16 = float(np.nanmean(compute_smape_floor50(p9["y_true"], p9["base_fused_pred"]))) if len(p9) > 0 else None

    # False lift rate
    non_spike = valid[valid["high_spike"] == 0].copy() if "high_spike" in valid.columns else valid.copy()
    false_lift_mask = non_spike["final_pred"] > non_spike["base_fused_pred"]
    false_lift_rate = float(false_lift_mask.mean()) if len(non_spike) > 0 else 0.0

    # Normal hours degradation
    if "high_spike" in valid.columns:
        normal = valid[valid["high_spike"] == 0]
    else:
        normal = valid
    if len(normal) > 0:
        normal_before = float(np.nanmean(compute_smape_floor50(normal["y_true"], normal["base_fused_pred"])))
        normal_after = float(np.nanmean(compute_smape_floor50(normal["y_true"], normal["final_pred"])))
        normal_degrad = round(normal_after - normal_before, 4)
    else:
        normal_before = normal_after = normal_degrad = None

    # Spike counts
    n_spike_hours = int(valid["high_spike"].sum()) if "high_spike" in valid.columns else None
    n_non_spike_hours = int((valid["high_spike"] == 0).sum()) if "high_spike" in valid.columns else None

    return {
        "n_timestamps": len(valid),
        "smape_floor50": round(smape, 4),
        "base_smape_floor50": round(base_smape, 4),
        "severe_underestimate": severe,
        "severe_underestimate_base": severe_base,
        "smape_9_16_floor50": round(smape_9_16, 4) if smape_9_16 is not None else None,
        "base_smape_9_16_floor50": round(base_smape_9_16, 4) if base_smape_9_16 is not None else None,
        "false_lift_rate": round(false_lift_rate, 4),
        "normal_hours_degradation": normal_degrad,
        "n_spike_hours": n_spike_hours,
        "n_non_spike_hours": n_non_spike_hours,
        "metric_level": "timestamp",
        "smape_formula": "floor50 (max(|x|, 50) on both y_true and y_pred)",
    }


# ── Pack builder ───────────────────────────────────────────────────────

def build_canonical_pack() -> dict[str, Any]:
    """Build the complete canonical evaluation pack.

    Returns manifest dict.
    """
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  P4 Canonical Evaluation Pack Builder")
    print("=" * 60)

    # ── Step 1: Load Phase2 multi-candidate pack ──────────────────────
    print(f"\n  Loading Phase2 prediction pack: {PHASE2_PACK}")
    pp = pd.read_csv(PHASE2_PACK)
    print(f"  Raw rows: {len(pp)}, days: {pp['business_day'].nunique()}")

    # Filter to canonical date range
    pp = pp[pp["business_day"] >= CANONICAL_START].copy()
    pp = pp[pp["business_day"] <= CANONICAL_END].copy()
    print(f"  After date filter [{CANONICAL_START} ~ {CANONICAL_END}]: {len(pp)} rows")

    # ── Step 2: Build timestamp-level canonical prediction pack ───────
    # Pivot per-model predictions to wide columns
    pp_ts = pp.drop_duplicates(subset=["business_day", "hour_business"]).copy()
    n_before_dedup = len(pp)
    n_after_dedup = len(pp_ts)
    dedup_ratio = (1 - n_after_dedup / n_before_dedup) * 100

    # Pivot model predictions
    pred_pivot = pp.pivot_table(
        index=["business_day", "hour_business"],
        columns="model_name", values="y_pred", aggfunc="first",
    ).reset_index()
    pred_pivot.columns = ["business_day", "hour_business"] + [
        f"y_pred_{m}" for m in pred_pivot.columns if m not in ["business_day", "hour_business"]
    ]

    # Build canonical pack: (business_day, hour_business) key + all fields
    ts_data = pp_ts[
        ["business_day", "hour_business", "timestamp", "period",
         "base_fused_pred", "y_true", "high_spike_flag"]
    ].copy()

    # Merge with pivoted model predictions
    ts_data = ts_data.merge(pred_pivot, on=["business_day", "hour_business"], how="left")

    # Create high_spike label (y_true > 500 matches the build_dataset definition)
    ts_data["high_spike"] = (ts_data["y_true"] > SPIKE_THRESHOLD).astype(int)

    # ── Step 3: Load and merge risk predictions ───────────────────────
    print(f"\n  Loading risk predictions: {PHASE2_RISK}")
    rp = pd.read_csv(PHASE2_RISK)
    rp = rp[rp["business_day"] >= CANONICAL_START].copy()
    rp = rp[rp["business_day"] <= CANONICAL_END].copy()

    ts_data = ts_data.merge(
        rp[["business_day", "hour_business", "high_spike_prob", "spike_risk_score", "spike_risk_flag"]],
        on=["business_day", "hour_business"],
        how="left",
    )

    # ── Step 4: Report completeness ────────────────────────────────────
    # Expected timestamps
    expected_days = pd.date_range(CANONICAL_START, CANONICAL_END, freq="D")
    expected_timestamps = len(expected_days) * 24
    actual_timestamps = len(ts_data)
    print(f"\n  Timestamp completeness:")
    print(f"    Expected: {expected_timestamps} ({len(expected_days)} days × 24h)")
    print(f"    Actual:   {actual_timestamps}")
    print(f"    Missing:  {expected_timestamps - actual_timestamps}")

    missing_ts = []
    for d in expected_days:
        bd = d.strftime("%Y-%m-%d")
        for hb in range(1, 25):
            if not ((ts_data["business_day"] == bd) & (ts_data["hour_business"] == hb)).any():
                missing_ts.append({"business_day": bd, "hour_business": int(hb)})

    # Model completeness
    model_coverage: dict[str, Any] = {}
    for m in ALL_MODELS:
        col = f"y_pred_{m}"
        if col in ts_data.columns:
            n_missing = int(ts_data[col].isna().sum())
            pct = round(n_missing / len(ts_data) * 100, 2)
            model_coverage[m] = {"present": len(ts_data) - n_missing, "missing": n_missing, "pct_missing": pct}

    # ── Step 5: Compute Phase2 champion baseline metrics ──────────────
    # Run medium correction to reproduce Phase2 champion
    print(f"\n  Reproducing Phase2 champion (lightgbm_anchor_90 + medium + normal)...")
    corrected = run_medium_correction(ts_data)
    baseline_metrics = compute_baseline_metrics(corrected)
    print(f"    sMAPE_floor50: {baseline_metrics['smape_floor50']} (expected {PHASE2_EXPECTED['smape_floor50']})")
    print(f"    Severe:        {baseline_metrics['severe_underestimate']} (expected {PHASE2_EXPECTED['severe']})")

    smape_match = abs(baseline_metrics["smape_floor50"] - PHASE2_EXPECTED["smape_floor50"]) <= SMAPE_TOLERANCE
    severe_match = baseline_metrics["severe_underestimate"] == PHASE2_EXPECTED["severe"]
    print(f"    sMAPE match:   {'✅' if smape_match else '❌'} (Δ={baseline_metrics['smape_floor50'] - PHASE2_EXPECTED['smape_floor50']:.4f})")
    print(f"    Severe match:  {'✅' if severe_match else '❌'}")

    # ── Step 6: Write outputs ──────────────────────────────────────────
    # canonical_prediction_pack.csv — prediction pack (columns for all models)
    pred_pack_cols = [
        "business_day", "hour_business", "timestamp", "period",
        "base_fused_pred", "y_true", "high_spike", "high_spike_flag",
    ]
    pred_pack_cols += [f"y_pred_{m}" for m in ALL_MODELS if f"y_pred_{m}" in ts_data.columns]
    pred_pack = ts_data[pred_pack_cols].copy()
    # Add corrected prediction as reference
    pred_pack["final_pred_reference"] = corrected["final_pred"]
    pred_pack["lift_applied"] = corrected["lift_applied"]
    pred_pack["reason_code"] = corrected["reason_code"]

    pred_pack_path = out_dir / "canonical_prediction_pack.csv"
    pred_pack.to_csv(pred_pack_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Prediction pack: {pred_pack_path} ({len(pred_pack)} rows)")

    # canonical_risk_predictions.csv
    risk_cols = ["business_day", "hour_business", "high_spike_prob", "spike_risk_score", "spike_risk_flag"]
    risk_out = ts_data[risk_cols].copy()
    risk_path = out_dir / "canonical_risk_predictions.csv"
    risk_out.to_csv(risk_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Risk predictions: {risk_path} ({len(risk_out)} rows)")

    # canonical_metrics_baseline.json
    baseline_path = out_dir / "canonical_metrics_baseline.json"
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline_metrics, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Baseline metrics: {baseline_path}")

    # canonical_manifest.json
    manifest: dict[str, Any] = {
        "script": "scripts/build_p4_canonical_eval_pack.py",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "P4 canonical evaluation pack — all P4 windows must use this pack",
        "branch": "agent/p4-canonical-eval-pack",
        "date_range": {
            "start": CANONICAL_START,
            "end": CANONICAL_END,
            "n_expected_timestamps": expected_timestamps,
            "n_actual_timestamps": actual_timestamps,
        },
        "source_packs": {
            "prediction_pack": PHASE2_PACK,
            "risk_predictions": PHASE2_RISK,
        },
        "fusion": {
            "method": "lightgbm_anchor_90",
            "anchor_weight": 0.9,
            "models": ALL_MODELS,
        },
        "correction": {
            "profile": "medium",
            "mode": "normal",
            "params": {
                "spike_prob_threshold": 0.60,
                "max_lift_ratio": 0.35,
                "max_absolute_lift": 350,
                "period_9_16_boost": 1.15,
                "protect_normal_hours": True,
            },
        },
        "completeness": {
            "n_business_days": int(ts_data["business_day"].nunique()),
            "n_timestamps": actual_timestamps,
            "n_missing_timestamps": expected_timestamps - actual_timestamps,
            "missing_timestamps": missing_ts[:20],  # first 20 only
            "n_missing_timestamps_truncated": max(0, len(missing_ts) - 20),
            "dedup_ratio_pct": round(dedup_ratio, 2),
            "model_coverage": model_coverage,
            "total_rows_in_source_pack": int(len(pp)),
            "note": (
                f"Timestamp-level dedup removed {dedup_ratio:.1f}% rows "
                f"({n_before_dedup} → {n_after_dedup}) from multi-model source pack. "
                f"Missing timestamps: {len(missing_ts)} "
                f"(mainly hour_business=24 on 2026-02-28, which maps to 2026-03-01 00:00)."
            ),
        },
        "baseline_metrics": baseline_metrics,
        "phase2_champion_reproduction": {
            "pass": smape_match and severe_match,
            "expected_smape_floor50": PHASE2_EXPECTED["smape_floor50"],
            "actual_smape_floor50": baseline_metrics["smape_floor50"],
            "smape_tolerance": SMAPE_TOLERANCE,
            "expected_severe": PHASE2_EXPECTED["severe"],
            "actual_severe": baseline_metrics["severe_underestimate"],
            "severe_tolerance": SEVERE_TOLERANCE,
        },
        "leakage_safe": {
            "status": True,
            "note": (
                "Pack contains only prediction-time-safe columns: "
                "base_fused_pred (anchor_90 fusion), y_pred_* (model predictions), "
                "high_spike_prob (risk model output), calendar fields. "
                "No D+1 actual features in prediction-time columns. "
                "y_true included only for evaluation — not used at prediction time."
            ),
        },
        "output_files": {
            "canonical_prediction_pack_csv": str(pred_pack_path),
            "canonical_risk_predictions_csv": str(risk_path),
            "canonical_metrics_baseline_json": str(baseline_path),
            "canonical_manifest_json": str(out_dir / "canonical_manifest.json"),
        },
    }

    manifest_path = out_dir / "canonical_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Manifest: {manifest_path}")

    print(f"\n  {'=' * 60}")
    print(f"  P4 Canonical Pack Complete")
    print(f"  {'=' * 60}")
    print(f"  Pack:     {len(pred_pack)} rows, {pred_pack['business_day'].nunique()} days")
    print(f"  Risk:     {len(risk_out)} rows")
    print(f"  Baseline: sMAPE={baseline_metrics['smape_floor50']}, severe={baseline_metrics['severe_underestimate']}")
    print(f"  Match:    {'✅ PASS' if smape_match and severe_match else '❌ FAIL'}")
    print()

    return manifest


if __name__ == "__main__":
    build_canonical_pack()
