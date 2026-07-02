#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p4_final_fusion_correction.py — P4 Final Fusion + Correction Evaluation.

Four combos:
  A) Phase2 canonical baseline          — use final_pred_reference directly
  B) W2 quantile LightGBM + Phase2 corr — replace lgbm, recompute base, run medium corr with P2 risk
  C) Phase2 base + W3 ml_gate           — use canonical base, run medium corr with W3 risk
  D) W2 quantile LightGBM + W3 ml_gate  — replace lgbm, recompute base, run medium corr with W3 risk

Two comparison tables:
  1. Full-window (2025-11-01 ~ 2026-02-28, 120 days): A vs C (non-W2 combos)
  2. Overlap-window (62 days where W2 has data): A_overlap vs B vs C_overlap vs D

Usage:
    python scripts/evaluate_p4_final_fusion_correction.py [--out-dir PATH]

Requires:
    reports/local/p4_canonical/canonical_prediction_pack.csv
    reports/local/p4_canonical/canonical_risk_predictions.csv
    reports/local/p4_lgbm_sota_tuning/full_obj_quantile_0p8/predictions.csv
    reports/local/p4_hybrid_gate/ml_gate/risk_predictions_gate.csv
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import CorrectionMode
from extreme.realtime_high_spike.residual_lift import (
    PERIOD_DEFS,
    ResidualLiftConfig,
    ResidualLiftCorrector,
)
from extreme.realtime_high_spike.guardrail import GuardrailConfig, SpikeGuardrail

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
CANONICAL_PACK = _PROJECT_ROOT / "reports/local/p4_canonical/canonical_prediction_pack.csv"
CANONICAL_RISK = _PROJECT_ROOT / "reports/local/p4_canonical/canonical_risk_predictions.csv"
W2_CSV = _PROJECT_ROOT / "reports/local/p4_lgbm_sota_tuning/full_obj_quantile_0p8/predictions.csv"
W3_RISK = _PROJECT_ROOT / "reports/local/p4_hybrid_gate/ml_gate/risk_predictions_gate.csv"
OUT_DIR_DEFAULT = _PROJECT_ROOT / "reports/local/p4_final_fusion_correction"

# Canonical champion baseline (from canonical_metrics_baseline.json)
PHASE2_CHAMPION = {"smape_floor50": 20.8675, "severe": 63, "false_lift_rate": 0.0664, "normal_degrad": -0.1929}

# DEPLOY GO thresholds
DEPLOY_GO = {"smape_floor50": 20.50, "severe": 63, "false_lift_rate": 0.10, "normal_degrad": 0.50}

# Anchor_90 weights
ANCHOR_WEIGHT = 0.9
BASELINE_WEIGHT = 0.1  # equally split among 3 baselines => 0.0333 each


# ══════════════════════════════════════════════════════════════════════════
#  sMAPE: EXACT canonical formula
# ══════════════════════════════════════════════════════════════════════════

def compute_smape_floor50(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> float:
    """Canonical floor50 sMAPE: floor |x| at 50 in BOTH numerator and denominator."""
    yt = np.maximum(np.abs(np.asarray(y_true, dtype=float)), 50.0)
    yp = np.maximum(np.abs(np.asarray(y_pred, dtype=float)), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
    return float(np.nanmean(np.minimum(smape, 50.0)))


# ══════════════════════════════════════════════════════════════════════════
#  In-memory correction (identical to build_p4_canonical_eval_pack)
# ══════════════════════════════════════════════════════════════════════════

def run_medium_correction(
    df: pd.DataFrame,
    risk_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply Phase2 medium + normal correction in-memory.

    df must have columns: business_day, hour_business, base_fused_pred, y_true.
    If risk_df is provided, merges on (business_day, hour_business) for
    high_spike_prob. Otherwise expects high_spike_prob in df.
    """
    result = df.copy()

    if risk_df is not None:
        merge_cols = ["business_day", "hour_business"]
        risk_merge = risk_df[merge_cols + ["high_spike_prob"]].copy()
        result = result.merge(risk_merge, on=merge_cols, how="left")

    if "high_spike_prob" not in result.columns:
        result["high_spike_prob"] = 0.0
    result["high_spike_prob"] = result["high_spike_prob"].fillna(0.0)

    # Medium profile params (from canonical pack builder)
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
    lift_applied_list: list[float] = []
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
        lift_applied_list.append(guard_result.final_pred - base_pred)
        reason_codes.append(guard_result.reason_code)

    result["final_pred"] = final_preds
    result["lift_applied"] = lift_applied_list
    result["reason_code"] = reason_codes
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Metrics (canonical, using floor50 sMAPE)
# ══════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, label: str = "") -> dict:
    """Compute canonical metrics matching canonical_metrics_baseline.json."""
    valid = df.dropna(subset=["y_true", "final_pred"]).copy()
    if len(valid) == 0:
        return {"combo": label, "error": "no valid rows", "n_timestamps": 0}

    y_true = valid["y_true"].values
    final_pred = valid["final_pred"].values
    base_pred = valid["base_fused_pred"].values

    smape = compute_smape_floor50(y_true, final_pred)
    base_smape = compute_smape_floor50(y_true, base_pred)

    # Severe: y_true - final_pred > 200 (canonical definition)
    severe = int((y_true - final_pred > 200).sum())
    severe_base = int((y_true - base_pred > 200).sum())

    # False lift: non-spike hours where final > base
    hs_col = "high_spike" if "high_spike" in valid.columns else None
    if hs_col is not None:
        non_spike = valid[valid[hs_col] == 0]
    else:
        non_spike = valid
    false_lift = float((non_spike["final_pred"] > non_spike["base_fused_pred"]).mean()) if len(non_spike) > 0 else 0.0

    # Normal hours degradation
    if hs_col is not None:
        normal = valid[valid[hs_col] == 0]
    else:
        normal = valid
    if len(normal) > 0:
        norm_before = compute_smape_floor50(normal["y_true"], normal["base_fused_pred"])
        norm_after = compute_smape_floor50(normal["y_true"], normal["final_pred"])
        norm_degrad = round(norm_after - norm_before, 4)
    else:
        norm_degrad = None

    # Spike hour stats
    if hs_col is not None:
        spike = valid[valid[hs_col] == 1]
    else:
        spike = pd.DataFrame()
    spike_mae = float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))) if len(spike) > 0 else None

    return {
        "combo": label,
        "n_timestamps": len(valid),
        "smape_floor50": round(smape, 4),
        "base_smape_floor50": round(base_smape, 4),
        "severe_underestimate": severe,
        "severe_underestimate_base": severe_base,
        "false_lift_rate": round(false_lift, 4),
        "normal_hours_degradation": norm_degrad if norm_degrad is not None else None,
        "spike_hours_mae": round(spike_mae, 2) if spike_mae is not None else None,
        "spike_hours_n": len(spike),
    }


# ══════════════════════════════════════════════════════════════════════════
#  GO assessment
# ══════════════════════════════════════════════════════════════════════════

def assess_go(metrics: dict) -> dict:
    """Assess DEPLOY GO / PAPER GO / NO-GO against canonical thresholds."""
    criteria = {}
    for name, key, threshold in [
        ("sMAPE <= 20.50", "smape_floor50", DEPLOY_GO["smape_floor50"]),
        ("Severe <= 63", "severe_underestimate", DEPLOY_GO["severe"]),
        ("False lift <= 10%", "false_lift_rate", DEPLOY_GO["false_lift_rate"]),
        ("Normal degrad <= 0.5", "normal_hours_degradation", DEPLOY_GO["normal_degrad"]),
    ]:
        actual = metrics.get(key)
        criteria[name] = {"threshold": threshold, "actual": actual, "met": actual is not None and actual <= threshold}

    all_met = all(c["met"] for c in criteria.values())

    # PAPER GO: sMAPE improves >= 1.0 vs Phase2 champion AND severe not worse
    si = PHASE2_CHAMPION["smape_floor50"] - metrics.get("smape_floor50", 999)
    sei = PHASE2_CHAMPION["severe"] - metrics.get("severe_underestimate", 999)
    paper = si >= 1.0 and sei >= 0

    verdict = "DEPLOY GO" if all_met else ("PAPER GO" if paper else "NO-GO")
    return {
        "verdict": verdict,
        "all_criteria_met": all_met,
        "paper_go": paper,
        "criteria": criteria,
        "smape_improvement_vs_champion": round(si, 2),
        "severe_improvement_vs_champion": sei,
    }


# ══════════════════════════════════════════════════════════════════════════
#  Data loading
# ══════════════════════════════════════════════════════════════════════════

def load_canonical(path: Path) -> pd.DataFrame:
    """Load canonical prediction pack with all model columns."""
    df = pd.read_csv(path)
    required = {"business_day", "hour_business", "base_fused_pred", "y_true",
                "y_pred_lightgbm", "y_pred_dayahead_proxy",
                "y_pred_naive_lag1", "y_pred_naive_lag7", "final_pred_reference"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Canonical pack missing columns: {missing}")
    df["business_day"] = df["business_day"].astype(str)
    return df


def load_w2(path: Path) -> pd.DataFrame:
    """Load W2 quantile LightGBM predictions.

    W2 hour convention: 1-24 (matches canonical hour_business).
    Maps each row to (business_day, hour_business) using:
        business_day = (ds - 1h).date
        hour_business = hour column (1-24)
    """
    df = pd.read_csv(path)
    if "pred_y" not in df.columns:
        raise ValueError(f"W2 CSV missing pred_y column. Found: {list(df.columns)}")
    if "hour" not in df.columns:
        raise ValueError(f"W2 CSV missing hour column")

    df["ds_dt"] = pd.to_datetime(df["ds"])
    # business_day = ds minus 1 hour (handles hour=24 correctly:
    #   ds="2025-11-02 00:00:00", hour=24 => business_day="2025-11-01")
    df["business_day"] = (df["ds_dt"] - pd.Timedelta(hours=1)).dt.strftime("%Y-%m-%d")
    df["hour_business"] = df["hour"].astype(int)  # already 1-24

    result = df[["business_day", "hour_business", "pred_y"]].copy()
    result = result.drop_duplicates(subset=["business_day", "hour_business"]).reset_index(drop=True)
    return result


def load_risk(path: Path) -> pd.DataFrame:
    """Load risk predictions."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "high_spike_prob" not in df.columns:
        raise ValueError(f"Risk CSV missing high_spike_prob. Found: {list(df.columns)}")
    df["business_day"] = df["business_day"].astype(str)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Combo builders
# ══════════════════════════════════════════════════════════════════════════

def build_combo_a(
    canonical: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """Combo A: Phase2 canonical baseline.

    Uses final_pred_reference directly — NO re-correction.
    """
    merged = keys.merge(
        canonical[["business_day", "hour_business", "base_fused_pred", "y_true",
                    "final_pred_reference", "high_spike"]],
        on=["business_day", "hour_business"], how="left",
    )
    merged.rename(columns={"final_pred_reference": "final_pred"}, inplace=True)
    merged["lift_applied"] = merged["final_pred"] - merged["base_fused_pred"]
    return merged


def build_combo_b(
    canonical: pd.DataFrame,
    w2: pd.DataFrame,
    canonical_risk: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """Combo B: W2 quantile LightGBM + Phase2 correction.

    Replaces y_pred_lightgbm with W2 pred_y, recomputes anchor_90 base,
    then runs medium correction with canonical risk.
    """
    merged = keys.merge(
        canonical[["business_day", "hour_business", "y_true", "high_spike",
                    "y_pred_dayahead_proxy", "y_pred_naive_lag1", "y_pred_naive_lag7"]],
        on=["business_day", "hour_business"], how="left",
    )
    merged = merged.merge(
        w2.rename(columns={"pred_y": "w2_pred"}),
        on=["business_day", "hour_business"], how="left",
    )
    # Replace lightgbm with W2, fallback to canonical lightgbm where W2 missing
    merged["y_pred_lightgbm"] = merged["w2_pred"].fillna(canonical["y_pred_lightgbm"])
    # Recompute anchor_90 base fusion
    baseline_mean = (
        merged["y_pred_dayahead_proxy"]
        + merged["y_pred_naive_lag1"]
        + merged["y_pred_naive_lag7"]
    ) / 3.0
    merged["base_fused_pred"] = (
        ANCHOR_WEIGHT * merged["y_pred_lightgbm"]
        + BASELINE_WEIGHT * baseline_mean
    )
    # Run correction
    result = run_medium_correction(merged, risk_df=canonical_risk)
    return result


def build_combo_c(
    canonical: pd.DataFrame,
    w3_risk: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """Combo C: Phase2 base + W3 ml_gate risk.

    Keeps canonical base_fused_pred, runs medium correction with W3 risk.
    """
    merged = keys.merge(
        canonical[["business_day", "hour_business", "base_fused_pred", "y_true", "high_spike"]],
        on=["business_day", "hour_business"], how="left",
    )
    result = run_medium_correction(merged, risk_df=w3_risk)
    return result


def build_combo_d(
    canonical: pd.DataFrame,
    w2: pd.DataFrame,
    w3_risk: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    """Combo D: W2 quantile LightGBM + W3 ml_gate risk.

    Replaces y_pred_lightgbm with W2 pred_y, recomputes anchor_90 base,
    runs medium correction with W3 risk.
    """
    merged = keys.merge(
        canonical[["business_day", "hour_business", "y_true", "high_spike",
                    "y_pred_dayahead_proxy", "y_pred_naive_lag1", "y_pred_naive_lag7"]],
        on=["business_day", "hour_business"], how="left",
    )
    merged = merged.merge(
        w2.rename(columns={"pred_y": "w2_pred"}),
        on=["business_day", "hour_business"], how="left",
    )
    merged["y_pred_lightgbm"] = merged["w2_pred"].fillna(canonical["y_pred_lightgbm"])
    baseline_mean = (
        merged["y_pred_dayahead_proxy"]
        + merged["y_pred_naive_lag1"]
        + merged["y_pred_naive_lag7"]
    ) / 3.0
    merged["base_fused_pred"] = (
        ANCHOR_WEIGHT * merged["y_pred_lightgbm"]
        + BASELINE_WEIGHT * baseline_mean
    )
    result = run_medium_correction(merged, risk_df=w3_risk)
    return result


# ══════════════════════════════════════════════════════════════════════════
#  Table builders
# ══════════════════════════════════════════════════════════════════════════

def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v) if v is not None else "\u2014"


TABLE_HEADER = (
    "| Combo | sMAPE | Base sMAPE | Severe | False Lift | Normal Degrad | N | Verdict |"
)
TABLE_SEP = "|------|:-----:|:----------:|:------:|:----------:|:-------------:|:---:|:-------:|"


_DASH = "\u2014"


def build_comparison_table(all_metrics: dict[str, dict], title: str) -> str:
    """Build a markdown comparison table from combo metrics."""
    lines = [f"### {title}", "", TABLE_HEADER, TABLE_SEP]
    for combo_key, label in [
        ("A", "A \u2014 Phase2 baseline"),
        ("B", "B \u2014 W2 + Phase2 corr"),
        ("C", "C \u2014 Phase2 + W3 gate"),
        ("D", "D \u2014 W2 + W3 gate"),
    ]:
        if combo_key not in all_metrics:
            continue
        m = all_metrics[combo_key]
        v = assess_go(m)
        severe_str = str(m.get("severe_underestimate", "")) or _DASH
        n_str = str(m.get("n_timestamps", "")) or _DASH
        lines.append(
            f"| {label} | {_fmt(m.get('smape_floor50'))} "
            f"| {_fmt(m.get('base_smape_floor50'))} "
            f"| {severe_str} "
            f"| {_fmt(m.get('false_lift_rate'))} "
            f"| {_fmt(m.get('normal_hours_degradation'))} "
            f"| {n_str} "
            f"| {v['verdict']} |"
        )

    lines.append("")
    lines.append(
        f"| Phase2 champion (ref) | {PHASE2_CHAMPION['smape_floor50']} | "
        f"{_DASH} | {PHASE2_CHAMPION['severe']} | "
        f"{PHASE2_CHAMPION['false_lift_rate']} | "
        f"{PHASE2_CHAMPION['normal_degrad']} | {_DASH} | {_DASH} |"
    )
    lines.append("")
    lines.append(
        "**DEPLOY GO thresholds**: sMAPE <= 20.50, "
        "Severe <= 63, False lift <= 10%, Normal degrad <= 0.5"
    )
    lines.append("")
    return "\n".join(lines)


def build_verdict_section(all_metrics: dict[str, dict], title: str) -> str:
    """Build a verdict text paragraph."""
    lines = [f"#### {title}", ""]
    for combo_key, label in [
        ("A", "Phase2 canonical baseline"),
        ("B", "W2 Quantile + Phase2 correction"),
        ("C", "Phase2 base + W3 ML gate"),
        ("D", "W2 Quantile + W3 ML gate"),
    ]:
        if combo_key not in all_metrics:
            continue
        m = all_metrics[combo_key]
        v = assess_go(m)
        si = v["smape_improvement_vs_champion"]
        sei = v["severe_improvement_vs_champion"]
        lines.append(
            f"- **{label}**: sMAPE={_fmt(m.get('smape_floor50'))}, "
            f"severe={m.get('severe_underestimate', '?')}, "
            f"false_lift={_fmt(m.get('false_lift_rate'))}, "
            f"normal_degrad={_fmt(m.get('normal_hours_degradation'))} "
            f"\u2192 **{v['verdict']}** "
            f"(sMAPE \u0394={_fmt(si)}, severe \u0394={sei})"
        )

    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    # Parse CLI args
    out_dir = Path(OUT_DIR_DEFAULT)
    if "--out-dir" in sys.argv:
        idx = sys.argv.index("--out-dir")
        if idx + 1 < len(sys.argv):
            out_dir = Path(sys.argv[idx + 1])
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    print("=" * 60)
    print("  P4 Final Fusion + Correction Evaluation")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    print("\n  Loading data...")
    canonical = load_canonical(CANONICAL_PACK)
    print(f"  Canonical pack: {len(canonical)} rows, "
          f"{canonical['business_day'].nunique()} days")

    canonical_risk = load_risk(CANONICAL_RISK)
    print(f"  Canonical risk: {len(canonical_risk)} rows")

    w2 = load_w2(W2_CSV)
    print(f"  W2 predictions: {len(w2)} rows, "
          f"{w2['business_day'].nunique()} days "
          f"({w2['business_day'].min()} ~ {w2['business_day'].max()})")

    w3_risk = load_risk(W3_RISK)
    print(f"  W3 risk: {len(w3_risk)} rows "
          f"({w3_risk['business_day'].min()} ~ {w3_risk['business_day'].max()})")

    # ── Define key sets ─────────────────────────────────────────────────
    # Full-window keys (all canonical timestamps)
    full_keys = (canonical[["business_day", "hour_business"]]
                 .drop_duplicates().reset_index(drop=True))
    # Overlap keys (canonical INTERSECT W2)
    overlap_keys = (full_keys
                    .merge(w2[["business_day", "hour_business"]],
                           on=["business_day", "hour_business"], how="inner")
                    .reset_index(drop=True))

    print(f"\n  Full-window:    {len(full_keys)} timestamps "
          f"({canonical['business_day'].nunique()} days)")
    print(f"  Overlap-window: {len(overlap_keys)} timestamps "
          f"({overlap_keys['business_day'].nunique()} days)")

    # ── Run combos ──────────────────────────────────────────────────────
    all_metrics: dict[str, dict] = {}

    # --- Combo A: Phase2 baseline (full + overlap) ---
    print(f"\n  {'-' * 50}")
    print("  A — Phase2 canonical baseline")
    print(f"  {'-' * 50}")
    result_a_full = build_combo_a(canonical, full_keys)
    metrics_a_full = compute_metrics(result_a_full, label="A_full")
    verdict_a_full = assess_go(metrics_a_full)
    print(f"  Full-window:  sMAPE={metrics_a_full['smape_floor50']:.4f}, "
          f"severe={metrics_a_full['severe_underestimate']}, "
          f"false_lift={metrics_a_full['false_lift_rate']:.4f} "
          f"\u2192 {verdict_a_full['verdict']}")

    result_a_overlap = build_combo_a(canonical, overlap_keys)
    metrics_a_overlap = compute_metrics(result_a_overlap, label="A_overlap")
    verdict_a_overlap = assess_go(metrics_a_overlap)
    print(f"  Overlap-window: sMAPE={metrics_a_overlap['smape_floor50']:.4f}, "
          f"severe={metrics_a_overlap['severe_underestimate']}, "
          f"false_lift={metrics_a_overlap['false_lift_rate']:.4f} "
          f"\u2192 {verdict_a_overlap['verdict']}")

    result_a_full.to_csv(out_dir / "combo_A_full_predictions.csv", index=False)
    result_a_overlap.to_csv(out_dir / "combo_A_overlap_predictions.csv",
                            index=False)
    json.dump(metrics_a_full,
              open(out_dir / "combo_A_full_metrics.json", "w"), indent=2)
    json.dump(metrics_a_overlap,
              open(out_dir / "combo_A_overlap_metrics.json", "w"), indent=2)

    all_metrics["A"] = metrics_a_full

    # --- Combo B: W2 + Phase2 correction (overlap only) ---
    print(f"\n  {'-' * 50}")
    print("  B — W2 Quantile LightGBM + Phase2 Correction")
    print(f"  {'-' * 50}")
    result_b = build_combo_b(canonical, w2, canonical_risk, overlap_keys)
    metrics_b = compute_metrics(result_b, label="B")
    verdict_b = assess_go(metrics_b)
    print(f"  Overlap-window: sMAPE={metrics_b['smape_floor50']:.4f}, "
          f"severe={metrics_b['severe_underestimate']}, "
          f"false_lift={metrics_b['false_lift_rate']:.4f} "
          f"\u2192 {verdict_b['verdict']}")

    result_b.to_csv(out_dir / "combo_B_predictions.csv", index=False)
    json.dump(metrics_b,
              open(out_dir / "combo_B_metrics.json", "w"), indent=2)
    all_metrics["B"] = metrics_b

    # --- Combo C: Phase2 + W3 ml_gate (full + overlap) ---
    print(f"\n  {'-' * 50}")
    print("  C — Phase2 Base + W3 ML Gate")
    print(f"  {'-' * 50}")
    result_c_full = build_combo_c(canonical, w3_risk, full_keys)
    metrics_c_full = compute_metrics(result_c_full, label="C_full")
    verdict_c_full = assess_go(metrics_c_full)
    print(f"  Full-window:   sMAPE={metrics_c_full['smape_floor50']:.4f}, "
          f"severe={metrics_c_full['severe_underestimate']}, "
          f"false_lift={metrics_c_full['false_lift_rate']:.4f} "
          f"\u2192 {verdict_c_full['verdict']}")

    result_c_overlap = build_combo_c(canonical, w3_risk, overlap_keys)
    metrics_c_overlap = compute_metrics(result_c_overlap, label="C_overlap")
    verdict_c_overlap = assess_go(metrics_c_overlap)
    print(f"  Overlap-window: sMAPE={metrics_c_overlap['smape_floor50']:.4f}, "
          f"severe={metrics_c_overlap['severe_underestimate']}, "
          f"false_lift={metrics_c_overlap['false_lift_rate']:.4f} "
          f"\u2192 {verdict_c_overlap['verdict']}")

    result_c_full.to_csv(out_dir / "combo_C_full_predictions.csv", index=False)
    result_c_overlap.to_csv(out_dir / "combo_C_overlap_predictions.csv",
                            index=False)
    json.dump(metrics_c_full,
              open(out_dir / "combo_C_full_metrics.json", "w"), indent=2)
    json.dump(metrics_c_overlap,
              open(out_dir / "combo_C_overlap_metrics.json", "w"), indent=2)
    all_metrics["C"] = metrics_c_full

    # --- Combo D: W2 + W3 ml_gate (overlap only) ---
    print(f"\n  {'-' * 50}")
    print("  D — W2 Quantile LightGBM + W3 ML Gate")
    print(f"  {'-' * 50}")
    result_d = build_combo_d(canonical, w2, w3_risk, overlap_keys)
    metrics_d = compute_metrics(result_d, label="D")
    verdict_d = assess_go(metrics_d)
    print(f"  Overlap-window: sMAPE={metrics_d['smape_floor50']:.4f}, "
          f"severe={metrics_d['severe_underestimate']}, "
          f"false_lift={metrics_d['false_lift_rate']:.4f} "
          f"\u2192 {verdict_d['verdict']}")

    result_d.to_csv(out_dir / "combo_D_predictions.csv", index=False)
    json.dump(metrics_d,
              open(out_dir / "combo_D_metrics.json", "w"), indent=2)
    all_metrics["D"] = metrics_d

    # ── Overlap-window results map ──────────────────────────────────────
    overlap_metrics = {
        "A": metrics_a_overlap,
        "B": metrics_b,
        "C": metrics_c_overlap,
        "D": metrics_d,
    }

    # ── Build comparison tables ─────────────────────────────────────────
    full_table = build_comparison_table(
        {"A": metrics_a_full, "C": metrics_c_full},
        "Full-Window Comparison (2025-11-01 ~ 2026-02-28, 120 days)",
    )
    overlap_table = build_comparison_table(
        overlap_metrics,
        "Overlap-Window Comparison (2025-11-01 ~ 2026-01-01, 62 days)",
    )

    print(f"\n{'=' * 60}")
    print("  COMPARISON")
    print(f"{'=' * 60}")
    print(f"\n{full_table}\n")
    print(f"\n{overlap_table}\n")

    # ── Verdict paragraphs ──────────────────────────────────────────────
    full_verdict = build_verdict_section(
        {"A": metrics_a_full, "C": metrics_c_full},
        "Full-Window Verdict",
    )
    overlap_verdict = build_verdict_section(
        overlap_metrics,
        "Overlap-Window Verdict",
    )

    print(full_verdict)
    print(overlap_verdict)

    # ── Write report ────────────────────────────────────────────────────
    report = (
        "# P4 Final Fusion + Correction Report\n\n"
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        "## Evaluation Structure\n\n"
        "- **A \u2014 Phase2 canonical baseline**: "
        "`final_pred_reference` from canonical pack (sanity check)\n"
        "- **B \u2014 W2 + Phase2 corr**: "
        "Replace lightgbm with W2 quantile, recompute anchor_90, "
        "run medium correction with canonical risk\n"
        "- **C \u2014 Phase2 + W3 gate**: "
        "Keep canonical base, run medium correction with W3 ML gate risk\n"
        "- **D \u2014 W2 + W3 gate**: "
        "Replace lightgbm with W2, recompute anchor_90, "
        "run medium correction with W3 ML gate risk\n\n"
        "**Full-window** uses all 120 canonical days. "
        "**Overlap-window** uses the 62 days where W2 has predictions.\n"
        "DEPLOY GO is assessed on full-window only. "
        "Overlap-window determines if W2 (B) or W3 gate (C/D) adds value.\n\n"
        "### sMAPE Formula\n\n"
        "Canonical floor50 sMAPE: `max(|x|, 50)` on both y_true and y_pred "
        "in denominator, using floored values in numerator. "
        "Matches `canonical_metrics_baseline.json`.\n\n"
        "---\n\n"
        f"{full_table}\n\n{full_verdict}\n\n"
        f"---\n\n"
        f"{overlap_table}\n\n{overlap_verdict}\n\n"
        "---\n\n"
        "## Comparison Summary\n\n"
        "| Aspect | Full-Window | Overlap-Window |\n"
        "|--------|:-----------:|:--------------:|\n"
        "| A sMAPE | {} | {} |\n"
        "| A severe | {} | {} |\n"
        "| C sMAPE | {} | {} |\n"
        "| C severe | {} | {} |\n"
        "| B sMAPE (overlap) | \u2014 | {} |\n"
        "| B severe (overlap) | \u2014 | {} |\n"
        "| D sMAPE (overlap) | \u2014 | {} |\n"
        "| D severe (overlap) | \u2014 | {} |\n"
        "\n"
        "## Conclusion\n\n"
    )
    # Fill placeholders
    report = report.format(
        _fmt(metrics_a_full.get("smape_floor50")),
        _fmt(metrics_a_overlap.get("smape_floor50")),
        metrics_a_full.get("severe_underestimate", "?"),
        metrics_a_overlap.get("severe_underestimate", "?"),
        _fmt(metrics_c_full.get("smape_floor50")),
        _fmt(metrics_c_overlap.get("smape_floor50")),
        metrics_c_full.get("severe_underestimate", "?"),
        metrics_c_overlap.get("severe_underestimate", "?"),
        _fmt(metrics_b.get("smape_floor50")),
        metrics_b.get("severe_underestimate", "?"),
        _fmt(metrics_d.get("smape_floor50")),
        metrics_d.get("severe_underestimate", "?"),
    )

    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"  Report: {out_dir / 'report.md'}")

    # ── Summary JSON ────────────────────────────────────────────────────
    summary = {
        "script": "scripts/evaluate_p4_final_fusion_correction.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase2_champion_ref": PHASE2_CHAMPION,
        "deploy_go_thresholds": DEPLOY_GO,
        "smape_formula": "floor50 "
                         "(max(|x|, 50) on both, numerator uses floored values)",
        "windows": {
            "full": {
                "label": "2025-11-01 ~ 2026-02-28",
                "n_timestamps": len(full_keys),
                "combos": {
                    "A_phase2_baseline": metrics_a_full,
                    "C_phase2_plus_w3_gate": metrics_c_full,
                },
            },
            "overlap": {
                "label": "W2 coverage (62 days within canonical range)",
                "n_timestamps": len(overlap_keys),
                "combos": {
                    "A_phase2_baseline": metrics_a_overlap,
                    "B_w2_plus_phase2_corr": metrics_b,
                    "C_phase2_plus_w3_gate": metrics_c_overlap,
                    "D_w2_plus_w3_gate": metrics_d,
                },
            },
        },
        "verdicts": {
            "full_window": {
                "A": verdict_a_full,
                "C": verdict_c_full,
            },
            "overlap_window": {
                "A": verdict_a_overlap,
                "B": verdict_b,
                "C": verdict_c_overlap,
                "D": verdict_d,
            },
        },
        "total_runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(summary,
              open(out_dir / "comparison_summary.json", "w"),
              indent=2, ensure_ascii=False)
    print(f"  Summary: {out_dir / 'comparison_summary.json'}")
    print(f"  Runtime: {time.time() - t0:.0f}s")
    print("Done.")


if __name__ == "__main__":
    main()
