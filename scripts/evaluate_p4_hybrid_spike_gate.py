#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p4_hybrid_spike_gate.py — Hybrid ML + Rule spike gate evaluation.

Problem:
  Old risk model probabilities are inflated → false_lift cliff.
  RF gate (P3.3) is cleaner but recall too low.

Solution:
  Three gates evaluated and compared:
  1. ml_gate       — RandomForest trained on prediction-time features
  2. rule_gate     — Heuristic rule conditions on safe features
  3. hybrid_gate   — 0.6 * ml_prob + 0.4 * rule_score (weighted combo)

Each gate replaces high_spike_prob in risk_predictions and feeds into
the standard correction pipeline (residual_lift → guardrail).

CLI:
    python scripts/evaluate_p4_hybrid_spike_gate.py \
        --prediction-pack <path/prediction_pack.csv> \
        --risk-predictions <path/risk_predictions.csv> \
        --profile-config config/p0_spike_correction_profiles.yaml \
        --out-dir reports/local/p4_hybrid_gate

Output:
    <out-dir>/ml_gate/{medium,conservative}/correction_result.csv
    <out-dir>/rule_gate/{medium,conservative}/correction_result.csv
    <out-dir>/hybrid_gate/{medium,conservative}/correction_result.csv
    <out-dir>/summary.json
    <out-dir>/tradeoff_curve.json
    <out-dir>/gate_features.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# ── Project setup ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

warnings.filterwarnings("ignore")

from extreme.realtime_high_spike.apply_correction import (
    CorrectionMode,
    CorrectionProfile,
    get_profile,
    run_correction,
    write_correction_manifest,
)

# ── Constants ──────────────────────────────────────────────────────
TARGET_SMAPE = 20.50
TARGET_SEVERE = 63
TARGET_FALSE_LIFT = 0.10


# ── Metrics ────────────────────────────────────────────────────────

def compute_smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Compute sMAPE floor50."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    s = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    return float(np.minimum(s, 50.0).mean())


def dedup_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to one row per (business_day, hour_business).

    Multicandidate packs have 4 model rows per timestamp — dedup to
    avoid inflating count-based metrics (severe, lift_applied).
    """
    ts_key = "business_day" if "business_day" in df.columns else "ds_date"
    hb_key = "hour_business" if "hour_business" in df.columns else "hour"
    if ts_key in df.columns and hb_key in df.columns:
        return df.drop_duplicates(subset=[ts_key, hb_key]).copy()
    return df.copy()


def compute_metrics(df_raw: pd.DataFrame) -> dict[str, Any]:
    """Compute all evaluation metrics, deduped to timestamp level."""
    df = dedup_timestamp(df_raw)
    m: dict[str, Any] = {}

    # Overall
    m["smape_floor50"] = round(compute_smape(df["y_true"], df["final_pred"]), 4)
    m["base_smape_floor50"] = round(compute_smape(df["y_true"], df["base_fused_pred"]), 4)

    # 9_16 period
    p9 = df[df["period"] == "9_16"]
    m["9_16_smape_floor50"] = round(compute_smape(p9["y_true"], p9["final_pred"]), 4) if len(p9) > 0 else None
    m["9_16_base_smape_floor50"] = round(compute_smape(p9["y_true"], p9["base_fused_pred"]), 4) if len(p9) > 0 else None

    # Severe underestimates
    m["severe_underestimate_count"] = int((df["y_true"] - df["final_pred"] > 200).sum())
    m["severe_underestimate_base_count"] = int((df["y_true"] - df["base_fused_pred"] > 200).sum())

    # High spike metrics
    if "high_spike_flag" in df.columns:
        spike = df[df["high_spike_flag"] == 1]
        if len(spike) > 0:
            m["high_spike_mae"] = round(float(np.mean(np.abs(spike["y_true"] - spike["final_pred"]))), 4)
            m["high_spike_base_mae"] = round(float(np.mean(np.abs(spike["y_true"] - spike["base_fused_pred"]))), 4)

    # False lift rate (non-spike hours with lift / total non-spike)
    non_spike = df[df["high_spike_flag"] == 0] if "high_spike_flag" in df.columns else df
    lifted_non_spike = non_spike[
        (non_spike["final_pred"] > non_spike["base_fused_pred"])
        & (non_spike["lift_applied"] > 0)
    ]
    m["false_lift_rate"] = round(float(len(lifted_non_spike)) / max(len(non_spike), 1), 4)

    # Recall / Precision among high-spike hours
    if "high_spike_flag" in df.columns:
        spike_all = df[df["high_spike_flag"] == 1]
        total_spike = len(spike_all)
        spike_lifted = len(spike_all[spike_all["lift_applied"] > 0])
        m["high_spike_recall"] = round(spike_lifted / max(total_spike, 1), 4)
        all_lifted = len(df[df["lift_applied"] > 0])
        m["high_spike_precision"] = round(spike_lifted / max(all_lifted, 1), 4)

    # Reason-code counts
    m["total_hours"] = len(df)
    m["lift_applied_count"] = int((df["lift_applied"] > 0).sum())
    m["lift_capped_count"] = int((df["reason_code"] == "GUARDRAIL_CLIPPED").sum())
    for code, key in [("NO_CORRECTION_LOW_PROB", "lift_rejected_low_prob"),
                      ("NO_CORRECTION_NEGATIVE_BASE", "lift_rejected_negative_base"),
                      ("NO_CORRECTION_NORMAL_HOUR", "lift_rejected_normal_hour")]:
        m[key] = int((df["reason_code"] == code).sum())

    return m


# ── Feature engineering ────────────────────────────────────────────

def build_timestamp_features(pp_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-timestamp features from multicandidate prediction pack.

    The pack has 4 rows per (business_day, hour_business) — one per model.
    Collapse to 1 row by computing prediction_spread and carrying base fields.
    """
    cols_base = ["business_day", "hour_business", "timestamp", "period",
                 "base_fused_pred", "y_true", "residual",
                 "high_spike_flag", "severe_underestimate_flag"]
    available = [c for c in cols_base if c in pp_df.columns]

    grouped = pp_df.groupby(["business_day", "hour_business"], sort=False)
    rows: list[dict[str, Any]] = []

    for (day, hour), group in grouped:
        row: dict[str, Any] = {
            "business_day": day,
            "hour_business": hour,
        }
        first = group.iloc[0]
        for c in available:
            row[c] = first[c]

        y_preds = group["y_pred"].values
        row["prediction_spread"] = float(np.std(y_preds)) if len(y_preds) > 0 else 0.0
        row["model_disagreement"] = float(np.max(y_preds) - np.min(y_preds)) if len(y_preds) > 1 else 0.0
        row["n_models"] = len(y_preds)

        rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Period dummies
    df["is_9_16"] = (df["period"] == "9_16").astype(int)
    df["is_17_24"] = (df["period"] == "17_24").astype(int)

    return df


def add_rolling_features(df: pd.DataFrame, window_days: int = 7) -> pd.DataFrame:
    """Add rolling-window features using only past data.

    For each hour_business, rolling stats over past N days using only
    historically available data (shifted by 1 to avoid look-ahead).
    """
    df = df.copy()
    window = max(window_days, 1)

    df["recent_severe_rate_by_hour"] = 0.0
    df["recent_mean_residual_by_hour"] = 0.0

    for hour in range(1, 25):
        mask = df["hour_business"] == hour
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        sub = df.loc[idx].copy()

        severe_roll = (
            sub["severe_underestimate_flag"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )
        residual_roll = (
            sub["residual"]
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

        df.loc[idx, "recent_severe_rate_by_hour"] = severe_roll.fillna(0.0).values
        df.loc[idx, "recent_mean_residual_by_hour"] = residual_roll.fillna(0.0).values

    return df


def prepare_gate_data(
    prediction_pack_path: str | Path,
    risk_predictions_path: str | Path,
) -> pd.DataFrame:
    """Load prediction pack + risk predictions and build full feature set."""
    pp = pd.read_csv(prediction_pack_path)
    rp = pd.read_csv(risk_predictions_path)

    ts = build_timestamp_features(pp)
    ts = add_rolling_features(ts, window_days=7)

    # Merge original risk scores for comparison
    rp["business_day"] = rp["business_day"].astype(str)
    ts["business_day"] = ts["business_day"].astype(str)
    ts = ts.merge(
        rp[["business_day", "hour_business", "spike_risk_score", "high_spike_prob"]],
        on=["business_day", "hour_business"],
        how="left",
    )

    return ts


# ── ML Gate — RandomForest ─────────────────────────────────────────

def train_ml_gate(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    seed: int = 42,
) -> Any:
    """Train RandomForest classifier for spike prediction."""
    from sklearn.ensemble import RandomForestClassifier

    X = train_df[feature_cols].fillna(0.0)
    y = train_df["high_spike_flag"].values

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    model.fit(X, y)
    return model


def apply_ml_gate(model: Any, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Apply trained ML gate to get spike probabilities."""
    X = df[feature_cols].fillna(0.0)
    return model.predict_proba(X)[:, 1]


def tune_ml_threshold(
    model: Any,
    val_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[float, dict[str, float]]:
    """Tune ML gate threshold for max F2 score (recall-weighted)."""
    probs = apply_ml_gate(model, val_df, feature_cols)
    y_true = val_df["high_spike_flag"].values

    best_f2 = -1.0
    best_thresh = 0.5
    best_metrics: dict[str, float] = {}

    for thresh in np.arange(0.05, 0.95, 0.025):
        preds = (probs >= thresh).astype(int)
        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f2 = (5 * precision * recall) / max((4 * precision + recall), 1e-10)

        if f2 > best_f2:
            best_f2 = f2
            best_thresh = thresh
            best_metrics = {
                "threshold": round(thresh, 3),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f2": round(f2, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            }

    return best_thresh, best_metrics


# ── Rule Gate ──────────────────────────────────────────────────────

def compute_rule_scores(df: pd.DataFrame) -> np.ndarray:
    """Compute rule-based spike score for each row.

    Components (all normalized 0-1, weighted sum → 0-1 output):
      - base_fused_pred magnitude (30%)
      - prediction_spread (15%)
      - recent_severe_rate_by_hour (20%)
      - is_9_16 period (20%)
      - recent_mean_residual_by_hour (15%)
    """
    EPS = 1e-10

    base = df["base_fused_pred"].values
    p5, p95 = np.percentile(base, [5, 95])
    base_norm = np.clip((base - p5) / max(p95 - p5, EPS), 0, 1)

    spread = df["prediction_spread"].values
    s5, s95 = np.percentile(spread, [5, 95])
    spread_norm = np.clip((spread - s5) / max(s95 - s5, EPS), 0, 1)

    severe_rate = df["recent_severe_rate_by_hour"].values
    is_9_16 = df["is_9_16"].values

    resid = np.maximum(df["recent_mean_residual_by_hour"].values, 0)
    r5, r95 = np.percentile(resid, [5, 95])
    resid_norm = np.clip((resid - r5) / max(r95 - r5, EPS), 0, 1) if r95 > r5 else np.zeros_like(resid)

    w = [0.30, 0.15, 0.20, 0.20, 0.15]
    scores = (
        w[0] * base_norm
        + w[1] * spread_norm
        + w[2] * severe_rate
        + w[3] * is_9_16
        + w[4] * resid_norm
    )

    return np.clip(scores, 0, 1)


# ── Hybrid Gate ────────────────────────────────────────────────────

def compute_hybrid_scores(ml_probs: np.ndarray, rule_scores: np.ndarray,
                           ml_weight: float = 0.6) -> np.ndarray:
    """hybrid = ml_weight * ml_prob + (1-ml_weight) * rule_score."""
    return np.clip(ml_weight * ml_probs + (1.0 - ml_weight) * rule_scores, 0, 1)


# ── Gate evaluation runner ─────────────────────────────────────────

def run_gate_evaluation(
    gate_name: str,
    spike_prob_array: np.ndarray,
    prediction_pack_path: str | Path,
    risk_predictions_path: str | Path,
    profiles_to_run: list[str],
    profile_config_path: str,
    out_dir: Path,
) -> dict[str, Any]:
    """Run correction pipeline with a custom spike probability array.

    Creates a modified risk predictions CSV, runs correction for each profile,
    returns {profile_name: metrics}.
    """
    rp_orig = pd.read_csv(risk_predictions_path)
    rp_gate = rp_orig.copy()
    rp_gate["high_spike_prob"] = spike_prob_array

    tmp_dir = out_dir / gate_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_rp_path = tmp_dir / "risk_predictions_gate.csv"
    rp_gate.to_csv(tmp_rp_path, index=False, encoding="utf-8-sig")

    all_metrics: dict[str, Any] = {}

    for pname in profiles_to_run:
        print(f"  [{gate_name}] Running profile: {pname}")
        profile = get_profile(pname, config_path=profile_config_path, mode=CorrectionMode.NORMAL)

        profile_out = tmp_dir / pname
        profile_out.mkdir(parents=True, exist_ok=True)

        result = run_correction(
            prediction_pack_path=str(prediction_pack_path),
            risk_predictions_path=str(tmp_rp_path),
            profile=profile,
        )

        # Save
        result.to_csv(profile_out / "correction_result.csv", index=False, encoding="utf-8-sig")

        # Metrics (deduped to timestamp level)
        metrics = compute_metrics(result)
        metrics["profile_used"] = pname
        metrics["gate_name"] = gate_name
        metrics["spike_prob_threshold"] = profile.spike_prob_threshold
        metrics["max_lift_ratio"] = profile.max_lift_ratio
        metrics["max_absolute_lift"] = profile.max_absolute_lift

        write_correction_manifest(profile_out, profile, metrics=metrics)

        with open(profile_out / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        all_metrics[pname] = metrics

        print(f"    sMAPE={metrics.get('smape_floor50', '?'):<8}"
              f"  severe={metrics.get('severe_underestimate_count', '?'):<6}"
              f"  false_lift={metrics.get('false_lift_rate', '?'):<8}"
              f"  recall={metrics.get('high_spike_recall', '?'):<6}"
              f"  precision={metrics.get('high_spike_precision', '?'):<6}"
              f"  lifted={metrics.get('lift_applied_count', '?'):<6}")

    return all_metrics


# ── CLI ────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P4 Hybrid Spike Gate evaluation.",
    )
    parser.add_argument("--prediction-pack", required=True,
                        help="Prediction pack CSV (multicandidate, 4 models per timestamp)")
    parser.add_argument("--risk-predictions", required=True,
                        help="Risk predictions CSV")
    parser.add_argument("--profile-config", default="config/p0_spike_correction_profiles.yaml",
                        help="Profile config (YAML)")
    parser.add_argument("--out-dir", default="reports/local/p4_hybrid_gate",
                        help="Output directory")
    parser.add_argument("--profiles", nargs="+", default=["medium", "conservative"],
                        help="Profiles to evaluate")
    parser.add_argument("--ml-weight", type=float, default=0.6,
                        help="ML weight in hybrid combo")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for RF")
    parser.add_argument("--window-days", type=int, default=7,
                        help="Rolling window days")
    return parser.parse_args(argv)


# ── Main ───────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    pp_path = Path(args.prediction_pack)
    rp_path = Path(args.risk_predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    profiles_to_run = args.profiles

    print("=" * 60)
    print("  P4 Hybrid Spike Gate Evaluation")
    print("=" * 60)

    # ── Step 1: Prepare data ───────────────────────────────────────
    print("\n[1/5] Loading and featurizing data...")
    data = prepare_gate_data(pp_path, rp_path)
    print(f"  → {len(data)} timestamps from {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")

    # Save gate features
    data.to_csv(out_dir / "gate_features.csv", index=False, encoding="utf-8-sig")

    # ── Step 2: ML Gate ───────────────────────────────────────────
    print("\n[2/5] Training ML gate (RandomForest)...")

    ml_feature_cols = [
        "base_fused_pred", "prediction_spread", "model_disagreement",
        "hour_business", "is_9_16", "is_17_24",
        "recent_severe_rate_by_hour", "recent_mean_residual_by_hour",
    ]

    # Chronological split: 60% train, 15% val (threshold tuning), 25% test
    n = len(data)
    train_end = int(n * 0.60)
    val_end = int(n * 0.75)

    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()

    print(f"  Train: {len(train_df)} ({train_df['timestamp'].min().date()} to {train_df['timestamp'].max().date()})")
    spike_pct = train_df["high_spike_flag"].mean() * 100
    print(f"  Spike label rate: {spike_pct:.2f}%")

    ml_model = train_ml_gate(train_df, ml_feature_cols, seed=args.seed)
    print(f"  RF trained: {ml_model.n_estimators} trees, {len(ml_feature_cols)} features")

    # Feature importance
    fi = sorted(zip(ml_feature_cols, ml_model.feature_importances_),
                key=lambda x: -x[1])
    print("  Top features:")
    for name, imp in fi[:5]:
        print(f"    {name}: {imp:.4f}")

    # ── Step 3: Tune threshold ────────────────────────────────────
    print("\n[3/5] Tuning ML gate threshold (F2-optimal)...")
    best_thresh, best_val_metrics = tune_ml_threshold(ml_model, val_df, ml_feature_cols)
    print(f"  Best threshold: {best_thresh:.3f}")
    print(f"    Val recall={best_val_metrics['recall']:.4f}, precision={best_val_metrics['precision']:.4f}, F2={best_val_metrics['f2']:.4f}")

    # Apply ML to full dataset
    ml_probs = apply_ml_gate(ml_model, data, ml_feature_cols)
    data["gate_ml_prob"] = ml_probs

    # ── Step 4: Rule Gate ─────────────────────────────────────────
    print("\n[4/5] Computing rule gate scores...")
    rule_scores = compute_rule_scores(data)
    data["gate_rule_score"] = rule_scores

    rule_gate_on = rule_scores >= 0.50
    rule_recall = rule_gate_on[data["high_spike_flag"] == 1].mean() if data["high_spike_flag"].sum() > 0 else 0
    print(f"  Rule gate active (thresh=0.50): {rule_gate_on.sum()}/{len(data)} ({rule_gate_on.mean()*100:.1f}%)")
    print(f"  Rule recall (spike): {rule_recall:.4f}")

    # ── Hybrid Gate ──────────────────────────────────────────────
    print("\n[5/5] Computing hybrid gate scores...")
    ml_weight = args.ml_weight
    hybrid_scores = compute_hybrid_scores(ml_probs, rule_scores, ml_weight=ml_weight)
    data["gate_hybrid_prob"] = hybrid_scores

    hybrid_gate_on = hybrid_scores >= best_thresh
    hybrid_recall = hybrid_gate_on[data["high_spike_flag"] == 1].mean() if data["high_spike_flag"].sum() > 0 else 0
    print(f"  Hybrid gate active: {hybrid_gate_on.sum()}/{len(data)} ({hybrid_gate_on.mean()*100:.1f}%)")
    print(f"  Hybrid recall (spike): {hybrid_recall:.4f}")

    # Save features with all gate probabilities
    data.to_csv(out_dir / "gate_features.csv", index=False, encoding="utf-8-sig")

    # ── Run correction for each gate ──────────────────────────────
    print("\n" + "=" * 60)
    print("  Running correction pipeline for each gate...")
    print("=" * 60)

    gates = {
        "ml_gate": data["gate_ml_prob"].values,
        "rule_gate": data["gate_rule_score"].values,
        "hybrid_gate": data["gate_hybrid_prob"].values,
        "baseline_old_risk": data["high_spike_prob"].fillna(0.0).values,
    }

    all_results: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []

    for gate_name, spike_probs in gates.items():
        print(f"\n--- Gate: {gate_name} ---")
        gate_metrics = run_gate_evaluation(
            gate_name=gate_name,
            spike_prob_array=spike_probs,
            prediction_pack_path=pp_path,
            risk_predictions_path=rp_path,
            profiles_to_run=profiles_to_run,
            profile_config_path=args.profile_config,
            out_dir=out_dir,
        )
        all_results[gate_name] = gate_metrics

        for pname, metrics in gate_metrics.items():
            summary_rows.append({
                "gate": gate_name,
                "profile": pname,
                "smape_floor50": metrics.get("smape_floor50"),
                "base_smape_floor50": metrics.get("base_smape_floor50"),
                "severe_underestimate_count": metrics.get("severe_underestimate_count"),
                "severe_underestimate_base_count": metrics.get("severe_underestimate_base_count"),
                "false_lift_rate": metrics.get("false_lift_rate"),
                "high_spike_recall": metrics.get("high_spike_recall"),
                "high_spike_precision": metrics.get("high_spike_precision"),
                "lift_applied_count": metrics.get("lift_applied_count"),
            })

    # ── Tradeoff curve ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Computing recall/precision tradeoff curve...")
    print("=" * 60)

    tradeoff_points: list[dict[str, Any]] = []
    spike_y = data["high_spike_flag"].values
    n_total = len(data)
    n_neg = int((spike_y == 0).sum())

    for thresh in np.arange(0.05, 0.95, 0.025):
        for gate_name, probs in [
            ("ml_gate", gates["ml_gate"]),
            ("rule_gate", gates["rule_gate"]),
            ("hybrid_gate", gates["hybrid_gate"]),
        ]:
            gate_on = probs >= thresh
            tp = int(np.sum((gate_on & (spike_y == 1))))
            fp = int(np.sum((gate_on & (spike_y == 0))))
            fn = int(np.sum((~gate_on & (spike_y == 1))))

            recall = tp / max(tp + fn, 1)
            precision = tp / max(tp + fp, 1)
            fpr = fp / max(n_neg, 1)

            tradeoff_points.append({
                "gate": gate_name,
                "threshold": round(thresh, 3),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "false_positive_rate": round(fpr, 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
            })

    tradeoff_path = out_dir / "tradeoff_curve.json"
    with open(tradeoff_path, "w", encoding="utf-8") as f:
        json.dump(tradeoff_points, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    all_results["_summary"] = {
        "best_ml_threshold": best_thresh,
        "ml_val_metrics": best_val_metrics,
        "ml_feature_cols": ml_feature_cols,
        "params": {
            "ml_weight": ml_weight,
            "seed": args.seed,
            "window_days": args.window_days,
            "profiles": profiles_to_run,
        },
    }

    summary_json = out_dir / "summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ── Final report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"\n{'Gate':<18} {'Profile':<15} {'sMAPE':<8} {'Base_sMAPE':<11} "
          f"{'Sev':<6} {'Base_Sev':<9} {'FL':<6} {'Recall':<8} {'Prec':<8} {'Lifted':<8}")
    print("-" * 105)
    for _, row in summary_df.iterrows():
        print(f"{row['gate']:<18} {row['profile']:<15} "
              f"{row['smape_floor50']:<8} {row['base_smape_floor50']:<11} "
              f"{row['severe_underestimate_count']:<6} {row['severe_underestimate_base_count']:<9} "
              f"{row['false_lift_rate']:<6} {row['high_spike_recall']:<8} "
              f"{row['high_spike_precision']:<8} {row['lift_applied_count']:<8}")

    print(f"\nTargets: sMAPE <= {TARGET_SMAPE}, severe <= {TARGET_SEVERE}, false_lift <= {TARGET_FALSE_LIFT}")
    for _, row in summary_df.iterrows():
        smape_ok = str(row["smape_floor50"]) != "None" and float(row["smape_floor50"]) <= TARGET_SMAPE
        severe_ok = str(row["severe_underestimate_count"]) != "None" and int(row["severe_underestimate_count"]) <= TARGET_SEVERE
        fl_ok = str(row["false_lift_rate"]) != "None" and float(row["false_lift_rate"]) <= TARGET_FALSE_LIFT
        all_ok = smape_ok and severe_ok and fl_ok
        status = "✅ PASS" if all_ok else "❌ FAIL"
        print(f"  {row['gate']:<18} {row['profile']:<15} → {status}")

    print(f"\nTradeoff curve: {tradeoff_path}")
    print(f"Full results: {out_dir}")
    print("\nDone.")


if __name__ == "__main__":
    main()
