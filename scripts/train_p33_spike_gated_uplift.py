#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_p33_spike_gated_uplift.py — Train spike-gated uplift model (P3.3).

Core idea:
  Train a leakage-safe classifier that predicts P(severe_underestimate)
  for each timestamp. Only apply correction lift on timestamps where
  the gate probability exceeds threshold. All features must be available
  at prediction time (D-1 or earlier).

Algorithm:
  1. Build timestamp-level features from model predictions + exogenous data
  2. Rolling time-split: fit on [D-30, D-1], apply on D
  3. Target: severe_underestimate_flag (y_true - base_fused_pred > 200)
  4. Lift = quantile of positive residuals from past similar high-risk hours

Features (all prediction-time safe):
  - hour_business, period (categorical)
  - base_pred (= base_fused_pred from LightGBM-anchored fusion)
  - prediction_spread (max-min across 4 models)
  - model_disagreement (std across 4 models)
  - dayahead_proxy - lightgbm diff
  - naive_lag7 - lightgbm diff
  - naive_lag1 - lightgbm diff
  - 10 exogenous forecast columns from xlsx
  - recent_30d_severe_rate_by_hour
  - recent_30d_severe_rate_by_period

Usage:
    python scripts/train_p33_spike_gated_uplift.py
        --pack-path reports/local/p0_full_run/prediction_pack_multicandidate/...
        --raw-data data/shandong_pmos_hourly.xlsx
        --out-dir reports/local/p33_spike_gated_uplift
        --model-type rf
        --threshold 0.50

Output:
    <out-dir>/
      - gate_predictions.csv           — gate probability per timestamp
      - gate_risk_predictions.csv      — risk predictions usable by correction pipeline
      - feature_importance.csv         — RF feature importance (mean across folds)
      - training_summary.json          — training config + stats
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

BASELINE_MODELS = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
ALL_MODELS = BASELINE_MODELS + ["lightgbm"]

FORECAST_COLS = [
    "地方电厂总加预测值",
    "联络线受电负荷预测值",
    "风电总加预测值",
    "光伏总加预测值",
    "核电总加预测值",
    "自备机组总加预测值",
    "试验机组总加预测值",
    "直调负荷预测值",
    "竞价空间预测值",
    "新能源总加预测值",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train P3.3 spike-gated uplift model.",
    )
    parser.add_argument("--pack-path", default=None,
                        help="Path to multi-candidate prediction pack CSV")
    parser.add_argument("--raw-data", default="data/shandong_pmos_hourly.xlsx",
                        help="Path to raw xlsx data with exogenous features")
    parser.add_argument("--out-dir", default="reports/local/p33_spike_gated_uplift",
                        help="Output directory")
    parser.add_argument("--start-date", default="2025-11-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-02-28",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--model-type", default="rf",
                        choices=["rf", "gradient_boosting", "logistic"],
                        help="Classifier type")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Gate probability threshold. If None, tuned on validation.")
    parser.add_argument("--lift-quantile", type=float, default=0.90,
                        help="Quantile of past residuals for lift amount")
    parser.add_argument("--train-warmup-days", type=int, default=60,
                        help="Days of training data required before first eval day")
    parser.add_argument("--train-window-days", type=int, default=None,
                        help="Sliding training window size. If None, uses cumulative (all prior data).")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of trees for RF/GB")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max tree depth for RF/GB")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args(argv)


def resolve_default_pack_path() -> Path:
    return (
        _PROJECT_ROOT
        / "reports/local/p0_full_run/prediction_pack_multicandidate"
        / "prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv"
    )


def load_prediction_pack(path: Path) -> pd.DataFrame:
    """Load and prepare the multi-candidate prediction pack."""
    df = pd.read_csv(path)
    df["business_day"] = df["business_day"].astype(str)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"])
    return df


def load_raw_data(path: Path) -> pd.DataFrame:
    """Load raw xlsx data with exogenous columns."""
    df = pd.read_excel(path)
    df["时刻"] = pd.to_datetime(df["时刻"])
    return df


def pivot_models(pack: pd.DataFrame) -> pd.DataFrame:
    """Pivot model predictions to one row per timestamp.

    Returns:
        DataFrame with one row per (business_day, hour_business) containing
        model prediction columns + y_true + base_fused_pred.
    """
    # Select model-specific y_pred columns
    ts_cols = ["business_day", "hour_business", "timestamp", "timestamp_dt",
               "period", "y_true", "base_fused_pred"]

    # Verify all ts_cols exist
    for c in ts_cols:
        if c not in pack.columns:
            # period might be missing if we need to infer
            if c == "period":
                continue
            raise KeyError(f"Required column '{c}' not found in prediction pack")

    # Create timestamp-level base
    ts_df = pack[ts_cols].drop_duplicates(
        subset=["business_day", "hour_business"]
    ).copy()

    # Pivot model y_pred values
    for model in ALL_MODELS:
        model_rows = pack[pack["model_name"] == model]
        if len(model_rows) == 0:
            ts_df[f"pred_{model}"] = np.nan
            continue
        model_map = model_rows.set_index(["business_day", "hour_business"])["y_pred"]
        ts_df[f"pred_{model}"] = ts_df.set_index(
            ["business_day", "hour_business"]
        ).index.map(model_map.get)

    # Ensure period exists
    if "period" not in ts_df.columns:
        ts_df["period"] = ts_df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
        )

    return ts_df


def compute_model_features(ts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute model-based features from pivoted predictions.

    Features (all prediction-time safe):
      - prediction_spread: max - min across models
      - model_disagreement: std across models
      - Dayahead-LightGBM, Lag7-LightGBM, Lag1-LightGBM diffs
    """
    df = ts_df.copy()

    # Collect available model predictions
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    pred_vals = df[pred_cols].values  # shape (n_timestamps, n_models)

    df["prediction_spread"] = np.nanmax(pred_vals, axis=1) - np.nanmin(pred_vals, axis=1)
    df["model_disagreement"] = np.nanstd(pred_vals, axis=1, ddof=1)
    df["n_models_available"] = (~np.isnan(pred_vals)).sum(axis=1)

    # Model differences
    lgbm_col = "pred_lightgbm"
    if lgbm_col in df.columns:
        for other in BASELINE_MODELS:
            other_col = f"pred_{other}"
            if other_col in df.columns:
                df[f"{other}_lightgbm_diff"] = (
                    df[other_col] - df[lgbm_col]
                )

    return df


def merge_exogenous(
    ts_df: pd.DataFrame,
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge exogenous forecast columns from raw data."""
    df = ts_df.copy()

    # The pack timestamp_dt is in natural time. Merge on 时刻.
    raw_ts = raw_data[["时刻"] + FORECAST_COLS].copy()
    raw_ts["时刻"] = pd.to_datetime(raw_ts["时刻"])

    df = df.merge(
        raw_ts,
        left_on="timestamp_dt",
        right_on="时刻",
        how="left",
    )
    df.drop(columns=["时刻"], inplace=True)
    return df


def compute_rolling_features(
    ts_df: pd.DataFrame,
    eval_date: pd.Timestamp,
    window_days: int = 30,
) -> pd.DataFrame:
    """Compute rolling historical features for a specific evaluation date.

    For timestamps on eval_date, compute features from the past window_days:
      - recent_30d_severe_rate_by_hour
      - recent_30d_severe_rate_by_period

    Args:
        ts_df: Full timestamp-level DataFrame with severe_underestimate_flag.
        eval_date: The evaluation date (D).
        window_days: Lookback window size.

    Returns:
        DataFrame with timestamps for eval_date, rolling features added.
    """
    window_start = eval_date - timedelta(days=window_days)

    # Get historical window data
    hist = ts_df[
        (ts_df["timestamp_dt"] >= window_start)
        & (ts_df["timestamp_dt"] < eval_date)
    ].copy()

    if len(hist) == 0:
        # No history — return zeros
        eval_df = ts_df[
            (ts_df["business_day"] == eval_date.strftime("%Y-%m-%d"))
        ].copy()
        eval_df["recent_severe_rate_by_hour"] = 0.0
        eval_df["recent_severe_rate_by_period"] = 0.0
        eval_df["recent_mean_residual_by_hour"] = 0.0
        eval_df["recent_mean_residual_by_period"] = 0.0
        return eval_df

    # Compute severe rate by hour
    hist["severe_flag"] = (hist["y_true"] - hist["base_fused_pred"] > 200).astype(int)
    hour_stats = hist.groupby("hour_business").agg(
        severe_rate=("severe_flag", "mean"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    hour_stats.rename(columns={
        "severe_rate": "recent_severe_rate_by_hour",
        "mean_residual": "recent_mean_residual_by_hour",
    }, inplace=True)

    # Compute severe rate by period
    period_stats = hist.groupby("period").agg(
        severe_rate=("severe_flag", "mean"),
        mean_residual=("residual", "mean"),
    ).reset_index()
    period_stats.rename(columns={
        "severe_rate": "recent_severe_rate_by_period",
        "mean_residual": "recent_mean_residual_by_period",
    }, inplace=True)

    # Merge onto eval timestamps
    eval_df = ts_df[
        (ts_df["business_day"] == eval_date.strftime("%Y-%m-%d"))
    ].copy()

    eval_df = eval_df.merge(hour_stats, on="hour_business", how="left")
    eval_df = eval_df.merge(period_stats, on="period", how="left")
    eval_df["recent_severe_rate_by_hour"] = eval_df["recent_severe_rate_by_hour"].fillna(0.0)
    eval_df["recent_severe_rate_by_period"] = eval_df["recent_severe_rate_by_period"].fillna(0.0)
    eval_df["recent_mean_residual_by_hour"] = eval_df["recent_mean_residual_by_hour"].fillna(0.0)
    eval_df["recent_mean_residual_by_period"] = eval_df["recent_mean_residual_by_period"].fillna(0.0)

    return eval_df


def get_feature_columns(ts_df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns for the classifier."""
    base_features = [
        "hour_business",
        "base_fused_pred",
        "prediction_spread",
        "model_disagreement",
        "n_models_available",
    ]

    # Model difference features
    for other in BASELINE_MODELS:
        col = f"{other}_lightgbm_diff"
        if col in ts_df.columns:
            base_features.append(col)

    # Exogenous forecast columns
    for col in FORECAST_COLS:
        if col in ts_df.columns:
            base_features.append(col)

    # Rolling features
    rolling_features = [
        "recent_severe_rate_by_hour",
        "recent_severe_rate_by_period",
        "recent_mean_residual_by_hour",
        "recent_mean_residual_by_period",
    ]

    return base_features + rolling_features


def encode_period(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode period column."""
    return pd.get_dummies(df, columns=["period"], prefix="period")


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str,
    seed: int,
    n_estimators: int = 200,
    max_depth: int = 6,
) -> Any:
    """Train a classifier for severe underestimate prediction."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=min(n_estimators, 100),
            max_depth=max_depth,
            min_samples_leaf=10,
            random_state=seed,
        )
    elif model_type == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            class_weight="balanced",
            random_state=seed,
            max_iter=1000,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


def compute_lift_amount(
    ts_eval: pd.DataFrame,
    hist_df: pd.DataFrame,
    gate_prob_col: str,
    threshold: float,
    lift_quantile: float = 0.90,
    max_lift_ratio: float = 0.35,
    max_absolute_lift: float = 350.0,
) -> pd.DataFrame:
    """Compute lift amount for timestamps where gate_prob > threshold.

    Lift = quantile of positive residuals from past high-risk hours
    in the same period, capped by max_lift_ratio and max_absolute_lift.
    """
    df = ts_eval.copy()

    # Ensure period column exists
    if "period" not in df.columns:
        df["period"] = df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
        )

    # Identify high-gate timestamps
    df["gate_active"] = (df[gate_prob_col] > threshold).astype(int)

    if len(hist_df) == 0:
        df["p33_lift"] = 0.0
        df["p33_lift_source"] = "no_history"
        return df

    # Compute positive residuals from history, grouped by period
    hist = hist_df.copy()
    if "period" not in hist.columns:
        # Reconstruct period from one-hot columns or infer from hour_business
        hist["period"] = hist["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
        )
    hist["positive_residual"] = np.maximum(0, hist["y_true"] - hist["base_fused_pred"])

    # Per-period lift candidate: quantile of positive residuals
    period_lifts: dict[str, float] = {}
    for period_name in ["1_8", "9_16", "17_24"]:
        mask = hist["period"] == period_name
        pr = hist.loc[mask, "positive_residual"].dropna().values
        if len(pr) > 0:
            lift_val = float(np.quantile(pr, lift_quantile))
            period_lifts[period_name] = max(0.0, lift_val)
        else:
            period_lifts[period_name] = 0.0

    # Apply lift where gate is active
    lifts: list[float] = []
    sources: list[str] = []
    for _, row in df.iterrows():
        if row["gate_active"] == 0:
            lifts.append(0.0)
            sources.append("gate_inactive")
            continue

        period = row["period"]
        raw_lift = period_lifts.get(period, 0.0)

        # Cap by ratio and absolute
        base_pred = row["base_fused_pred"]
        ratio_cap = base_pred * max_lift_ratio
        capped_lift = min(raw_lift, ratio_cap, max_absolute_lift)
        capped_lift = max(0.0, capped_lift)

        if capped_lift <= 0:
            lifts.append(0.0)
            sources.append("lift_zero_after_cap")
        else:
            lifts.append(round(capped_lift, 2))
            sources.append(f"period_{period}_p{int(lift_quantile*100)}")

    df["p33_lift"] = lifts
    df["p33_lift_source"] = sources
    df["p33_lift_period_candidates"] = str(period_lifts)

    return df


def tune_threshold(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Tune probability threshold to maximise F2 (recall-weighted)."""
    from sklearn.metrics import precision_recall_curve, fbeta_score

    proba = model.predict_proba(X_val)[:, 1]

    # Try thresholds from 0.1 to 0.9
    best_f2 = 0.0
    best_threshold = 0.5
    results: dict[str, float] = {}

    for t in np.arange(0.05, 0.95, 0.05):
        preds = (proba >= t).astype(int)
        f2 = fbeta_score(y_val, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    results["best_threshold"] = best_threshold
    results["best_f2"] = round(best_f2, 4)
    return best_threshold, results


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = args.seed
    window_days = args.train_warmup_days
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)

    print("=" * 60)
    print("  P3.3 — Train Spike-Gated Uplift Model")
    print("=" * 60)
    print(f"  Model type:     {args.model_type}")
    print(f"  Window:         {start_date.date()} to {end_date.date()}")
    print(f"  Train warm-up:  {window_days} days")
    print(f"  Lift quantile:  {args.lift_quantile}")
    print()

    # ── 1. Load data ──────────────────────────────────────────────────
    pack_path = Path(args.pack_path) if args.pack_path else resolve_default_pack_path()
    if not pack_path.exists():
        print(f"  [ERR] Pack not found: {pack_path}")
        sys.exit(1)
    print(f"  [OK] Prediction pack: {pack_path.name}")

    raw_path = Path(args.raw_data)
    if not raw_path.exists():
        print(f"  [ERR] Raw data not found: {raw_path}")
        sys.exit(1)

    pack = load_prediction_pack(pack_path)
    raw_data = load_raw_data(raw_path)
    print(f"  [OK] Pack: {len(pack)} rows | Raw data: {len(raw_data)} rows")

    # ── 2. Build timestamp-level features ─────────────────────────────
    print("\n  Building timestamp-level features...")
    ts_df = pivot_models(pack)
    print(f"  -> {len(ts_df)} timestamps")

    # Target
    ts_df["severe_underestimate_flag"] = (
        ts_df["y_true"] - ts_df["base_fused_pred"] > 200
    ).astype(int)
    n_severe = ts_df["severe_underestimate_flag"].sum()
    severe_rate = n_severe / len(ts_df) * 100
    print(f"  -> Severe underestimates: {n_severe} / {len(ts_df)} ({severe_rate:.1f}%)")

    # Compute residual
    ts_df["residual"] = ts_df["y_true"] - ts_df["base_fused_pred"]

    # Model-based features
    ts_df = compute_model_features(ts_df)
    print(f"  -> Model features computed")

    # Merge exogenous data
    ts_df = merge_exogenous(ts_df, raw_data)
    print(f"  -> Exogenous features merged ({len(FORECAST_COLS)} cols)")

    # Ensure period (in case it was missing)
    if "period" not in ts_df.columns:
        ts_df["period"] = ts_df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8 else ("9_16" if 9 <= h <= 16 else "17_24")
        )

    # ── 3. Rolling window training ────────────────────────────────────
    print(f"\n  Rolling time-split training (fit on [D-{window_days}, D-1], apply on D)...")

    eval_dates = sorted(ts_df["business_day"].unique())
    # First eval date needs warmup_days of prior data
    first_eval = pd.Timestamp(eval_dates[0]) + timedelta(days=window_days)
    eval_date_objs = [d for d in sorted(ts_df["timestamp_dt"].unique())
                      if d.date() >= first_eval.date() and d.date() <= end_date.date()]
    eval_day_set = sorted(set(d.date() for d in eval_date_objs))

    print(f"  Evaluation days: {len(eval_day_set)} "
          f"({eval_day_set[0]} to {eval_day_set[-1]})")

    if len(eval_day_set) == 0:
        print("  [ERR] No evaluation days after warmup period")
        sys.exit(1)

    # Store predictions
    all_gate_preds: list[dict[str, Any]] = []
    all_importances: list[dict[str, float]] = []
    daily_results: list[dict[str, Any]] = []

    # For threshold tuning: collect all val predictions
    val_probas: list[float] = []
    val_targets: list[int] = []
    # For final eval: collect train/val dates for last model
    last_trained_model = None
    last_feature_cols: list[str] = []

    for day_idx, eval_day in enumerate(eval_day_set):
        eval_dt = pd.Timestamp(eval_day)
        window_start = eval_dt - timedelta(days=window_days)

        # Training set: either sliding [D-window, D-1] or cumulative [start, D-1]
        if args.train_window_days:
            train_mask = (
                (ts_df["timestamp_dt"] >= window_start)
                & (ts_df["timestamp_dt"] < eval_dt)
            )
        else:
            train_mask = ts_df["timestamp_dt"] < eval_dt
        train_df = ts_df[train_mask].copy()

        # Evaluation set: day D
        eval_mask = ts_df["business_day"] == eval_day.strftime("%Y-%m-%d")
        eval_df = ts_df[eval_mask].copy()

        if len(train_df) < 24:  # Need at least 1 day of data
            continue

        # Compute rolling stats from training window [D-30, D-1]
        train_df["severe_flag_tmp"] = train_df["severe_underestimate_flag"]
        hour_stats = train_df.groupby("hour_business").agg(
            recent_severe_rate_by_hour=("severe_flag_tmp", "mean"),
            recent_mean_residual_by_hour=("residual", "mean"),
        ).reset_index()
        period_stats = train_df.groupby("period").agg(
            recent_severe_rate_by_period=("severe_flag_tmp", "mean"),
            recent_mean_residual_by_period=("residual", "mean"),
        ).reset_index()

        # Merge rolling features onto training data
        train_df_roll = train_df.merge(hour_stats, on="hour_business", how="left")
        train_df_roll = train_df_roll.merge(period_stats, on="period", how="left")

        # Drop temp column
        train_df_roll.drop(columns=["severe_flag_tmp"], inplace=True)

        # Merge rolling features onto evaluation data
        eval_df = eval_df.merge(hour_stats, on="hour_business", how="left")
        eval_df = eval_df.merge(period_stats, on="period", how="left")
        eval_df["recent_severe_rate_by_hour"] = eval_df["recent_severe_rate_by_hour"].fillna(0.0)
        eval_df["recent_severe_rate_by_period"] = eval_df["recent_severe_rate_by_period"].fillna(0.0)
        eval_df["recent_mean_residual_by_hour"] = eval_df["recent_mean_residual_by_hour"].fillna(0.0)
        eval_df["recent_mean_residual_by_period"] = eval_df["recent_mean_residual_by_period"].fillna(0.0)

        # Encode period
        train_df_roll = encode_period(train_df_roll)
        eval_df_enc = encode_period(eval_df)

        # Get feature columns
        feature_cols = get_feature_columns(ts_df)
        # Add one-hot period columns
        period_cols = [c for c in train_df_roll.columns
                       if c.startswith("period_")]
        feature_cols = [c for c in feature_cols if c in train_df_roll.columns]
        feature_cols = list(dict.fromkeys(feature_cols + period_cols))

        # Ensure eval has same columns
        for c in feature_cols:
            if c not in eval_df_enc.columns:
                eval_df_enc[c] = 0.0

        # Filter to available columns
        avail_train_cols = [c for c in feature_cols if c in train_df_roll.columns]
        avail_eval_cols = [c for c in feature_cols if c in eval_df_enc.columns]

        if len(avail_train_cols) < 3:
            continue

        X_train = train_df_roll[avail_train_cols].fillna(0).values
        y_train = train_df_roll["severe_underestimate_flag"].values

        X_eval = eval_df_enc[avail_eval_cols].fillna(0).values

        # Skip if no positive training samples
        if y_train.sum() < 2:
            # Not enough severe events to train — use default threshold
            gate_prob = np.zeros(len(eval_df))
        else:
            # Train classifier
            model = train_classifier(
                X_train, y_train,
                model_type=args.model_type,
                seed=seed,
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
            )

            last_trained_model = model
            last_feature_cols = avail_train_cols

            # Predict on evaluation set
            gate_prob = model.predict_proba(X_eval)[:, 1]

            # Collect for threshold tuning (use first 10 days as validation)
            if day_idx < 10:
                val_probas.extend(gate_prob.tolist())
                val_targets.extend(eval_df["severe_underestimate_flag"].tolist())

            # Feature importance
            if hasattr(model, "feature_importances_"):
                for col, imp in zip(avail_train_cols, model.feature_importances_):
                    all_importances.append({
                        "feature": col,
                        "importance": round(float(imp), 6),
                        "eval_day": str(eval_day),
                    })

        eval_df_enc["gate_prob"] = gate_prob
        eval_df_enc["severe_flag"] = eval_df["severe_underestimate_flag"].values

        # ── 4. Compute lift ───────────────────────────────────────────
        # Use training data as history for lift quantiles
        lift_df = compute_lift_amount(
            eval_df_enc,
            train_df_roll,
            gate_prob_col="gate_prob",
            threshold=args.threshold if args.threshold else 0.5,
            lift_quantile=args.lift_quantile,
            max_lift_ratio=0.35,
            max_absolute_lift=350.0,
        )

        # Record
        for _, row in lift_df.iterrows():
            all_gate_preds.append({
                "business_day": str(eval_day),
                "hour_business": int(row.get("hour_business", -1)),
                "timestamp": str(row.get("timestamp_dt", "")),
                "period": row.get("period", ""),
                "gate_prob": round(float(row.get("gate_prob", 0)), 6),
                "severe_flag": int(row.get("severe_flag", 0)),
                "gate_active": int(row.get("gate_active", 0)),
                "p33_lift": float(row.get("p33_lift", 0)),
                "p33_lift_source": row.get("p33_lift_source", ""),
            })

        n_gate_active = lift_df["gate_active"].sum()
        daily_results.append({
            "eval_day": str(eval_day),
            "n_timestamps": len(lift_df),
            "n_gate_active": int(n_gate_active),
            "gate_active_rate": round(n_gate_active / max(len(lift_df), 1), 4),
        })

        if (day_idx + 1) % 20 == 0 or day_idx == len(eval_day_set) - 1:
            print(f"  Day {day_idx+1}/{len(eval_day_set)}: eval={eval_day} "
                  f"train_samples={len(y_train)} severe_train={int(y_train.sum())} "
                  f"gate_active={n_gate_active}/{len(lift_df)}")

    # ── 5. Determine threshold ────────────────────────────────────────
    best_threshold = args.threshold if args.threshold else 0.50
    threshold_source = "user_specified" if args.threshold else "default"

    if len(val_probas) > 50:
        # Tune threshold on validation set
        from sklearn.metrics import precision_recall_curve, fbeta_score

        val_probas_arr = np.array(val_probas)
        val_targets_arr = np.array(val_targets)

        best_f2 = 0
        best_t = 0.5
        for t in np.arange(0.05, 0.95, 0.05):
            preds = (val_probas_arr >= t).astype(int)
            f2 = fbeta_score(val_targets_arr, preds, beta=2, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_t = t

        if args.threshold is None:
            best_threshold = best_t
            threshold_source = f"tuned_f2={best_f2:.4f}"
        print(f"  Threshold tuning: best={best_t:.2f} (F2={best_f2:.4f})")

    # ── 6. Write outputs ──────────────────────────────────────────────
    print(f"\n  Writing outputs to {out_dir}...")

    # Gate predictions
    gate_df = pd.DataFrame(all_gate_preds)
    if len(gate_df) > 0:
        # Merge back with pack for timestamps not in eval
        all_ts = ts_df[["business_day", "hour_business", "timestamp", "period",
                        "y_true", "base_fused_pred"]].copy()

        # Apply gate: if no gate prediction, use 0
        gate_df["business_day"] = gate_df["business_day"].astype(str)
        all_ts["business_day"] = all_ts["business_day"].astype(str)

        merged = all_ts.merge(
            gate_df[["business_day", "hour_business", "gate_prob",
                     "severe_flag", "gate_active", "p33_lift", "p33_lift_source"]],
            on=["business_day", "hour_business"],
            how="left",
        )
        merged["gate_prob"] = merged["gate_prob"].fillna(0.0)
        merged["severe_flag"] = merged["severe_flag"].fillna(0).astype(int)
        merged["gate_active"] = merged["gate_active"].fillna(0).astype(int)
        merged["p33_lift"] = merged["p33_lift"].fillna(0.0)

        # Gate risk predictions (for correction pipeline)
        gate_risk = merged[["business_day", "hour_business", "timestamp",
                            "period", "gate_prob"]].copy()
        gate_risk.rename(columns={"gate_prob": "high_spike_prob"}, inplace=True)
        gate_risk["spike_risk_score"] = gate_risk["high_spike_prob"]

        # Save
        gate_csv = out_dir / "gate_predictions.csv"
        merged.to_csv(gate_csv, index=False, encoding="utf-8")
        print(f"  [OK] Gate predictions: {gate_csv} ({len(merged)} rows)")

        risk_csv = out_dir / "gate_risk_predictions.csv"
        gate_risk.to_csv(risk_csv, index=False, encoding="utf-8")
        print(f"  [OK] Gate risk predictions: {risk_csv} ({len(gate_risk)} rows)")
    else:
        print(f"  [WARN] No gate predictions generated — using zeros")
        # Create empty gate predictions
        all_ts = ts_df[["business_day", "hour_business", "timestamp", "period",
                        "y_true", "base_fused_pred"]].copy()
        all_ts["gate_prob"] = 0.0
        all_ts["severe_flag"] = 0
        all_ts["gate_active"] = 0
        all_ts["p33_lift"] = 0.0
        all_ts["p33_lift_source"] = "no_data"
        all_ts.to_csv(out_dir / "gate_predictions.csv", index=False, encoding="utf-8")

        gate_risk = all_ts[["business_day", "hour_business", "timestamp",
                            "period"]].copy()
        gate_risk["high_spike_prob"] = 0.0
        gate_risk["spike_risk_score"] = 0.0
        gate_risk.to_csv(out_dir / "gate_risk_predictions.csv", index=False, encoding="utf-8")

    # Feature importance
    if all_importances:
        imp_df = pd.DataFrame(all_importances)
        # Aggregate mean importance across days
        imp_agg = (
            imp_df.groupby("feature")["importance"]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        imp_csv = out_dir / "feature_importance.csv"
        imp_agg.to_csv(imp_csv, index=False, encoding="utf-8")
        print(f"  [OK] Feature importance: {imp_csv}")

        print(f"\n  Top 10 features (mean importance across rolling folds):")
        for _, row in imp_agg.head(10).iterrows():
            bar = "█" * int(row["importance"] * 200)
            print(f"    {row['feature']:40s} {row['importance']:.4f}  {bar}")

    # Daily summary
    daily_df = pd.DataFrame(daily_results)
    if len(daily_df) > 0:
        daily_csv = out_dir / "daily_gate_summary.csv"
        daily_df.to_csv(daily_csv, index=False, encoding="utf-8")

    # Training summary
    n_active = int(merged["gate_active"].sum()) if len(gate_df) > 0 else 0
    summary = {
        "script": "scripts/train_p33_spike_gated_uplift.py",
        "model_type": args.model_type,
        "date_range": {"start": args.start_date, "end": args.end_date},
        "train_warmup_days": window_days,
        "threshold": best_threshold,
        "threshold_source": threshold_source,
        "lift_quantile": args.lift_quantile,
        "n_eval_days": len(eval_day_set),
        "n_timestamps": len(merged) if len(gate_df) > 0 else 0,
        "n_gate_active": n_active,
        "gate_active_rate": round(n_active / max(len(merged), 1), 4) if len(gate_df) > 0 else 0,
        "n_features": len(last_feature_cols),
        "feature_cols": last_feature_cols,
        "total_severe": int(ts_df["severe_underestimate_flag"].sum()),
        "seed": seed,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = out_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  [OK] Training summary: {summary_path}")

    # ── 7. Quick evaluation on training data ──────────────────────────
    if len(all_gate_preds) > 0:
        gate_eval = pd.DataFrame(all_gate_preds)
        total = len(gate_eval)
        active = gate_eval["gate_active"].sum()
        severe_total = gate_eval["severe_flag"].sum()
        severe_caught = gate_eval[
            (gate_eval["gate_active"] == 1) & (gate_eval["severe_flag"] == 1)
        ].shape[0]
        recall = severe_caught / max(severe_total, 1)
        precision = severe_caught / max(active, 1)

        print(f"\n  ── Gate Performance (on eval days) ──")
        print(f"    Total timestamps evaluated:  {total}")
        print(f"    Gate active:                  {active} ({active/max(total,1)*100:.1f}%)")
        print(f"    Severe underestimates:        {severe_total}")
        print(f"    Severe caught by gate:        {severe_caught}")
        print(f"    Recall (severe caught rate):  {recall:.4f}")
        print(f"    Precision:                    {precision:.4f}")

    print(f"\n  Done. Output in: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
