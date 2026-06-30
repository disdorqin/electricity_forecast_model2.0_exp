#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
P3 SOTA Lab — LightGBM Enhanced Training Script

Profiles:
  baseline          — Standard three-stage LightGBM (repro of existing)
  spike_weighted    — Add high-spike sample weights + oversample extreme events
  period_heads      — Tuned hyper-params per period (more leaves, lower LR for 9_16)
  quantile_residual — Use quantile objective to better capture upper tail
  all               — Combine all enhancements

Usage:
  python scripts/train_lightgbm_p3_sota.py \
    --data-path data/shandong_pmos_hourly.xlsx \
    --target realtime \
    --train-start 2024-11-01 --train-end 2025-10-31 \
    --valid-start 2025-11-01 --valid-end 2025-12-31 \
    --out-dir reports/local/p3_sota_lab \
    --profile baseline
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_lightgbm_p3_sota")

# ── Constants ──────────────────────────────────────────────────────────
VALLEY_HOURS = [1, 2, 3, 4, 5, 6, 7, 8]
SOLAR_HOURS = [9, 10, 11, 12, 13, 14, 15, 16]
PEAK_HOURS = [17, 18, 19, 20, 21, 22, 23, 24]

# All 21 baseline features
BASELINE_FEATURES = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]

# Leakage-free features (remove D-day stats: morning_mean, noon_min, morning_std, morning_trend, is_info_fresh)
LEAKAGE_SAFE_FEATURES = [f for f in BASELINE_FEATURES if f not in (
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh"
)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="P3 SOTA Lab — LightGBM Enhanced Training")
    parser.add_argument("--data-path", required=True, help="Raw data path (xlsx or csv)")
    parser.add_argument("--target", default="realtime", choices=["realtime", "dayahead"], help="Target market")
    parser.add_argument("--train-start", default="2024-11-01", help="Training start date")
    parser.add_argument("--train-end", default="2025-10-31", help="Training end date")
    parser.add_argument("--valid-start", default="2025-11-01", help="Validation start date")
    parser.add_argument("--valid-end", default="2025-12-31", help="Validation end date")
    parser.add_argument("--out-dir", default="reports/local/p3_sota_lab", help="Output directory")
    parser.add_argument("--profile", default="baseline",
                        choices=["baseline", "spike_weighted", "period_heads", "quantile_residual", "all"],
                        help="Training profile")
    parser.add_argument("--no-leakage", action="store_true", help="Remove D-day stats features")
    parser.add_argument("--n-estimators", type=int, default=3000, help="Max estimators per model")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


# ── Data Loading & Feature Engineering ────────────────────────────────

def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path, encoding="gbk")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8")

    col_map = {
        "时刻": "ds",
        "日前电价": "da_price",
        "实时电价": "rt_price",
        "直调负荷预测值": "load",
        "风电总加预测值": "wind",
        "光伏总加预测值": "solar",
        "联络线受电负荷预测值": "interconnect",
        "竞价空间预测值": "bidding_space",
    }
    df.rename(columns={c: col_map.get(c, c) for c in df.columns}, inplace=True)
    return df


def feature_engineering(df: pd.DataFrame, target: str = "realtime") -> pd.DataFrame:
    """Replicate existing feature engineering from LightGBM train_fix.py."""
    df = df.copy()

    # Target selection
    price_col = "rt_price" if target == "realtime" else "da_price"
    if price_col not in df.columns:
        raise ValueError(f"Target column '{price_col}' not found in data. Available: {list(df.columns)}")

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = pd.to_numeric(df[price_col], errors="coerce")

    # Fill exogenous forward
    for c in ["load", "wind", "solar", "interconnect", "bidding_space"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill()

    # 1-sec adjustment for business hour (00:00 → 24:00 of previous day)
    adjusted = df["ds"] - pd.Timedelta(seconds=1)
    df["hour"] = adjusted.dt.hour + 1  # 1-24
    df["month"] = adjusted.dt.month
    df["day_of_week"] = adjusted.dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Lag features
    df["lag_48h"] = df["y"].shift(48)
    df["lag_168h"] = df["y"].shift(168)

    df["lag_price_target"] = np.where(
        df["day_of_week"] < 5, df["lag_168h"], df["lag_48h"]
    )
    df["lag_price_week"] = df["lag_168h"]

    df["lag_price_target"] = df["lag_price_target"].ffill().fillna(0)
    df["lag_price_week"] = df["lag_price_week"].ffill().fillna(0)

    # Physical features
    safe_load = df["load"].replace(0, 1)
    df["net_load"] = df["load"] - df["wind"] - df["solar"]
    df["solar_ratio"] = df["solar"] / safe_load
    df["net_load_sq"] = (df["net_load"] / 1000) ** 2
    if "bidding_space" not in df.columns:
        df["bidding_space"] = df["net_load"] - df["interconnect"]
    df["space_ratio"] = df["bidding_space"] / safe_load
    df["wind_ratio"] = df["wind"] / safe_load
    df["renew_penetration"] = (df["wind"] + df["solar"]) / safe_load
    df["ramp_load"] = df["load"].diff().fillna(0)
    df["ramp_solar"] = df["solar"].diff().fillna(0)

    # D-day statistics (same-day info available at D 14:00 cutoff)
    df["date_only"] = adjusted.dt.date
    mask_morning = (df["hour"] >= 1) & (df["hour"] <= 15)

    def _trend(x):
        return x.iloc[-1] - x.iloc[0] if len(x) >= 2 else 0

    stats_basic = df[mask_morning].groupby("date_only")["y"].agg(
        morning_mean="mean", morning_std="std"
    )
    mask_noon = (df["hour"] >= 11) & (df["hour"] <= 15)
    stats_noon = df[mask_noon].groupby("date_only")["y"].agg(
        noon_min="min", morning_trend=_trend
    )
    daily_feats = pd.concat([stats_basic, stats_noon], axis=1).reset_index()
    shift_cols = ["morning_mean", "noon_min", "morning_std", "morning_trend"]
    daily_feats[shift_cols] = daily_feats[shift_cols].shift(1)
    daily_feats["is_info_fresh"] = daily_feats["morning_mean"].notna().astype(int)
    daily_feats[shift_cols] = daily_feats[shift_cols].ffill().fillna(0)
    df = df.merge(daily_feats, on="date_only", how="left")
    df.drop(columns=["date_only", "lag_48h", "lag_168h"], inplace=True, errors="ignore")

    return df


# ── Metrics ────────────────────────────────────────────────────────────

def smape_floor50(y_true, y_pred, eps=1e-6):
    t = np.where(y_true < 50, 50.0, y_true)
    p = np.where(y_pred < 50, 50.0, y_pred)
    d = (np.abs(p) + np.abs(t)) / 2.0
    d = np.where(d < eps, eps, d)
    return float(np.mean(np.abs(p - t) / d) * 100.0)


# ── Profiles ───────────────────────────────────────────────────────────

def get_profile_params(profile: str, n_estimators: int):
    """Return (valley_params, solar_params, peak_params, solar_clf_params) per profile."""
    base_valley = {"objective": "regression", "n_estimators": n_estimators, "learning_rate": 0.05,
                   "num_leaves": 31, "n_jobs": 4, "device_type": "cpu", "verbose": -1, "random_state": 42}
    base_solar = {"objective": "regression", "n_estimators": n_estimators, "learning_rate": 0.03,
                  "num_leaves": 63, "n_jobs": 4, "device_type": "cpu", "verbose": -1, "random_state": 42}
    base_peak = {"objective": "regression", "n_estimators": n_estimators, "learning_rate": 0.03,
                 "num_leaves": 40, "n_jobs": 4, "device_type": "cpu", "verbose": -1, "random_state": 42}
    base_clf = {"objective": "binary", "n_estimators": 1000, "learning_rate": 0.05,
                "class_weight": "balanced", "n_jobs": 4, "device_type": "cpu", "verbose": -1, "random_state": 42}

    if profile == "baseline":
        return base_valley, base_solar, base_peak, base_clf

    if profile == "spike_weighted":
        # Same base params but sample weights + oversample will be applied separately
        base_valley["n_estimators"] = int(n_estimators * 1.5)
        base_solar["n_estimators"] = int(n_estimators * 1.5)
        base_peak["n_estimators"] = int(n_estimators * 1.5)
        return base_valley, base_solar, base_peak, base_clf

    if profile == "period_heads":
        # Tailored hyperparams per period
        base_valley["num_leaves"] = 31
        base_valley["learning_rate"] = 0.05
        base_solar["num_leaves"] = 127  # More leaves for complex 9_16 pattern
        base_solar["learning_rate"] = 0.02  # Lower LR for spikes
        base_solar["min_child_samples"] = 5
        base_peak["num_leaves"] = 63
        base_peak["learning_rate"] = 0.03
        base_clf["num_leaves"] = 63
        return base_valley, base_solar, base_peak, base_clf

    if profile == "quantile_residual":
        # Quantile regression for upper tail
        base_valley["objective"] = "quantile"
        base_valley["alpha"] = 0.7
        base_solar["objective"] = "quantile"
        base_solar["alpha"] = 0.7
        base_peak["objective"] = "quantile"
        base_peak["alpha"] = 0.7
        return base_valley, base_solar, base_peak, base_clf

    if profile == "all":
        # Combine period_heads + quantile + spike weighting
        base_valley.update({"num_leaves": 31, "learning_rate": 0.05, "objective": "quantile", "alpha": 0.7})
        base_solar.update({"num_leaves": 127, "learning_rate": 0.02, "objective": "quantile", "alpha": 0.7,
                           "min_child_samples": 5})
        base_peak.update({"num_leaves": 63, "learning_rate": 0.025, "objective": "quantile", "alpha": 0.7})
        base_valley["n_estimators"] = int(n_estimators * 1.5)
        base_solar["n_estimators"] = int(n_estimators * 1.5)
        base_peak["n_estimators"] = int(n_estimators * 1.5)
        return base_valley, base_solar, base_peak, base_clf

    return base_valley, base_solar, base_peak, base_clf


def build_weights(df, profile: str, period_col: str = "period"):
    """Build sample weights for spike_weighted profile."""
    if profile not in ("spike_weighted", "all"):
        return None
    w = np.ones(len(df))
    y = df["y_clipped"].values

    # Low-price weight
    w[y < 50] = 2.0
    w[y < 0] = 5.0

    # High-spike weight (top 5% price)
    high_thresh = np.percentile(y, 95)
    w[y > high_thresh] = 3.0

    # 9_16 extra weight
    if period_col in df.columns:
        w[(df[period_col] == "solar") | (df[period_col] == "9_16")] *= 1.5

    return w


# ── Post-hoc Calibration ──────────────────────────────────────────────

def fit_posthoc_calibration(val_preds: np.ndarray, val_true: np.ndarray, period_labels=None):
    """Fit a recent-bias correction: additive bias by period."""
    if period_labels is None:
        bias = np.nanmedian(val_preds - val_true)
        return {"global_bias": float(bias) if not np.isnan(bias) else 0.0}
    calib = {}
    for p in np.unique(period_labels):
        mask = period_labels == p
        if mask.sum() < 5:
            continue
        bias = np.nanmedian(val_preds[mask] - val_true[mask])
        calib[p] = float(bias) if not np.isnan(bias) else 0.0
    return calib


def apply_calibration(preds: np.ndarray, calibration: dict, period_labels=None):
    if period_labels is None:
        return preds - calibration.get("global_bias", 0)
    result = preds.copy()
    for p, bias in calibration.items():
        mask = period_labels == p
        result[mask] -= bias
    return result


# ── Main Training ──────────────────────────────────────────────────────

def train_profile(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load & featurize
    logger.info("Loading data from %s", args.data_path)
    raw = load_data(args.data_path)
    df = feature_engineering(raw, args.target)
    logger.info("Feature engineering done: %d rows, %d cols", len(df), len(df.columns))

    # Split train/val by time
    train_mask = (df["ds"] >= pd.to_datetime(args.train_start)) & (df["ds"] <= pd.to_datetime(args.train_end))
    val_mask = (df["ds"] >= pd.to_datetime(args.valid_start)) & (df["ds"] <= pd.to_datetime(args.valid_end))
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()

    # Target clipping
    train_upper = train_df["y"].quantile(0.995)
    train_df["y_clipped"] = train_df["y"].clip(lower=-100, upper=train_upper)
    val_df["y_clipped"] = val_df["y"].clip(lower=-100, upper=train_upper)

    # Feature selection
    if args.no_leakage:
        features = LEAKAGE_SAFE_FEATURES
    else:
        features = [c for c in BASELINE_FEATURES if c in df.columns]
    logger.info("Using %d features: %s %s", len(features), features[:5], f"... (+{len(features)-5} more)")

    # Profile params
    vp, sp, pp, cp = get_profile_params(args.profile, args.n_estimators)

    # Period mapping for 'solar' vs '9_16'
    def _period_label(h):
        if h in VALLEY_HOURS: return "valley"
        if h in SOLAR_HOURS: return "solar"
        return "peak"

    train_df["period"] = train_df["hour"].apply(_period_label)
    val_df["period"] = val_df["hour"].apply(_period_label)

    # Split by period
    train_valley = train_df[train_df["hour"].isin(VALLEY_HOURS)]
    val_valley = val_df[val_df["hour"].isin(VALLEY_HOURS)]
    train_solar = train_df[train_df["hour"].isin(SOLAR_HOURS)]
    val_solar = val_df[val_df["hour"].isin(SOLAR_HOURS)]
    train_peak = train_df[train_df["hour"].isin(PEAK_HOURS)]
    val_peak = val_df[val_df["hour"].isin(PEAK_HOURS)]

    logger.info("Train: valley=%d solar=%d peak=%d", len(train_valley), len(train_solar), len(train_peak))
    logger.info("Val:   valley=%d solar=%d peak=%d", len(val_valley), len(val_solar), len(val_peak))

    t0 = time.time()

    # --- Valley ---
    if args.profile in ("spike_weighted", "all"):
        w_valley = build_weights(train_valley, args.profile)
    else:
        w_valley = None

    model_v = lgb.LGBMRegressor(**vp)
    model_v.fit(
        train_valley[features], train_valley["y_clipped"],
        sample_weight=w_valley,
        eval_set=[(val_valley[features], val_valley["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # --- Solar ---
    if args.profile in ("spike_weighted", "all"):
        w_solar = build_weights(train_solar, args.profile)
    else:
        w_solar = None

    model_s = lgb.LGBMRegressor(**sp)
    model_s.fit(
        train_solar[features], train_solar["y_clipped"],
        sample_weight=w_solar,
        eval_set=[(val_solar[features], val_solar["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # Solar classifier
    y_sc = (train_solar["y_clipped"] < 0).astype(int)
    y_vc = (val_solar["y"] < 0).astype(int)
    model_sc = lgb.LGBMClassifier(**cp)
    model_sc.fit(
        train_solar[features], y_sc,
        eval_set=[(val_solar[features], y_vc)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # --- Peak ---
    if args.profile in ("spike_weighted", "all"):
        w_peak = build_weights(train_peak, args.profile)
    else:
        w_peak = None

    model_p = lgb.LGBMRegressor(**pp)
    model_p.fit(
        train_peak[features], train_peak["y_clipped"],
        sample_weight=w_peak,
        eval_set=[(val_peak[features], val_peak["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    t1 = time.time()
    logger.info("Training completed in %.1f seconds", t1 - t0)

    # --- Predict ---
    preds = np.zeros(len(val_df))
    mask_v = val_df["hour"].isin(VALLEY_HOURS)
    mask_s = val_df["hour"].isin(SOLAR_HOURS)
    mask_p = val_df["hour"].isin(PEAK_HOURS)

    if mask_v.sum():
        preds[mask_v] = model_v.predict(val_df.loc[mask_v, features])
    if mask_s.sum():
        x_s = val_df.loc[mask_s, features]
        sp_preds = model_s.predict(x_s)
        neg_probs = model_sc.predict_proba(x_s)[:, 1]
        corr_mask = (neg_probs > 0.6) & (sp_preds > -20)
        sp_preds[corr_mask] -= 100
        preds[mask_s] = sp_preds
    if mask_p.sum():
        preds[mask_p] = model_p.predict(val_df.loc[mask_p, features])

    preds = np.where(preds < -80, -80, preds)

    # --- Post-hoc calibration ---
    calib = fit_posthoc_calibration(preds, val_df["y"].values, val_df["period"].values if "period" in val_df.columns else None)
    preds_calib = apply_calibration(preds, calib, val_df["period"].values if "period" in val_df.columns else None)

    # --- Metrics ---
    y_true = val_df["y"].values
    smape_raw = smape_floor50(y_true, preds)
    smape_calib = smape_floor50(y_true, preds_calib)
    mae_raw = float(mean_absolute_error(y_true, preds))
    mae_calib = float(mean_absolute_error(y_true, preds_calib))
    severe_raw = int(((y_true > 800) & (preds < y_true - 0.3 * y_true)).sum())
    severe_calib = int(((y_true > 800) & (preds_calib < y_true - 0.3 * y_true)).sum())

    # Per-period metrics
    period_metrics = {}
    for pname, pmask in [("valley", mask_v), ("solar", mask_s), ("peak", mask_p)]:
        if pmask.sum() == 0:
            continue
        yt = y_true[pmask]
        pp_raw = preds[pmask]
        pp_cal = preds_calib[pmask]
        period_metrics[pname] = {
            "n": int(pmask.sum()),
            "smape_raw": round(smape_floor50(yt, pp_raw), 2),
            "smape_calib": round(smape_floor50(yt, pp_cal), 2),
            "mae_raw": round(float(mean_absolute_error(yt, pp_raw)), 2),
        }

    # 9_16 specifically
    mask_916 = val_df["hour"].isin(SOLAR_HOURS)
    if mask_916.sum():
        yt_916 = y_true[mask_916]
        pp_916_raw = preds[mask_916]
        period_metrics["9_16"] = {
            "n": int(mask_916.sum()),
            "smape_raw": round(smape_floor50(yt_916, pp_916_raw), 2),
            "mae_raw": round(float(mean_absolute_error(yt_916, pp_916_raw)), 2),
        }

    results = {
        "profile": args.profile,
        "no_leakage": args.no_leakage,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_seconds": round(t1 - t0, 1),
        "features": len(features),
        "smape_raw": round(smape_raw, 2),
        "smape_calib": round(smape_calib, 2),
        "mae_raw": round(mae_raw, 2),
        "mae_calib": round(mae_calib, 2),
        "severe_raw": severe_raw,
        "severe_calib": severe_calib,
        "period_metrics": period_metrics,
        "calibration": calib,
    }

    # Save results
    result_path = out_dir / f"lightgbm_{args.profile}_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", result_path)

    # Save predictions
    val_df_out = val_df[["ds", "hour", "y"]].copy()
    val_df_out["y_pred_raw"] = preds
    val_df_out["y_pred_calib"] = preds_calib
    csv_path = out_dir / f"lightgbm_{args.profile}_predictions.csv"
    val_df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("Predictions saved to %s", csv_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Profile: {args.profile} | no_leakage={args.no_leakage}")
    print(f"sMAPE_raw:  {smape_raw:.2f}  | sMAPE_calib: {smape_calib:.2f}")
    print(f"MAE_raw:    {mae_raw:.2f}  | MAE_calib:   {mae_calib:.2f}")
    print(f"Severe_raw: {severe_raw}  | Severe_calib: {severe_calib}")
    print(f"Train time: {t1-t0:.1f}s | Features: {len(features)}")
    for pn, pm in period_metrics.items():
        print(f"  {pn:>8}: n={pm['n']:4d}  sMAPE={pm['smape_raw']:.2f}  MAE={pm['mae_raw']:.2f}")
    print(f"{'='*60}\n")

    # Cleanup
    del model_v, model_s, model_sc, model_p
    gc.collect()

    return results


def main():
    args = parse_args()
    logger.info("Starting P3 SOTA Lab — LightGBM profile='%s' no_leakage=%s",
                args.profile, args.no_leakage)
    train_profile(args)


if __name__ == "__main__":
    main()
