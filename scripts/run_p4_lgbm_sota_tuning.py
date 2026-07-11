#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_p4_lgbm_sota_tuning.py — P4 SOTA LightGBM hyperparameter tuning.

4 directions × ≤16 combos:

  1. sample_weight_profile: none, spike_weighted, severe_underestimate_weighted,
     period_spike_weighted
  2. objective/profile: regression_l1, huber, quantile_0.7, quantile_0.8
  3. period-specific params: 9_16_deep, peak_conservative
  4. feature set: forecast_spread_features, net_load_features

Budget:
  - Small window 2025-11-01 ~ 2025-11-15: 12 combos
  - Full window 2025-11-01 ~ 2025-12-31: top 3 combos

Targets:
  Single-model GO: sMAPE ≤ 22.02, severe ≤ 80
  Strong GO:       sMAPE ≤ 20.86, severe ≤ 63

Usage:
    python scripts/run_p4_lgbm_sota_tuning.py
    python scripts/run_p4_lgbm_sota_tuning.py --quick  (smoke test, 1 combo)

Output:
    reports/local/p4_lgbm_sota_tuning/
    ├── tuning_summary.json          — all small-window results + full-window top 3
    ├── best_model_config.json       — best combo config for deployment
    ├── predictions_top1.csv         — full-window predictions from best combo
    └── combo_{name}/
        └── metrics.json
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lightGBM.train_fix import LGBMPowerPredictor, ThreeStageLGBM
from lightGBM.infer_fix import PowerInference

# ── Constants ──────────────────────────────────────────────────────────

VALLEY_HOURS = [1, 2, 3, 4, 5, 6, 7, 8]
SOLAR_HOURS = [9, 10, 11, 12, 13, 14, 15, 16]
PEAK_HOURS = [17, 18, 19, 20, 21, 22, 23, 24]

SAMPLE_WEIGHT_PROFILES = (
    "none", "spike_weighted", "severe_underestimate_weighted", "period_spike_weighted"
)

OUT_DIR = Path("reports/local/p4_lgbm_sota_tuning")
SMALL_WINDOW_START = "2025-11-01"
SMALL_WINDOW_END = "2025-11-15"
FULL_WINDOW_START = "2025-11-01"
FULL_WINDOW_END = "2025-12-31"
DATA_PATH = "data/shandong_pmos_hourly.xlsx"

# ── Combo definitions ──────────────────────────────────────────────────

PERIOD_PARAMS_MAP = {
    "baseline": {
        "valley": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31},
        "solar":  {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 63},
        "peak":   {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 40},
    },
    "9_16_deep": {
        "valley": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31},
        "solar":  {"n_estimators": 5000, "learning_rate": 0.02, "num_leaves": 127},
        "peak":   {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 40},
    },
    "peak_conservative": {
        "valley": {"n_estimators": 2000, "learning_rate": 0.05, "num_leaves": 31},
        "solar":  {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 63},
        "peak":   {"n_estimators": 1500, "learning_rate": 0.02, "num_leaves": 31},
    },
}

BASELINE_FEATURES = [
    "hour", "month", "day_of_week", "is_weekend",
    "lag_price_target", "lag_price_week",
    "load", "wind", "solar", "interconnect",
    "bidding_space", "space_ratio",
    "net_load", "solar_ratio", "net_load_sq",
    "wind_ratio", "renew_penetration", "ramp_load", "ramp_solar",
    "morning_mean", "noon_min", "morning_std", "morning_trend", "is_info_fresh",
]

FORECAST_SPREAD_FEATURES = [
    "local_power_ratio", "nuclear_ratio", "self_unit_ratio",
    "renewable_ratio", "forecast_spread",
]

NET_LOAD_FEATURES = [
    "net_load_ma_3", "net_load_ma_6", "net_load_change",
    "net_load_volatility", "net_load_acceleration",
]

# Ordered combos — 12 unique, ≤ 16 budget
def build_combo_list() -> list[dict[str, Any]]:
    """Return the list of 12 combos to evaluate."""
    combos: list[dict[str, Any]] = []

    # D1: sample_weight_profile (baseline obj, pp, fs)
    for sw_name in ["none", "spike_weighted", "severe_underestimate_weighted", "period_spike_weighted"]:
        combos.append({
            "name": f"sw_{sw_name}",
            "sample_weight_profile": sw_name,
            "objective": "regression",
            "alpha": None,
            "period_params": "baseline",
            "feature_set": "baseline",
            "direction": "sample_weight_profile",
        })

    # D2: objective (no sample weighting)
    obj_defs = [
        ("regression_l1", None),
        ("huber", None),
        ("quantile", 0.70),
        ("quantile", 0.80),
    ]
    for obj_name, alpha in obj_defs:
        label = obj_name if alpha is None else f"{obj_name}_{alpha:.1f}"
        combos.append({
            "name": f"obj_{label}".replace(".", "p"),
            "sample_weight_profile": "none",
            "objective": obj_name,
            "alpha": alpha,
            "period_params": "baseline",
            "feature_set": "baseline",
            "direction": "objective",
        })

    # D3: period-specific params (baseline obj, no weighting)
    for pp_name in ["9_16_deep", "peak_conservative"]:
        combos.append({
            "name": f"pp_{pp_name}",
            "sample_weight_profile": "none",
            "objective": "regression",
            "alpha": None,
            "period_params": pp_name,
            "feature_set": "baseline",
            "direction": "period_params",
        })

    # D4: feature set (baseline obj, no weighting)
    for fs_name in ["forecast_spread", "net_load"]:
        combos.append({
            "name": f"fs_{fs_name}",
            "sample_weight_profile": "none",
            "objective": "regression",
            "alpha": None,
            "period_params": "baseline",
            "feature_set": f"{fs_name}_features",
            "direction": "feature_set",
        })

    return combos


# ── Spike weight helper (from main_fix) ───────────────────────────────

def _compute_spike_weights(
    train_df: pd.DataFrame,
    profile: str,
    y_col: str = "y_clipped",
    hour_col: str = "hour",
) -> np.ndarray:
    """Compute sample weights based on spike/severe profile (leakage-safe)."""
    weights = np.ones(len(train_df), dtype=np.float64)
    if profile == "none":
        return weights

    y = train_df[y_col].values
    hour = train_df[hour_col].values

    p90 = float(np.percentile(y, 90))
    p95 = float(np.percentile(y, 95))

    spike_threshold = max(p90, 150.0)
    severe_threshold = max(p95, 250.0)

    is_high_spike = y > spike_threshold
    is_severe = y > severe_threshold
    is_9_16 = (hour >= 9) & (hour <= 16)

    if profile == "spike_weighted":
        weights[is_high_spike & is_9_16] = 6.0
        weights[is_high_spike & ~is_9_16] = 3.0
    elif profile == "severe_underestimate_weighted":
        weights[is_severe] = 4.0
    elif profile == "period_spike_weighted":
        weights[is_high_spike & is_9_16] = 8.0
        weights[is_high_spike & ~is_9_16] = 3.0
    else:
        raise ValueError(f"Unknown sample_weight_profile: {profile}")

    return weights


# ── Feature engineering helpers ───────────────────────────────────────

def _add_forecast_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features from additional exogenous forecast columns.

    Uses CSV columns not loaded by default LGBMPowerPredictor.
    """
    df = df.copy()
    safe_load = df["load"].replace(0, np.nan).fillna(1)

    # Exogenous forecast columns from CSV
    exog_map = {
        "local_power_ratio":      ("地方电厂总加预测值", lambda v: v / safe_load),
        "nuclear_ratio":          ("核电总加预测值", lambda v: v / safe_load),
        "self_unit_ratio":        ("自备机组总加预测值", lambda v: v / safe_load),
        "renewable_ratio":        ("新能源总加预测值", lambda v: v / safe_load),
    }

    for feat_name, (col_name, transform) in exog_map.items():
        if col_name in df.columns:
            vals = pd.to_numeric(df[col_name], errors="coerce").fillna(0)
            df[feat_name] = transform(vals)
        else:
            df[feat_name] = 0.0

    # Forecast spread: std across all available forecast columns
    forecast_cols = ["load", "wind", "solar", "interconnect"]
    for col_name in ["地方电厂总加预测值", "核电总加预测值", "自备机组总加预测值", "新能源总加预测值"]:
        if col_name in df.columns:
            forecast_cols.append(col_name)
    # Normalize each column before computing std
    spread_vals = []
    for col in forecast_cols:
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0).values
        col_max = np.max(np.abs(vals))
        if col_max > 0:
            spread_vals.append(vals / col_max)
        else:
            spread_vals.append(np.zeros_like(vals))
    if spread_vals:
        df["forecast_spread"] = np.std(spread_vals, axis=0)
    else:
        df["forecast_spread"] = 0.0

    return df


def _add_net_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived net_load features."""
    df = df.copy()
    nl = df["net_load"].values

    # Moving averages (use expanding at edges)
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = np.mean(arr[start:i + 1])
        return result

    df["net_load_ma_3"] = rolling_mean(nl, 3)
    df["net_load_ma_6"] = rolling_mean(nl, 6)
    df["net_load_change"] = np.diff(nl, prepend=nl[0])
    # Volatility: rolling 6-hour std
    def rolling_std(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            result[i] = np.std(arr[start:i + 1])
        return result
    df["net_load_volatility"] = rolling_std(nl, 6)
    df["net_load_acceleration"] = np.diff(df["net_load_change"].values, prepend=0)

    # Fill NaN from edge effects
    for col in ["net_load_ma_3", "net_load_ma_6", "net_load_change",
                "net_load_volatility", "net_load_acceleration"]:
        df[col] = df[col].ffill().fillna(0)

    return df


def _get_feature_list(feature_set: str) -> list[str]:
    """Return the feature list for a given feature set."""
    if feature_set == "baseline":
        return BASELINE_FEATURES.copy()
    elif feature_set == "forecast_spread_features":
        return BASELINE_FEATURES + FORECAST_SPREAD_FEATURES
    elif feature_set == "net_load_features":
        return BASELINE_FEATURES + NET_LOAD_FEATURES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")


# ── Metrics ───────────────────────────────────────────────────────────

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE_floor50: symmetric MAPE with 50 floor and cap."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    smape = np.minimum(smape, 50.0)
    return float(np.mean(smape))


def compute_combo_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute evaluation metrics from a predictions DataFrame.

    Expects columns: 'y' (actual) and 'pred_y' (predicted).
    """
    valid = df.dropna(subset=["y", "pred_y"]).copy()
    if len(valid) == 0:
        return {"smape_floor50": None, "severe_count": None, "n": 0}

    y_true = valid["y"].values
    y_pred = valid["pred_y"].values

    smape = compute_smape(y_true, y_pred)
    severe = int((y_true - y_pred > 200).sum())

    # Per-period sMAPE
    if "hour" in valid.columns:
        p9_16 = valid[valid["hour"].between(9, 16)]
        p9_16_smape = compute_smape(p9_16["y"].values, p9_16["pred_y"].values) if len(p9_16) > 0 else None
    else:
        p9_16_smape = None

    return {
        "smape_floor50": round(smape, 4),
        "severe_count": severe,
        "9_16_smape_floor50": round(p9_16_smape, 4) if p9_16_smape is not None else None,
        "n_total": len(valid),
        "n_9_16": len(p9_16) if "hour" in valid.columns else None,
    }


# ── Core training (parameterized version of main_fix._fit_realtime_fixed_window) ─

def _p4_fit_window(
    predictor: LGBMPowerPredictor,
    data_path: str,
    history_start_date: str,
    history_end_date: str,
    target: str,
    raw_df: pd.DataFrame | None = None,
    val_ratio: float = 0.2,
    sample_weight_profile: str | None = None,
    objective: str = "regression",
    alpha: float | None = None,
    period_params: str = "baseline",
    feature_set: str = "baseline",
) -> dict[str, Any]:
    """Train a ThreeStageLGBM with tunable params. Returns result dict."""
    # — 1. Load & feature engineering —
    if raw_df is None:
        raw_df = predictor.load_and_process_data(data_path, target)
    else:
        raw_df = raw_df.copy()

    full_df = predictor.feature_engineering(raw_df)

    # Add extra features if needed
    if feature_set == "forecast_spread_features":
        full_df = _add_forecast_spread_features(full_df)
    elif feature_set == "net_load_features":
        full_df = _add_net_load_features(full_df)

    # Update features list
    features_list = _get_feature_list(feature_set)
    predictor.features_list = features_list

    # — 2. Train/val split —
    history_start_dt = pd.to_datetime(history_start_date)
    history_end_dt = pd.to_datetime(history_end_date)
    history_mask = (full_df["ds"] >= history_start_dt) & (full_df["ds"] <= history_end_dt)
    history_df = full_df[history_mask].copy()
    if len(history_df) < 2000:
        raise RuntimeError(f"Training set too small: {len(history_df)} rows")

    train_df, test_df_raw = _split_history_train_val(history_df, val_ratio=val_ratio)
    predictor.validate_optimize_dataset(
        test_df_raw,
        str(test_df_raw["ds"].min()),
        str(test_df_raw["ds"].max()),
    )

    train_upper = train_df["y"].quantile(0.995)
    train_df["y_clipped"] = train_df["y"].clip(lower=-100, upper=train_upper)

    use_profile_weights = sample_weight_profile is not None
    pp_cfg = PERIOD_PARAMS_MAP.get(period_params, PERIOD_PARAMS_MAP["baseline"])

    # Build LGBMRegressor base kwargs
    def _make_regressor_kwargs(period: str) -> dict:
        cfg = pp_cfg[period].copy()
        cfg["objective"] = objective
        if objective == "quantile" and alpha is not None:
            cfg["alpha"] = alpha
        cfg["n_jobs"] = predictor.lgbm_n_jobs
        cfg["device_type"] = predictor._device_type()
        cfg["verbose"] = -1
        cfg["random_state"] = 42
        return cfg

    # — 3. Valley model —
    train_valley = train_df[train_df["hour"].isin(VALLEY_HOURS)]
    test_valley = test_df_raw[test_df_raw["hour"].isin(VALLEY_HOURS)]
    w_valley = _compute_spike_weights(train_valley, sample_weight_profile) if use_profile_weights else np.ones(len(train_valley))

    valley_kwargs = _make_regressor_kwargs("valley")
    model_valley_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(**valley_kwargs),
        train_valley[predictor.features_list],
        train_valley["y_clipped"],
        sample_weight=w_valley,
        eval_set=[(test_valley[predictor.features_list], test_valley["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # — 4. Solar model —
    train_solar = train_df[train_df["hour"].isin(SOLAR_HOURS)]
    test_solar = test_df_raw[test_df_raw["hour"].isin(SOLAR_HOURS)]
    if use_profile_weights:
        w_solar = _compute_spike_weights(train_solar, sample_weight_profile)
    else:
        w_solar = np.ones(len(train_solar))
        y_solar_val = train_solar["y_clipped"].values
        w_solar[y_solar_val < 50] = 2
        w_solar[y_solar_val < 0] = 5

    solar_kwargs = _make_regressor_kwargs("solar")
    model_solar_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(**solar_kwargs),
        train_solar[predictor.features_list],
        train_solar["y_clipped"],
        sample_weight=w_solar,
        eval_set=[(test_solar[predictor.features_list], test_solar["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    y_solar_class = (train_solar["y_clipped"] < 0).astype(int)
    y_solar_test_class = (test_solar["y"] < 0).astype(int)
    model_solar_clf = predictor._fit_with_cuda_fallback(
        lgb.LGBMClassifier(
            objective="binary",
            n_estimators=1000,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=predictor.lgbm_n_jobs,
            device_type=predictor._device_type(),
            verbose=-1,
            random_state=42,
        ),
        train_solar[predictor.features_list],
        y_solar_class,
        eval_set=[(test_solar[predictor.features_list], y_solar_test_class)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # — 5. Peak model —
    train_peak = train_df[train_df["hour"].isin(PEAK_HOURS)]
    test_peak = test_df_raw[test_df_raw["hour"].isin(PEAK_HOURS)]
    if use_profile_weights:
        w_peak = _compute_spike_weights(train_peak, sample_weight_profile)
    else:
        w_peak = np.ones(len(train_peak))
        high_wind_threshold = train_peak["wind"].quantile(0.8)
        w_peak[train_peak["wind"] > high_wind_threshold] = 3

    peak_kwargs = _make_regressor_kwargs("peak")
    model_peak_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(**peak_kwargs),
        train_peak[predictor.features_list],
        train_peak["y_clipped"],
        sample_weight=w_peak,
        eval_set=[(test_peak[predictor.features_list], test_peak["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # — 6. Combine & predict —
    combined_model = ThreeStageLGBM(model_valley_reg, model_solar_reg, model_solar_clf, model_peak_reg)
    pred = combined_model.predict(test_df_raw[predictor.features_list])
    pred = np.where(pred < -80, -80, pred)
    mae = mean_absolute_error(test_df_raw["y"], pred)
    smape = predictor.calculate_smape(test_df_raw["y"], pred)

    return {
        "months_back": 12,
        "train_start": history_start_dt,
        "mae": mae,
        "smape": smape,
        "model": combined_model,
        "features_list": predictor.features_list,
    }


def _split_history_train_val(history_df, val_ratio=0.2, min_val_rows=24 * 7):
    """Chronological train/val split (same as main_fix)."""
    history_df = history_df.sort_values("ds").copy()
    if history_df.empty:
        raise RuntimeError("History window is empty.")
    val_rows = max(int(len(history_df) * float(val_ratio)), int(min_val_rows))
    val_rows = min(val_rows, max(1, len(history_df) - 1))
    split_idx = len(history_df) - val_rows
    train_df = history_df.iloc[:split_idx].copy()
    val_df = history_df.iloc[split_idx:].copy()
    if train_df.empty or val_df.empty:
        raise RuntimeError("Chronological train/val split failed.")
    return train_df, val_df


# ── Rolling simulation (parameterized version) ────────────────────────

def _p4_run_simulation(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    target: str = "实时电价",
    training_months: int = 12,
    val_ratio: float = 0.2,
    sample_weight_profile: str | None = None,
    objective: str = "regression",
    alpha: float | None = None,
    period_params: str = "baseline",
    feature_set: str = "baseline",
) -> pd.DataFrame | None:
    """Rolling day-by-day training + inference with tunable params."""
    predictor = LGBMPowerPredictor()
    inference = PowerInference(model_path=None)

    # Wrap inference feature_engineering for custom feature sets
    if feature_set != "baseline":
        _orig_fe = inference.feature_engineering
        if feature_set == "forecast_spread_features":
            def _wrapped_fe(df, _orig=_orig_fe):
                return _add_forecast_spread_features(_orig_fe(df))
            inference.feature_engineering = _wrapped_fe
        elif feature_set == "net_load_features":
            def _wrapped_fe(df, _orig=_orig_fe):
                return _add_net_load_features(_orig_fe(df))
            inference.feature_engineering = _wrapped_fe

    requested_start_date = pd.to_datetime(forecast_start)
    current_target_date = requested_start_date
    end_target_date = pd.to_datetime(forecast_end)

    # Load data once
    raw_df = predictor.load_and_process_data(data_path, target)

    all_days_preds: list[pd.DataFrame] = []

    while current_target_date <= end_target_date:
        target_day_str = current_target_date.strftime("%Y-%m-%d")
        decision_day_dt = current_target_date - datetime.timedelta(days=1)
        val_end_str = f"{decision_day_dt.strftime('%Y-%m-%d')} 14:00:00"
        val_start_str = (
            decision_day_dt - pd.DateOffset(months=int(training_months))
        ).strftime("%Y-%m-%d 01:00:00")

        best_res = None
        try:
            best_res = _p4_fit_window(
                predictor=predictor,
                data_path=data_path,
                history_start_date=val_start_str,
                history_end_date=val_end_str,
                target=target,
                raw_df=raw_df,
                val_ratio=val_ratio,
                sample_weight_profile=sample_weight_profile,
                objective=objective,
                alpha=alpha,
                period_params=period_params,
                feature_set=feature_set,
            )

            inference_start = f"{target_day_str} 01:00:00"
            inference_end = (
                current_target_date + datetime.timedelta(days=1)
            ).strftime("%Y-%m-%d 00:00:00")

            inference.model = best_res["model"]

            # Sync features_list for inference (important for custom feature sets)
            if "features_list" in best_res:
                inference.features_list = best_res["features_list"]

            # Single-day prediction
            day_result_df = inference.predict_range(
                data_path, inference_start, inference_end, target=target,
            )

            if day_result_df is not None:
                day_result_df["target_day"] = target_day_str
                all_days_preds.append(day_result_df)

        except Exception as e:
            print(f"  [ERROR] {target_day_str}: {e}")
            traceback.print_exc()

        current_target_date += datetime.timedelta(days=1)
        if best_res is not None:
            del best_res
        gc.collect()

    if all_days_preds:
        return pd.concat(all_days_preds, axis=0)
    return None


# ── Combo runner ──────────────────────────────────────────────────────

def run_combo(
    combo: dict[str, Any],
    data_path: str,
    start_date: str,
    end_date: str,
    out_dir_combo: Path,
    target: str = "实时电价",
    training_months: int = 12,
) -> dict[str, Any]:
    """Run a single combo and return metrics."""
    name = combo["name"]
    print(f"\n  ── Combo: {name} ──")
    print(f"      sw={combo['sample_weight_profile']}, "
          f"obj={combo['objective']}, "
          f"alpha={combo['alpha']}, "
          f"pp={combo['period_params']}, "
          f"fs={combo['feature_set']}")

    t0 = time.time()

    result_df = _p4_run_simulation(
        data_path=data_path,
        forecast_start=start_date,
        forecast_end=end_date,
        target=target,
        training_months=training_months,
        val_ratio=0.2,
        sample_weight_profile=combo["sample_weight_profile"],
        objective=combo["objective"],
        alpha=combo["alpha"],
        period_params=combo["period_params"],
        feature_set=combo["feature_set"],
    )

    elapsed = time.time() - t0

    if result_df is None or len(result_df) == 0:
        print(f"  [WARN] {name}: no predictions generated")
        return {"name": name, "error": "no predictions"}

    # Compute metrics
    metrics = compute_combo_metrics(result_df)
    metrics["runtime_seconds"] = round(elapsed, 1)
    metrics["name"] = name
    metrics["config"] = {
        "sample_weight_profile": combo["sample_weight_profile"],
        "objective": combo["objective"],
        "alpha": combo["alpha"],
        "period_params": combo["period_params"],
        "feature_set": combo["feature_set"],
        "direction": combo["direction"],
    }

    print(f"      sMAPE={metrics['smape_floor50']}, "
          f"severe={metrics['severe_count']}, "
          f"n={metrics['n_total']}, "
          f"runtime={elapsed:.0f}s")

    # Save combo metrics
    out_dir_combo.mkdir(parents=True, exist_ok=True)
    with open(out_dir_combo / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    if result_df is not None and len(result_df) > 0:
        result_df.to_csv(out_dir_combo / "predictions.csv", index=False)

    return metrics


# ── Ranking ───────────────────────────────────────────────────────────

def rank_combos(
    all_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Rank combos by composite score (normalized sMAPE + severe)."""
    valid = [m for m in all_metrics if m.get("smape_floor50") is not None]
    if len(valid) == 0:
        return []

    smape_vals = np.array([m["smape_floor50"] for m in valid])
    severe_vals = np.array([m["severe_count"] for m in valid])

    # Normalize to [0, 1]
    smape_norm = (smape_vals - smape_vals.min()) / max(smape_vals.max() - smape_vals.min(), 1e-10)
    severe_norm = (severe_vals - severe_vals.min()) / max(severe_vals.max() - severe_vals.min(), 1e-10)

    # Composite: equal weight
    composite = smape_norm + severe_norm

    ranked = sorted(
        zip(valid, composite),
        key=lambda x: x[1],
    )

    result = []
    for i, (m, comp) in enumerate(ranked):
        result.append({
            "rank": i + 1,
            "name": m["name"],
            "smape_floor50": m["smape_floor50"],
            "severe_count": m["severe_count"],
            "composite_score": round(float(comp), 4),
            "runtime_seconds": m.get("runtime_seconds"),
            "config": m.get("config", {}),
        })

    return result


# ── Report generation ─────────────────────────────────────────────────

def generate_tuning_report(
    small_window_results: list[dict[str, Any]],
    ranked: list[dict[str, Any]],
    full_window_results: list[dict[str, Any]] | None = None,
    out_dir: Path = OUT_DIR,
) -> str:
    """Generate the full P4 tuning report as markdown."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build summary JSON
    summary = {
        "small_window": {
            "start": SMALL_WINDOW_START,
            "end": SMALL_WINDOW_END,
            "n_combos": len(small_window_results),
            "results": small_window_results,
            "ranking": ranked,
        },
    }
    if full_window_results:
        summary["full_window"] = {
            "start": FULL_WINDOW_START,
            "end": FULL_WINDOW_END,
            "results": full_window_results,
        }

    with open(out_dir / "tuning_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Best config
    if ranked:
        best = ranked[0]
        best_config = {
            "rank": 1,
            "name": best["name"],
            "smape_floor50": best["smape_floor50"],
            "severe_count": best["severe_count"],
            "config": best["config"],
        }
        with open(out_dir / "best_model_config.json", "w", encoding="utf-8") as f:
            json.dump(best_config, f, indent=2, ensure_ascii=False)

    # ── Markdown report ──
    lines = []
    lines.append("# P4 LightGBM SOTA Tuning Report")
    lines.append("")
    lines.append(f"> Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> Branch: `agent/p4-lgbm-sota-tuning`")
    lines.append("")
    lines.append("## Tuning Directions")
    lines.append("")
    lines.append("| Direction | Values | Combos |")
    lines.append("|-----------|--------|--------|")
    lines.append("| sample_weight_profile | none, spike_weighted, severe_underestimate_weighted, period_spike_weighted | 4 |")
    lines.append("| objective | regression_l1, huber, quantile(0.7), quantile(0.8) | 4 |")
    lines.append("| period_params | 9_16_deep, peak_conservative (+ baseline) | 2 new |")
    lines.append("| feature_set | forecast_spread_features, net_load_features (+ baseline) | 2 new |")
    lines.append("| **Total** | | **12** |")
    lines.append("")
    lines.append("## Small Window Results (2025-11-01 ~ 2025-11-15)")
    lines.append("")
    lines.append("| Rank | Combo | Direction | sMAPE_floor50 | Severe | Composite | Runtime |")
    lines.append("|------|-------|-----------|:-------------:|:------:|:---------:|:-------:|")

    for entry in ranked:
        cfg = entry.get("config", {})
        direction = cfg.get("direction", "?")
        lines.append(
            f"| {entry['rank']} | {entry['name']} | {direction} "
            f"| {entry['smape_floor50']} | {entry['severe_count']} "
            f"| {entry['composite_score']} | {entry.get('runtime_seconds', '?')}s |"
        )

    lines.append("")

    # Targets
    best_smape = ranked[0]["smape_floor50"] if ranked else None
    best_severe = ranked[0]["severe_count"] if ranked else None

    lines.append("### Target Achievement (Single-Model)")
    lines.append("")
    lines.append(f"| Target | sMAPE ≤ 22.02 | Severe ≤ 80 | Verdict |")
    lines.append(f"|--------|:-------------:|:-----------:|:-------:|")
    smape_met = best_smape is not None and best_smape <= 22.02
    severe_met = best_severe is not None and best_severe <= 80
    best_name = ranked[0]["name"] if ranked else "N/A"
    if smape_met and severe_met:
        verdict = "**GO** ✅"
    elif smape_met or severe_met:
        verdict = "**CONDITIONAL** ⚠️"
    else:
        verdict = "**NO-GO** ❌"
    lines.append(f"| Best ({best_name}) | {best_smape} | {best_severe} | {verdict} |")
    lines.append("")

    # Full window results
    if full_window_results:
        lines.append("## Full Window Results (2025-11-01 ~ 2025-12-31)")
        lines.append("")
        lines.append("| Rank | Combo | sMAPE_floor50 | Severe | 9_16 sMAPE | n |")
        lines.append("|------|-------|:-------------:|:------:|:----------:|:--:|")

        for i, m in enumerate(full_window_results):
            lines.append(
                f"| {i + 1} | {m.get('name', '?')} "
                f"| {m.get('smape_floor50', '?')} | {m.get('severe_count', '?')} "
                f"| {m.get('9_16_smape_floor50', '?')} | {m.get('n_total', '?')} |"
            )
        lines.append("")

        # Strong GO check
        fw_best = full_window_results[0] if full_window_results else {}
        fw_smape = fw_best.get("smape_floor50")
        fw_severe = fw_best.get("severe_count")
        lines.append("### Strong GO Assessment")
        lines.append("")
        lines.append(f"| Target | sMAPE ≤ 20.86 | Severe ≤ 63 | Verdict |")
        lines.append(f"|--------|:-------------:|:-----------:|:-------:|")
        if fw_smape and fw_severe:
            s_met = fw_smape <= 20.86
            se_met = fw_severe <= 63
            if s_met and se_met:
                fw_verdict = "**GO** ✅"
            elif s_met or se_met:
                fw_verdict = "**CONDITIONAL** ⚠️"
            else:
                fw_verdict = "**NO-GO** ❌"
        else:
            fw_verdict = "N/A"
        lines.append(f"| Best | {fw_smape} | {fw_severe} | {fw_verdict} |")
        lines.append("")

    lines.append("## Configuration Details")
    lines.append("")
    lines.append("### 4 Tuning Directions")
    lines.append("")
    lines.append("**1. sample_weight_profile** — LightGBM internal sample weighting:")
    lines.append("- `none`: uniform weights (baseline)")
    lines.append("- `spike_weighted`: high-spike rows get 3×/6× weight")
    lines.append("- `severe_underestimate_weighted`: severe rows get 4× weight")
    lines.append("- `period_spike_weighted`: 9-16 spikes get 8× weight")
    lines.append("")
    lines.append("**2. Objective/Profile** — LightGBM loss function:")
    lines.append("- `regression`: L2 loss (baseline)")
    lines.append("- `regression_l1`: L1 loss (MAE)")
    lines.append("- `huber`: Huber loss (robust)")
    lines.append("- `quantile(α)`: Quantile regression at α=0.7, 0.8")
    lines.append("")
    lines.append("**3. Period-Specific Params** — Per-period LGBMRegressor hyperparams:")
    lines.append("- `baseline`: valley(2000, lr=0.05, lv=31), solar(3000, lr=0.03, lv=63), peak(3000, lr=0.03, lv=40)")
    lines.append("- `9_16_deep`: solar → (5000, lr=0.02, lv=127)")
    lines.append("- `peak_conservative`: peak → (1500, lr=0.02, lv=31)")
    lines.append("")
    lines.append("**4. Feature Set** — Input features:")
    lines.append("- `baseline`: 24 original features")
    lines.append("- `forecast_spread_features`: +5 features (local_power_ratio, nuclear_ratio, self_unit_ratio, renewable_ratio, forecast_spread)")
    lines.append("- `net_load_features`: +5 features (net_load MA 3/6, net_load change, volatility, acceleration)")
    lines.append("")

    # Best config details
    if ranked:
        best = ranked[0]
        lines.append("## Best Combo")
        lines.append("")
        lines.append(f"**{best['name']}**")
        lines.append("- sMAPE_floor50: " + str(best['smape_floor50']))
        lines.append("- Severe count: " + str(best['severe_count']))
        lines.append("- Config: " + json.dumps(best.get("config", {})))
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Generated by `scripts/run_p4_lgbm_sota_tuning.py`*")

    report = "\n".join(lines)

    # Also write to the markdown report path
    report_path = _PROJECT_ROOT / "docs" / "reports" / "P4_lgbm_sota_tuning_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


# ── Update execution board ────────────────────────────────────────────

def update_execution_board(
    ranked: list[dict[str, Any]],
    full_window_results: list[dict[str, Any]] | None = None,
) -> None:
    """Update docs/p3_execution_board.md with P4 W2 results."""
    board_path = _PROJECT_ROOT / "docs" / "p3_execution_board.md"
    if not board_path.exists():
        print(f"  [WARN] Board not found: {board_path}")
        return

    # Build the P4 W2 results section
    lines = []
    lines.append("### W2 Results")
    lines.append("")
    lines.append(f"| Date | Combo | sMAPE | Severe | Verdict |")
    lines.append(f"|------|-------|:-----:|:------:|:-------:|")

    if ranked:
        best = ranked[0]
        smape_met = best["smape_floor50"] is not None and best["smape_floor50"] <= 22.02
        severe_met = best["severe_count"] is not None and best["severe_count"] <= 80
        if smape_met and severe_met:
            sw_verdict = "GO ✅"
        elif smape_met or severe_met:
            sw_verdict = "CONDITIONAL ⚠️"
        else:
            sw_verdict = "NO-GO ❌"
        lines.append(
            f"| {SMALL_WINDOW_START}~{SMALL_WINDOW_END} (small) | {best['name']} "
            f"| {best['smape_floor50']} | {best['severe_count']} | {sw_verdict} |"
        )

    if full_window_results:
        fw = full_window_results[0]
        lines.append(
            f"| {FULL_WINDOW_START}~{FULL_WINDOW_END} (full) | {fw.get('name', '?')} "
            f"| {fw.get('smape_floor50', '?')} | {fw.get('severe_count', '?')} | see report |"
        )

    lines.append("")

    w2_section = "\n".join(lines)

    # Read existing board and find the W2 section
    with open(board_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Look for existing W2 Results section and replace, or append after "Results Log"
    marker = "### W2 Results"
    if marker in content:
        # Find the next ### or end of file, replace content
        start_idx = content.index(marker)
        next_section = content.find("###", start_idx + len(marker))
        if next_section == -1:
            next_section = len(content)
        content = content[:start_idx] + w2_section + content[next_section:]
    else:
        # Append before "Blockers" section
        blockers_marker = "## Blockers"
        if blockers_marker in content:
            content = content.replace(blockers_marker, w2_section + "\n" + blockers_marker)
        else:
            content += "\n" + w2_section

    with open(board_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  Updated board: {board_path}")


# ── Main ──────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P4 LightGBM SOTA hyperparameter tuning",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: run only 1 combo (sw_none) on 2-day window")
    parser.add_argument("--data-path", default=DATA_PATH,
                        help=f"Data file path (default: {DATA_PATH})")
    parser.add_argument("--out-dir", default=str(OUT_DIR),
                        help=f"Output directory (default: {OUT_DIR})")
    parser.add_argument("--small-start", default=SMALL_WINDOW_START)
    parser.add_argument("--small-end", default=SMALL_WINDOW_END)
    parser.add_argument("--full-start", default=FULL_WINDOW_START)
    parser.add_argument("--full-end", default=FULL_WINDOW_END)
    parser.add_argument("--skip-full", action="store_true",
                        help="Skip full-window eval (for debugging)")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve data path relative to project root if relative
    data_path = args.data_path
    if not Path(data_path).is_absolute():
        data_path = str(_PROJECT_ROOT / data_path)

    print("=" * 60)
    print("  P4 LightGBM SOTA Tuning")
    print("=" * 60)
    print(f"  Data: {data_path}")
    print(f"  Output: {out_dir}")

    # Build combo list
    if args.quick:
        combos = [{
            "name": "sw_none",
            "sample_weight_profile": "none",
            "objective": "regression",
            "alpha": None,
            "period_params": "baseline",
            "feature_set": "baseline",
            "direction": "sample_weight_profile",
        }]
        small_start = "2025-11-01"
        small_end = "2025-11-02"
        print(f"\n  [QUICK MODE] 1 combo, {small_start}~{small_end}")
    else:
        combos = build_combo_list()
        small_start = args.small_start
        small_end = args.small_end
        print(f"\n  Combos: {len(combos)}")
        print(f"  Small window: {small_start} ~ {small_end}")
        print(f"  Full window:  {args.full_start} ~ {args.full_end}")

    # ── Phase 1: Small window ──
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Small Window ({small_start} ~ {small_end})")
    print(f"{'='*60}")

    all_metrics: list[dict[str, Any]] = []

    for combo in combos:
        combo_out = out_dir / f"combo_{combo['name']}"
        try:
            metrics = run_combo(
                combo=combo,
                data_path=data_path,
                start_date=small_start,
                end_date=small_end,
                out_dir_combo=combo_out,
            )
            all_metrics.append(metrics)
        except Exception as e:
            print(f"  [FAIL] {combo['name']}: {e}")
            traceback.print_exc()
            all_metrics.append({"name": combo["name"], "error": str(e)})

    # Rank results
    ranked = rank_combos(all_metrics)
    print(f"\n  ── Ranking ──")
    print(f"  {'Rank':<6} {'Combo':<25} {'sMAPE':<8} {'Severe':<8} {'Dir':<22}")
    print(f"  {'-'*6} {'-'*25} {'-'*8} {'-'*8} {'-'*22}")
    for entry in ranked:
        cfg = entry.get("config", {})
        direction = cfg.get("direction", "?")
        print(f"  {entry['rank']:<6} {entry['name']:<25} "
              f"{entry['smape_floor50']:<8} {entry['severe_count']:<8} {direction:<22}")

    # ── Phase 2: Full window (top 3) ──
    full_results: list[dict[str, Any]] = []
    if not args.skip_full and not args.quick:
        top_n = min(3, len(ranked))
        top_names = [entry["name"] for entry in ranked[:top_n]]
        top_configs = [entry.get("config", {}) for entry in ranked[:top_n]]

        print(f"\n{'='*60}")
        print(f"  PHASE 2: Full Window ({args.full_start} ~ {args.full_end})")
        print(f"  Top {top_n} combos: {top_names}")
        print(f"{'='*60}")

        for i, (name, cfg) in enumerate(zip(top_names, top_configs)):
            combo_full = {
                "name": name,
                "sample_weight_profile": cfg.get("sample_weight_profile", "none"),
                "objective": cfg.get("objective", "regression"),
                "alpha": cfg.get("alpha"),
                "period_params": cfg.get("period_params", "baseline"),
                "feature_set": cfg.get("feature_set", "baseline"),
                "direction": cfg.get("direction", ""),
            }
            combo_out = out_dir / f"full_{name}"
            try:
                metrics = run_combo(
                    combo=combo_full,
                    data_path=data_path,
                    start_date=args.full_start,
                    end_date=args.full_end,
                    out_dir_combo=combo_out,
                )
                full_results.append(metrics)
            except Exception as e:
                print(f"  [FAIL] {name} (full): {e}")
                traceback.print_exc()
                full_results.append({"name": name, "error": str(e)})

        # Save best combo predictions
        if full_results and full_results[0].get("smape_floor50") is not None:
            best_name = full_results[0]["name"]
            best_pred_path = out_dir / f"full_{best_name}" / "predictions.csv"
            # Copy to top-level
            import shutil
            src = out_dir / f"full_{best_name}" / "predictions.csv"
            if src.exists():
                shutil.copy(src, out_dir / "predictions_top1.csv")
                print(f"  Saved best predictions: {out_dir / 'predictions_top1.csv'}")

    # ── Generate report ──
    print(f"\n{'='*60}")
    print(f"  Generating report...")
    print(f"{'='*60}")

    report = generate_tuning_report(
        small_window_results=all_metrics,
        ranked=ranked,
        full_window_results=full_results if full_results else None,
        out_dir=out_dir,
    )

    # Update execution board
    update_execution_board(
        ranked=ranked,
        full_window_results=full_results if full_results else None,
    )

    # Print summary
    print(f"\n  Report: {_PROJECT_ROOT / 'docs' / 'reports' / 'P4_lgbm_sota_tuning_report.md'}")
    print(f"  Board:  {_PROJECT_ROOT / 'docs' / 'p3_execution_board.md'}")

    if ranked:
        print(f"\n  Best combo: {ranked[0]['name']} "
              f"(sMAPE={ranked[0]['smape_floor50']}, severe={ranked[0]['severe_count']})")
        single_go = ranked[0]["smape_floor50"] <= 22.02 and ranked[0]["severe_count"] <= 80
        print(f"  Single-model GO: {'✅' if single_go else '❌'} "
              f"(sMAPE≤22.02, severe≤80)")

    if full_results:
        fw = full_results[0]
        strong_go = (fw.get("smape_floor50") is not None and fw["smape_floor50"] <= 20.86
                     and fw.get("severe_count") is not None and fw["severe_count"] <= 63)
        print(f"  Strong GO: {'✅' if strong_go else '❌'} "
              f"(sMAPE≤20.86, severe≤63)")

    print(f"\n  Output: {out_dir}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
