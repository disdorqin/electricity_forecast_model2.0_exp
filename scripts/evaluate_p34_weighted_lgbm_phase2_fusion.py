#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P3.4 Weighted-LightGBM + Phase2 Fusion/Correction.

Pipeline:
  1. Generate weighted LGBM predictions (period_spike_weighted) — or load cache
  2. Convert to prediction pack format (model_name="lightgbm_weighted")
  3. Build multi-candidate packs (3 fusion modes):
     - weighted_lgbm_anchor_90  — 0.9 * weighted + 0.1 * mean(others)
     - weighted_lgbm_anchor_80  — 0.8 * weighted + 0.2 * mean(others)
     - custom                   — 0.85*w_lgbm + 0.08*dayahead + 0.05*lag7 + 0.02*lag1
  4. Run correction (3 profiles × 3 fusion modes = 9 runs, all normal mode)
  5. Compute metrics and compare vs:
     - Phase2 champion:   sMAPE 20.86 / severe 63
     - Weighted LGBM:     sMAPE 23.76 / severe 54

Usage:
  python scripts/evaluate_p34_weighted_lgbm_phase2_fusion.py
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from extreme.realtime_high_spike.apply_correction import (
    CorrectionProfile,
    load_and_merge,
    run_correction,
    write_correction_manifest,
)
from extreme.realtime_high_spike.residual_lift import CorrectionMode
from lightGBM.infer_fix import PowerInference
from lightGBM.main_fix import (
    _split_history_train_val,
    VALLEY_HOURS,
    SOLAR_HOURS,
    PEAK_HOURS,
    validate_business_day_filled,
)
from lightGBM.train_fix import LGBMPowerPredictor, ThreeStageLGBM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("evaluate_p34_weighted_lgbm_phase2_fusion")

# ── Constants ─────────────────────────────────────────────────────────────
DATA_PATH = "data/shandong_pmos_hourly.xlsx"
START_DATE = "2025-11-01"
END_DATE = "2026-02-28"
OUT_DIR = Path("reports/local/p34_weighted_lgbm_phase2_fusion")
LEVEL0_PACK = (
    "reports/local/p0_full_run/prediction_pack_level0/"
    "prediction_pack_realtime_level0_2025_11_01_2026_02_28.csv"
)
RISK_PREDICTIONS = (
    "reports/local/p0_full_run/level0/risk_model/spike_risk_predictions.csv"
)

WEIGHTED_PROFILES = ("none", "spike_weighted", "severe_underestimate_weighted", "period_spike_weighted")
FUSION_MODES = ["weighted_lgbm_anchor_90", "weighted_lgbm_anchor_80", "custom"]
CORRECTION_PROFILES = ["conservative", "medium", "aggressive"]

PHASE2_CHAMPION = {"smape": 20.86, "severe": 63}
WEIGHTED_LGBM_STANDALONE = {"smape": 23.76, "severe": 54}

DEPLOY_GO = {"smape": 20.50, "severe": 63, "false_lift": 0.10, "degradation": 0.5}
RESEARCH_GO = {"smape": 20.00, "severe": 70, "false_lift": 0.12, "degradation": 1.0}


# ═══════════════════════════════════════════════════════════════════════════
# Part 1 — Spike weighting (from p33 branch)
# ═══════════════════════════════════════════════════════════════════════════

def _compute_spike_weights(
    train_df: pd.DataFrame,
    profile: str,
    y_col: str = "y_clipped",
    hour_col: str = "hour",
) -> np.ndarray:
    """Compute sample weights using only historical y_true (leakage-safe)."""
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
        raise ValueError(f"Unknown profile: {profile}. Choose from {WEIGHTED_PROFILES}")
    return weights


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — Weighted LightGBM daily walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def _fit_weighted_fixed_window(
    predictor,
    data_path,
    history_start_date,
    history_end_date,
    target,
    raw_df=None,
    val_ratio=0.2,
    sample_weight_profile: str | None = None,
):
    """Modified _fit_realtime_fixed_window with spike weighting support."""
    import lightgbm as lgb
    from sklearn.metrics import mean_absolute_error

    raw_df = raw_df.copy() if raw_df is not None else predictor.load_and_process_data(data_path, target)
    full_df = predictor.feature_engineering(raw_df)
    history_start_dt = pd.to_datetime(history_start_date)
    history_end_dt = pd.to_datetime(history_end_date)
    history_mask = (full_df["ds"] >= history_start_dt) & (full_df["ds"] <= history_end_dt)
    history_df = full_df[history_mask].copy()
    if len(history_df) < 2000:
        raise RuntimeError("Realtime LightGBM fixed-window training set is too small.")

    train_df, test_df_raw = _split_history_train_val(history_df, val_ratio=val_ratio)
    predictor.validate_optimize_dataset(
        test_df_raw,
        str(test_df_raw["ds"].min()),
        str(test_df_raw["ds"].max()),
    )

    train_upper = train_df["y"].quantile(0.995)
    train_df["y_clipped"] = train_df["y"].clip(lower=-100, upper=train_upper)

    use_profile_weights = sample_weight_profile is not None

    # ── Valley ──
    train_valley = train_df[train_df["hour"].isin(VALLEY_HOURS)]
    test_valley = test_df_raw[test_df_raw["hour"].isin(VALLEY_HOURS)]
    w_valley = _compute_spike_weights(train_valley, sample_weight_profile) if use_profile_weights else np.ones(len(train_valley))
    model_valley_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(
            objective="regression", n_estimators=2000, learning_rate=0.05,
            num_leaves=31, n_jobs=predictor.lgbm_n_jobs,
            device_type=predictor._device_type(), verbose=-1, random_state=42,
        ),
        train_valley[predictor.features_list],
        train_valley["y_clipped"],
        sample_weight=w_valley,
        eval_set=[(test_valley[predictor.features_list], test_valley["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Solar ──
    train_solar = train_df[train_df["hour"].isin(SOLAR_HOURS)]
    test_solar = test_df_raw[test_df_raw["hour"].isin(SOLAR_HOURS)]
    if use_profile_weights:
        w_solar = _compute_spike_weights(train_solar, sample_weight_profile)
    else:
        w_solar = np.ones(len(train_solar))
        y_solar_val = train_solar["y_clipped"].values
        w_solar[y_solar_val < 50] = 2
        w_solar[y_solar_val < 0] = 5
    model_solar_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(
            objective="regression", n_estimators=3000, learning_rate=0.03,
            num_leaves=63, n_jobs=predictor.lgbm_n_jobs,
            device_type=predictor._device_type(), verbose=-1, random_state=42,
        ),
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
            objective="binary", n_estimators=1000, learning_rate=0.05,
            class_weight="balanced", n_jobs=predictor.lgbm_n_jobs,
            device_type=predictor._device_type(), verbose=-1, random_state=42,
        ),
        train_solar[predictor.features_list],
        y_solar_class,
        eval_set=[(test_solar[predictor.features_list], y_solar_test_class)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Peak ──
    train_peak = train_df[train_df["hour"].isin(PEAK_HOURS)]
    test_peak = test_df_raw[test_df_raw["hour"].isin(PEAK_HOURS)]
    if use_profile_weights:
        w_peak = _compute_spike_weights(train_peak, sample_weight_profile)
    else:
        w_peak = np.ones(len(train_peak))
        high_wind_threshold = train_peak["wind"].quantile(0.8)
        w_peak[train_peak["wind"] > high_wind_threshold] = 3
    model_peak_reg = predictor._fit_with_cuda_fallback(
        lgb.LGBMRegressor(
            objective="regression", n_estimators=3000, learning_rate=0.03,
            num_leaves=40, n_jobs=predictor.lgbm_n_jobs,
            device_type=predictor._device_type(), verbose=-1, random_state=42,
        ),
        train_peak[predictor.features_list],
        train_peak["y_clipped"],
        sample_weight=w_peak,
        eval_set=[(test_peak[predictor.features_list], test_peak["y"])],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )

    combined_model = ThreeStageLGBM(model_valley_reg, model_solar_reg, model_solar_clf, model_peak_reg)
    pred = combined_model.predict(test_df_raw[predictor.features_list])
    pred = np.where(pred < -80, -80, pred)
    mae = float(np.mean(np.abs(test_df_raw["y"] - pred)))  # noqa: NPY201
    smape = predictor.calculate_smape(test_df_raw["y"], pred)
    predictor.save_model(
        [{"months_back": 12, "train_start": history_start_dt, "mae": mae, "smape": smape, "model": combined_model}],
        target,
    )
    return {"months_back": 12, "mae": mae, "smape": smape, "model": combined_model}


def run_weighted_precision_simulation(
    data_path,
    forecast_start,
    forecast_end,
    target="实时电价",
    use_predicted_temp=False,
    training_months=12,
    val_ratio=0.2,
    sample_weight_profile: str | None = None,
) -> pd.DataFrame | None:
    """Daily walk-forward with spike-weighted training.

    Modified from run_precision_simulation to pass sample_weight_profile
    through to _fit_weighted_fixed_window.
    """
    predictor = LGBMPowerPredictor()
    inference = PowerInference(model_path=None)
    requested_start_date = pd.to_datetime(forecast_start)
    current_target_date = requested_start_date
    end_target_date = pd.to_datetime(forecast_end)

    all_days_preds = []
    while current_target_date <= end_target_date:
        target_day_str = current_target_date.strftime("%Y-%m-%d")
        decision_day_dt = current_target_date - pd.Timedelta(days=1)
        val_end_str = decision_day_dt.strftime("%Y-%m-%d 14:00:00")
        val_start_str = (decision_day_dt - pd.DateOffset(months=int(training_months))).strftime("%Y-%m-%d 01:00:00")

        best_res = None
        try:
            best_res = _fit_weighted_fixed_window(
                predictor=predictor,
                data_path=data_path,
                history_start_date=val_start_str,
                history_end_date=val_end_str,
                target=target,
                raw_df=None,
                val_ratio=val_ratio,
                sample_weight_profile=sample_weight_profile,
            )
            inference_start = current_target_date.strftime("%Y-%m-%d 01:00:00")
            inference_end = (current_target_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
            inference.model = best_res["model"]

            day_result_df = inference.predict_range(
                data_path, inference_start, inference_end, target=target
            )

            if day_result_df is not None and not day_result_df.empty:
                day_result_df["target_day"] = target_day_str
                day_result_df["best_window"] = int(training_months)
                all_days_preds.append(day_result_df)

        except Exception as e:
            logger.error("[%s] weighted walk-forward failed: %s", target_day_str, e, exc_info=True)

        current_target_date += datetime.timedelta(days=1)
        if best_res is not None:
            del best_res
        gc.collect()

    if all_days_preds:
        return pd.concat(all_days_preds, axis=0)
    return None


def generate_weighted_predictions(
    data_path: str,
    start_date: str,
    end_date: str,
    profile: str = "period_spike_weighted",
    cache_dir: Path | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Generate or load cached weighted LightGBM predictions."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"predictions_{profile}.csv"
        if cache_path.exists() and not force:
            logger.info("Loading cached predictions from %s", cache_path)
            return pd.read_csv(cache_path)
        elif cache_path.exists():
            logger.info("Forced regeneration — removing cache at %s", cache_path)

    logger.info(
        "Generating weighted LGBM predictions (profile=%s, %s to %s)",
        profile, start_date, end_date,
    )
    t0 = time.time()
    result = run_weighted_precision_simulation(
        data_path=data_path,
        forecast_start=start_date,
        forecast_end=end_date,
        target="实时电价",
        sample_weight_profile=profile,
    )
    elapsed = time.time() - t0
    if result is None or result.empty:
        raise RuntimeError("Weighted LightGBM produced no predictions.")

    logger.info("Generated %d rows in %.1fs", len(result), elapsed)

    if cache_dir is not None:
        result.to_csv(cache_path, index=False, encoding="utf-8-sig")
        logger.info("Cached to %s", cache_path)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Part 3 — Prediction pack conversion
# ═══════════════════════════════════════════════════════════════════════════

def convert_to_pack_format(
    pred_df: pd.DataFrame,
    model_name: str = "lightgbm_weighted",
    source: str = "lightgbm_weighted_p34",
) -> pd.DataFrame:
    """Convert raw LightGBM output → prediction pack rows."""
    rows = []
    for _, row in pred_df.iterrows():
        ds = pd.to_datetime(row["ds"])
        hour_business = int(ds.hour) if ds.hour != 0 else 24
        business_day = ds.normalize() - pd.Timedelta(hours=1) if hour_business == 24 else ds.normalize()
        business_day_str = business_day.strftime("%Y-%m-%d")

        y_true = float(row.get("y", float("nan")))
        y_pred = float(row.get("pred_y", float("nan")))

        # Period mapping
        if 9 <= hour_business <= 16:
            period = "9_16"
        elif 1 <= hour_business <= 8:
            period = "night"
        else:
            period = "evening"

        residual = y_true - y_pred if not (np.isnan(y_true) or np.isnan(y_pred)) else 0.0
        abs_error = abs(residual)
        smape = _smape_floor50_scalar(y_true, y_pred)

        high_spike = 1 if (not np.isnan(y_true) and not np.isnan(y_pred) and abs(y_true - y_pred) > 200) else 0
        severe = 1 if (not np.isnan(y_true) and not np.isnan(y_pred) and y_true - y_pred > 200) else 0

        rows.append({
            "business_day": business_day_str,
            "hour_business": hour_business,
            "timestamp": str(ds),
            "period": period,
            "target": "realtime",
            "model_name": model_name,
            "y_pred": round(y_pred, 4),
            "base_fused_pred": round(y_pred, 4),
            "final_pred": round(y_pred, 4),
            "y_true": round(y_true, 4),
            "residual": round(residual, 4),
            "abs_error": round(abs_error, 4),
            "smape_floor50": round(smape, 4),
            "high_spike_flag": high_spike,
            "severe_underestimate_flag": severe,
            "source_file": source,
            "coverage_status": "available",
        })

    return pd.DataFrame(rows)


def _smape_floor50_scalar(y_true, y_pred):
    if np.isnan(y_true) or np.isnan(y_pred):
        return float("nan")
    yt = max(abs(y_true), 50.0)
    yp = max(abs(y_pred), 50.0)
    denom = (yt + yp) / 2.0
    if denom < 1e-10:
        return 0.0
    return min(abs(yt - yp) / denom * 100.0, 50.0)


# ═══════════════════════════════════════════════════════════════════════════
# Part 4 — Multi-candidate pack builder (weighted_lgbm anchored)
# ═══════════════════════════════════════════════════════════════════════════

def compute_smape_floor50(y_true, y_pred):
    yt = np.maximum(np.abs(y_true), 50.0)
    yp = np.maximum(np.abs(y_pred), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100.0, 0.0)
    return np.minimum(smape, 50.0)


def load_baseline_models(path: str | Path) -> pd.DataFrame:
    """Load baseline model rows (naive_lag1, naive_lag7, dayahead_proxy)."""
    baseline_models = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
    df = pd.read_csv(path)
    df = df[df["model_name"].isin(baseline_models)].copy()
    return df


def build_fused_pack(
    weighted_lgbm: pd.DataFrame,
    baselines: pd.DataFrame,
    anchor_model: str = "lightgbm_weighted",
    fusion_mode: str = "weighted_lgbm_anchor_90",
    custom_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build multi-candidate pack with weighted_lgbm anchored fusion.

    Fusion modes:
      - weighted_lgbm_anchor_90: 0.9 * weighted + 0.1 * mean(others)
      - weighted_lgbm_anchor_80: 0.8 * weighted + 0.2 * mean(others)
      - custom:                  explicit per-model weights
    """
    # Drop derived columns from both inputs
    drop_cols = ["base_fused_pred", "final_pred", "residual", "abs_error",
                 "smape_floor50", "high_spike_flag", "severe_underestimate_flag"]
    for df in (weighted_lgbm, baselines):
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    # Combine
    combined = pd.concat([baselines, weighted_lgbm], ignore_index=True)

    # Compute fused base
    if fusion_mode == "weighted_lgbm_anchor_90":
        anchor_w = 0.9
        fused = _fuse_anchor(combined, anchor_model, anchor_w)
    elif fusion_mode == "weighted_lgbm_anchor_80":
        anchor_w = 0.8
        fused = _fuse_anchor(combined, anchor_model, anchor_w)
    elif fusion_mode == "custom":
        fused = _fuse_custom(combined, custom_weights or {})
    else:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")

    fused = fused.reset_index()
    fused.columns = ["business_day", "hour_business", "base_fused_pred"]

    # Merge back
    pack = combined.merge(fused, on=["business_day", "hour_business"], how="left")
    pack["final_pred"] = pack["base_fused_pred"]
    pack["residual"] = pack["y_true"] - pack["base_fused_pred"]
    pack["abs_error"] = pack["residual"].abs()
    pack["smape_floor50"] = compute_smape_floor50(pack["y_true"], pack["base_fused_pred"])

    pack["high_spike_flag"] = ((pack["y_true"] - pack["base_fused_pred"]).abs() > 200).astype(int)
    pack["severe_underestimate_flag"] = (pack["y_true"] - pack["base_fused_pred"] > 200).astype(int)
    pack["source_file"] = f"multicandidate_{fusion_mode}"
    pack["coverage_status"] = "available"
    pack["target"] = "realtime"

    if "period" not in pack.columns:
        pack["period"] = pack["hour_business"].apply(_get_period)
    else:
        pack["period"] = pack["period"].fillna(pack["hour_business"].apply(_get_period))

    pack = pack.sort_values(["business_day", "hour_business", "model_name"]).reset_index(drop=True)

    # Select & order columns
    out_cols = [
        "business_day", "hour_business", "timestamp", "period",
        "target", "model_name", "y_pred", "base_fused_pred",
        "final_pred", "y_true", "residual", "abs_error",
        "smape_floor50", "high_spike_flag",
        "severe_underestimate_flag", "source_file", "coverage_status",
    ]
    for c in out_cols:
        if c not in pack.columns:
            pack[c] = None
    return pack[out_cols]


def _fuse_anchor(df: pd.DataFrame, anchor_model: str, anchor_weight: float) -> pd.Series:
    """Generic anchored fusion: anchor_weight * anchor + (1-anchor_weight) * mean(others)."""
    anchor = df[df["model_name"] == anchor_model].set_index(["business_day", "hour_business"])["y_pred"]
    others = df[df["model_name"] != anchor_model]
    other_mean = others.groupby(["business_day", "hour_business"])["y_pred"].mean()
    idx = other_mean.index.union(anchor.index)
    anchor = anchor.reindex(idx)
    other_mean = other_mean.reindex(idx)
    return anchor_weight * anchor.fillna(other_mean) + (1 - anchor_weight) * other_mean.fillna(anchor)


def _fuse_custom(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Weighted average with explicit per-model weights."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to > 0")
    norm = {k: v / total for k, v in weights.items()}
    results = []
    for (bd, hb), grp in df.groupby(["business_day", "hour_business"]):
        avail = grp.set_index("model_name")["y_pred"]
        w_sum = 0.0
        w_used = 0.0
        for model, w in norm.items():
            if model in avail.index and not pd.isna(avail[model]):
                w_sum += w * avail[model]
                w_used += w
        fused_val = w_sum / w_used if w_used > 0 else float("nan")
        results.append({"business_day": bd, "hour_business": hb, "fused": fused_val})
    return pd.DataFrame(results).set_index(["business_day", "hour_business"])["fused"]


def _get_period(hour: int) -> str:
    if 9 <= hour <= 16:
        return "9_16"
    elif 1 <= hour <= 8:
        return "night"
    return "evening"


def build_risk_predictions(
    pack: pd.DataFrame,
    source_risk_path: str | Path,
) -> pd.DataFrame:
    """Build risk predictions (1 row per timestamp)."""
    source = pd.read_csv(source_risk_path)
    risk_map = source.groupby(["business_day", "hour_business"])["spike_risk_score"].max().reset_index()
    timestamps = pack[["business_day", "hour_business", "timestamp", "period"]].drop_duplicates().copy()
    risk = timestamps.merge(risk_map, on=["business_day", "hour_business"], how="left")
    risk["spike_risk_score"] = risk["spike_risk_score"].fillna(0.0)
    risk["high_spike_prob"] = risk["spike_risk_score"]
    risk["spike_risk_flag"] = (risk["spike_risk_score"] > 0.8).astype(int)
    return risk


# ═══════════════════════════════════════════════════════════════════════════
# Part 5 — Metrics & evaluation
# ═══════════════════════════════════════════════════════════════════════════

def dedup_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=["business_day", "hour_business"]).copy()


def compute_ts_metrics(ts_df: pd.DataFrame, label: str) -> dict[str, Any]:
    """Compute timestamp-level metrics."""
    smape = float(np.nanmean(compute_smape_floor50(ts_df["y_true"], ts_df["final_pred"])))
    base_smape = float(np.nanmean(compute_smape_floor50(ts_df["y_true"], ts_df["base_fused_pred"])))
    severe = int((ts_df["y_true"] - ts_df["final_pred"] > 200).sum())
    severe_base = int((ts_df["y_true"] - ts_df["base_fused_pred"] > 200).sum())

    # 9_16
    mask_9 = ts_df["period"] == "9_16"
    smape_9_16 = float(np.nanmean(compute_smape_floor50(ts_df.loc[mask_9, "y_true"], ts_df.loc[mask_9, "final_pred"]))) if mask_9.sum() > 0 else float("nan")
    base_smape_9_16 = float(np.nanmean(compute_smape_floor50(ts_df.loc[mask_9, "y_true"], ts_df.loc[mask_9, "base_fused_pred"]))) if mask_9.sum() > 0 else float("nan")

    # Normal hours
    normal_mask = ts_df["high_spike_flag"] == 0
    if normal_mask.sum() > 0:
        nb = float(np.nanmean(compute_smape_floor50(ts_df.loc[normal_mask, "y_true"], ts_df.loc[normal_mask, "base_fused_pred"])))
        na = float(np.nanmean(compute_smape_floor50(ts_df.loc[normal_mask, "y_true"], ts_df.loc[normal_mask, "final_pred"])))
        normal_degradation = round(na - nb, 4)
        false_lift_mask = normal_mask & (ts_df["final_pred"] > ts_df["base_fused_pred"])
        false_lift_rate = false_lift_mask.sum() / max(normal_mask.sum(), 1)
    else:
        nb, na = float("nan"), float("nan")
        normal_degradation = float("nan")
        false_lift_rate = 0.0

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
        "normal_hours_before": round(nb, 4),
        "normal_hours_after": round(na, 4),
        "normal_hours_degradation": normal_degradation,
        "false_lift_rate": round(false_lift_rate, 4),
        "lift_applied_count": int((ts_df["final_pred"] != ts_df["base_fused_pred"]).sum()),
    }


def compute_go_nogo(metrics: dict) -> str:
    """Apply unified P3.4 GO criteria."""
    s = metrics["smape"]
    sev = metrics["severe_underestimate"]
    fl = metrics["false_lift_rate"]
    deg = metrics["normal_hours_degradation"]

    # DEPLOY GO
    if s <= DEPLOY_GO["smape"] and sev <= DEPLOY_GO["severe"] and fl <= DEPLOY_GO["false_lift"] and deg <= DEPLOY_GO["degradation"]:
        return "DEPLOY GO"

    # RESEARCH GO
    if s <= RESEARCH_GO["smape"] and sev <= RESEARCH_GO["severe"] and fl <= RESEARCH_GO["false_lift"] and deg <= RESEARCH_GO["degradation"]:
        return "RESEARCH GO"

    return "NO-GO"


# ═══════════════════════════════════════════════════════════════════════════
# Part 6 — Correction wrapper
# ═══════════════════════════════════════════════════════════════════════════

def run_correction_eval(
    pack_path: str | Path,
    risk_path: str | Path,
    profile_name: str,
    out_dir: str | Path,
    corr_mode: CorrectionMode = CorrectionMode.NORMAL,
) -> dict[str, Any] | None:
    """Run correction on a fused pack and compute metrics."""
    profile = CorrectionProfile(
        name=profile_name,
        mode=corr_mode,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = run_correction(
            prediction_pack_path=pack_path,
            risk_predictions_path=risk_path,
            profile=profile,
        )

        # Save correction result
        result.to_csv(out_dir / "correction_result.csv", index=False, encoding="utf-8-sig")

        # Compute metrics
        ts_df = dedup_timestamp(result)
        if "abs_error" not in ts_df.columns:
            ts_df["abs_error"] = (ts_df["y_true"] - ts_df["final_pred"]).abs()
        if "high_spike_flag" not in ts_df.columns:
            ts_df["high_spike_flag"] = ((ts_df["y_true"] - ts_df["base_fused_pred"]).abs() > 200).astype(int)
        if "period" not in ts_df.columns:
            ts_df["period"] = ts_df["hour_business"].apply(_get_period)

        label = f"{Path(pack_path).parent.name}/{profile_name}"
        metrics = compute_ts_metrics(ts_df, label)
        metrics["verdict"] = compute_go_nogo(metrics)

        write_correction_manifest(out_dir, profile, metrics=metrics)

        return metrics

    except Exception as e:
        logger.error("Correction failed for %s/%s: %s", Path(pack_path).parent.name, profile_name, e)
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Part 7 — Main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="P3.4 Weighted-LightGBM + Phase2 Fusion/Correction")
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--start-date", default=START_DATE)
    parser.add_argument("--end-date", default=END_DATE)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--level0-pack", default=LEVEL0_PACK)
    parser.add_argument("--risk-predictions", default=RISK_PREDICTIONS)
    parser.add_argument("--weight-profile", default="period_spike_weighted", choices=WEIGHTED_PROFILES)
    parser.add_argument("--skip-generation", action="store_true", help="Skip weighted LGBM prediction generation")
    parser.add_argument("--force", action="store_true", help="Force regeneration of cached predictions")
    args = parser.parse_args()

    t_start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  P3.4 Weighted-LightGBM + Phase2 Fusion/Correction")
    print(f"{'='*70}")

    # ── Step 1: Generate weighted LGBM predictions ──
    print(f"\n  [1/5] Generating weighted LGBM predictions (profile={args.weight_profile})...")

    if args.skip_generation:
        # Check if cached predictions exist
        cache = out_dir / "predictions" / f"predictions_{args.weight_profile}.csv"
        if not cache.exists():
            print(f"  [ERR] --skip-generation set but no cache at {cache}")
            sys.exit(1)
        raw_preds = pd.read_csv(cache)
        print(f"    Loaded {len(raw_preds)} cached predictions from {cache}")
    else:
        raw_preds = generate_weighted_predictions(
            data_path=args.data_path,
            start_date=args.start_date,
            end_date=args.end_date,
            profile=args.weight_profile,
            cache_dir=out_dir / "predictions",
            force=args.force,
        )

    # ── Step 2: Convert to pack format ──
    print(f"\n  [2/5] Converting to pack format...")
    wlgbm_pack = convert_to_pack_format(raw_preds)
    weights_dir = out_dir / "packs" / "weighted_lgbm"
    weights_dir.mkdir(parents=True, exist_ok=True)
    wlgbm_pack.to_csv(weights_dir / "lightgbm_weighted_pack.csv", index=False, encoding="utf-8-sig")
    print(f"    {len(wlgbm_pack)} rows from weighted LGBM")

    # ── Step 3: Build multi-candidate packs ──
    print(f"\n  [3/5] Building multi-candidate packs...")
    baselines = load_baseline_models(args.level0_pack)
    print(f"    Loaded {len(baselines)} baseline rows")

    pack_results = {}
    for fm in FUSION_MODES:
        custom_w = None
        if fm == "custom":
            custom_w = {"lightgbm_weighted": 0.85, "dayahead_proxy": 0.08, "naive_lag7": 0.05, "naive_lag1": 0.02}

        pack = build_fused_pack(wlgbm_pack, baselines, fusion_mode=fm, custom_weights=custom_w)
        pack_dir = out_dir / "packs" / fm
        pack_dir.mkdir(parents=True, exist_ok=True)
        pack_path = pack_dir / f"prediction_pack_realtime_multicandidate_{args.start_date.replace('-', '_')}_{args.end_date.replace('-', '_')}.csv"
        pack.to_csv(pack_path, index=False, encoding="utf-8-sig")

        # Risk predictions
        risk = build_risk_predictions(pack, args.risk_predictions)
        risk_path = pack_dir / "risk_predictions_multicandidate.csv"
        risk.to_csv(risk_path, index=False, encoding="utf-8-sig")

        pack_results[fm] = {"pack_path": pack_path, "risk_path": risk_path}
        print(f"    {fm}: {len(pack)} rows, {len(risk)} risk rows")

    # ── Step 4: Run correction ──
    print(f"\n  [4/5] Running correction (3 profiles × {len(FUSION_MODES)} fusion modes)...")
    all_metrics = []

    for fm in FUSION_MODES:
        pp = pack_results[fm]["pack_path"]
        rp = pack_results[fm]["risk_path"]

        for profile_name in CORRECTION_PROFILES:
            corr_out = out_dir / "correction" / fm / "normal" / profile_name
            metrics = run_correction_eval(pp, rp, profile_name, corr_out)
            if metrics is not None:
                metrics["fusion_mode"] = fm
                all_metrics.append(metrics)

            status = f"{metrics['verdict']:>15}" if metrics else "  FAILED  "
            smape_str = f"{metrics['smape']:.2f}" if metrics else "N/A"
            severe_str = str(metrics['severe_underestimate']) if metrics else "N/A"
            fl_str = f"{metrics['false_lift_rate']:.1%}" if metrics else "N/A"
            print(f"    {fm:>30}/{profile_name:15}: "
                  f"sMAPE={smape_str}, "
                  f"severe={severe_str}, "
                  f"false_lift={fl_str} | {status}")

    # ── Step 5: Compute comparison + generate report ──
    print(f"\n  [5/5] Generating report...")
    _write_report(out_dir, all_metrics, args, time.time() - t_start)

    print(f"\n{'='*70}")
    print(f"  Done. Results in {out_dir}")
    print(f"{'='*70}\n")


def _write_report(out_dir: Path, all_metrics: list[dict], args, elapsed: float):
    """Write comparison report and summary JSON."""
    best = None
    for m in all_metrics:
        if best is None or (m["verdict"] != "NO-GO" and best["verdict"] == "NO-GO"):
            best = m
        elif m["verdict"] == best["verdict"] and m["smape"] < best["smape"]:
            best = m

    # Summary JSON
    summary = {
        "pipeline": "P3.4 Weighted-LightGBM + Phase2 Fusion/Correction",
        "weight_profile": args.weight_profile,
        "fusion_modes_tested": FUSION_MODES,
        "correction_profiles_tested": CORRECTION_PROFILES,
        "n_configurations": len(all_metrics),
        "comparison": {
            "phase2_champion": PHASE2_CHAMPION,
            "weighted_lgbm_standalone": WEIGHTED_LGBM_STANDALONE,
        },
        "results": [],
        "best_result": None,
        "runtime_seconds": round(elapsed, 1),
    }

    for m in sorted(all_metrics, key=lambda x: (x["verdict"] != "DEPLOY GO", x["verdict"] != "RESEARCH GO", x["smape"])):
        summary["results"].append({
            "fusion_mode": m["fusion_mode"],
            "correction_profile": m["label"].split("/")[-1],
            "verdict": m["verdict"],
            "smape": m["smape"],
            "base_smape": m["base_smape"],
            "smape_9_16": m["smape_9_16"],
            "severe_underestimate": m["severe_underestimate"],
            "severe_delta": m["severe_delta"],
            "normal_hours_degradation": m["normal_hours_degradation"],
            "false_lift_rate": m["false_lift_rate"],
            "lift_applied_count": m["lift_applied_count"],
        })

    if best:
        summary["best_result"] = {
            "fusion_mode": best["fusion_mode"],
            "correction_profile": best["label"].split("/")[-1],
            "verdict": best["verdict"],
            "smape": best["smape"],
            "severe": best["severe_underestimate"],
            "false_lift": best["false_lift_rate"],
            "degradation": best["normal_hours_degradation"],
        }

    json_path = out_dir / "p34_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Markdown report ──
    lines = [
        "# P3.4 Weighted-LightGBM + Phase2 Fusion/Correction Report",
        "",
        "## Summary",
        "",
        f"**Weight profile**: {args.weight_profile}",
        f"**Date range**: {args.start_date} → {args.end_date}",
        f"**Runtime**: {elapsed:.1f}s",
        f"**Configurations evaluated**: {len(all_metrics)}",
        "",
        "## Comparison Baselines",
        "",
        "| Candidate | sMAPE | Severe |",
        "|-----------|-------|--------|",
        f"| Phase2 champion (lightgbm_anchor_90 + medium normal) | {PHASE2_CHAMPION['smape']} | {PHASE2_CHAMPION['severe']} |",
        f"| Weighted LGBM standalone (period_spike_weighted) | {WEIGHTED_LGBM_STANDALONE['smape']} | {WEIGHTED_LGBM_STANDALONE['severe']} |",
        "",
        "## All Results",
        "",
        "| # | Fusion | Correction | sMAPE | Base | 9_16 | Severe | ΔSev | Degrad | False Lift | Lift | Verdict |",
        "|---|--------|------------|-------|------|------|--------|------|--------|------------|------|---------|",
    ]

    for i, m in enumerate(sorted(all_metrics, key=lambda x: (x["verdict"] != "DEPLOY GO", x["verdict"] != "RESEARCH GO", x["smape"])), 1):
        parts = m["label"].split("/")
        lines.append(
            f"| {i} | {m['fusion_mode']} | {parts[-1]} "
            f"| {m['smape']} | {m['base_smape']} | {m['smape_9_16']} "
            f"| {m['severe_underestimate']} | {m['severe_delta']:+d} "
            f"| {m['normal_hours_degradation']:+.2f} "
            f"| {m['false_lift_rate']:.1%} | {m['lift_applied_count']} | {m['verdict']} |"
        )

    # Verdict distribution
    verdicts = {}
    for m in all_metrics:
        v = m["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1

    lines += [
        "",
        "## Verdict Distribution",
        "",
    ]
    for v, count in sorted(verdicts.items()):
        lines.append(f"- **{v}**: {count} configurations")

    if best:
        lines += [
            "",
            "## Best Candidate",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Fusion | {best['fusion_mode']} |",
            f"| Correction | {best['label'].split('/')[-1]} |",
            f"| sMAPE | {best['smape']} |",
            f"| Severe | {best['severe_underestimate']} |",
            f"| False Lift | {best['false_lift_rate']:.1%} |",
            f"| Degradation | {best['normal_hours_degradation']:+.2f} |",
            f"| Verdict | {best['verdict']} |",
            "",
            "## GO/NO-GO Assessment",
            "",
            f"| Criterion | DEPLOY GO | RESEARCH GO | Best | Met? |",
            f"|-----------|-----------|-------------|------|------|",
            f"| sMAPE | ≤ {DEPLOY_GO['smape']} | ≤ {RESEARCH_GO['smape']} | {best['smape']} | {'✅' if best['smape'] <= RESEARCH_GO['smape'] else '❌'} |",
            f"| Severe | ≤ {DEPLOY_GO['severe']} | ≤ {RESEARCH_GO['severe']} | {best['severe_underestimate']} | {'✅' if best['severe_underestimate'] <= RESEARCH_GO['severe'] else '❌'} |",
            f"| False Lift | ≤ {DEPLOY_GO['false_lift']:.0%} | ≤ {RESEARCH_GO['false_lift']:.0%} | {best['false_lift_rate']:.1%} | {'✅' if best['false_lift_rate'] <= RESEARCH_GO['false_lift'] else '❌'} |",
            f"| Degradation | ≤ {DEPLOY_GO['degradation']} | ≤ {RESEARCH_GO['degradation']} | {best['normal_hours_degradation']:+.2f} | {'✅' if best['normal_hours_degradation'] <= RESEARCH_GO['degradation'] else '❌'} |",
        ]

    # Comparison
    lines += [
        "",
        "## Comparison vs Baselines",
        "",
        "| Candidate | sMAPE | Severe | Δ sMAPE vs Phase2 | Δ Severe vs Phase2 |",
        "|-----------|-------|--------|-------------------|--------------------|",
        f"| Phase2 champion | {PHASE2_CHAMPION['smape']} | {PHASE2_CHAMPION['severe']} | — | — |",
        f"| Weighted LGBM standalone | {WEIGHTED_LGBM_STANDALONE['smape']} | {WEIGHTED_LGBM_STANDALONE['severe']} | +{WEIGHTED_LGBM_STANDALONE['smape'] - PHASE2_CHAMPION['smape']:.2f} ❌ | -{PHASE2_CHAMPION['severe'] - WEIGHTED_LGBM_STANDALONE['severe']} ✅ |",
    ]

    if best:
        d_smape = round(best['smape'] - PHASE2_CHAMPION['smape'], 2)
        d_sev = best['severe_underestimate'] - PHASE2_CHAMPION['severe']
        lines.append(
            f"| **P3.4 best** ({best['fusion_mode']} / {best['label'].split('/')[-1]}) | {best['smape']} | {best['severe_underestimate']} | "
            f"{d_smape:+.2f} {'✅' if d_smape <= 0 else '❌'} | {d_sev:+d} {'✅' if d_sev <= 0 else '❌'} |"
        )

    lines += [
        "",
        "## Files Changed",
        "",
        "| File | Change |",
        "|------|--------|",
        "| `scripts/evaluate_p34_weighted_lgbm_phase2_fusion.py` | **New** — Full evaluation pipeline |",
        "| `tests/test_p34_weighted_lgbm_fusion.py` | **New** — Unit tests |",
        "| `docs/reports/P34_weighted_lgbm_phase2_fusion_report.md` | **New** — This report |",
        "| `docs/p3_execution_board.md` | **Updated** — P3.4 status |",
    ]

    md_path = out_dir / "P34_weighted_lgbm_phase2_fusion_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] Report: {md_path}")
    print(f"  [OK] Summary: {json_path}")


if __name__ == "__main__":
    main()
