#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_rolling_30d_fusion.py — Rolling 30-day weight fusion for P3.

Core algorithm:
    For each prediction business_day D:
        train_window = [D-30, D-1]
        Use y_true only inside train_window to fit per-model weights.
        Apply fitted weights to day D predictions.
        Never use D y_true to fit D weights.

Supported weight modes:
    convex   — non-negative weights summing to 1 (scipy constrained opt).
    ridge    — OLS with L2 penalty (no non-negativity constraint).
    softmax  — exp(-normalized_error / temperature), normalised.
    anchor   — fixed anchor_weight for a designated model, remaining
               split convexly among other models.

Outputs (--out-dir):
    rolling_weights.csv        — per-day per-model weights (long).
    rolling_predictions.csv    — fused predictions, 1 row per timestamp.
    rolling_metrics.csv        — per-day eval metrics (sMAPE/severe).
    rolling_manifest.json      — run config + summary.
    rolling_summary.md         — human-readable report.

Usage:
    python scripts/run_rolling_30d_fusion.py \\
        --prediction-pack outputs/prediction_pack_multicandidate.csv \\
        --fusion-mode convex \\
        --out-dir reports/local/p3_rolling_fusion \\
        --start-date 2025-11-01 --end-date 2026-02-28

CLI args (standard):
    --data-path, --target, --start-date, --end-date, --out-dir
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

# Ensure project root in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Optional scipy for convex mode
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Optional sklearn for ridge mode
try:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ── Constants ────────────────────────────────────────────────────────────

FUSION_MODES = [
    "convex", "ridge", "softmax", "anchor",
    "severe_softmax", "severe_anchor", "quantile_guarded",
]
DEFAULT_TRAIN_WINDOW_DAYS = 30
ANCHOR_MODEL = "lightgbm"
ANCHOR_WEIGHT = 0.9  # default anchor weight for anchor mode
SEVERE_ANCHOR_MIN = 0.85  # minimum LightGBM weight for severe_anchor mode
DEFAULT_ALPHA = 1.0    # severe_softmax: severe underestimate penalty weight
DEFAULT_BETA = 0.5     # severe_softmax: underprediction MAE penalty weight
DEFAULT_RISK_THRESHOLD = 0.6  # quantile_guarded: spike risk probability threshold
DEFAULT_SEVERE_RATE_THRESHOLD = 0.10  # quantile_guarded: recent severe rate threshold
P75_LOOKBACK = 14  # quantile_guarded: lookback days for p75 calculation

# Models expected in a multi-candidate pack
BASELINE_MODELS = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
ALL_MODELS = BASELINE_MODELS + ["lightgbm"]


# ── Helpers ──────────────────────────────────────────────────────────────

def compute_smape_floor50(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
    """Compute sMAPE with 50 floor on both values (vectorised)."""
    yt = np.maximum(np.abs(y_true.values), 50.0)
    yp = np.maximum(np.abs(y_pred.values), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
    return np.minimum(smape, 50.0)


def get_period(hour_business: int) -> str:
    if 9 <= hour_business <= 16:
        return "9_16"
    elif 1 <= hour_business <= 8:
        return "night"
    elif 17 <= hour_business <= 24:
        return "evening"
    return "night"


# ── Weight fitters ───────────────────────────────────────────────────────

def fit_convex_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
) -> dict[str, float]:
    """Fit non-negative weights summing to 1 via constrained optimisation.

    train_preds:  DataFrame with one column per model, 1 row per timestamp.
    y_true_series:  Series aligned with train_preds index.
    """
    n_models = len(models)
    if n_models == 0:
        return {}
    if n_models == 1:
        return {models[0]: 1.0}

    Y = y_true_series.values
    X = train_preds[models].fillna(0).values

    def objective(w):
        pred = X @ w
        return float(np.mean((Y - pred) ** 2))

    # Constraints: sum(w) == 1, w >= 0
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n_models
    initial = np.ones(n_models) / n_models

    if HAS_SCIPY:
        result = minimize(objective, initial, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        weights = result.x if result.success else initial
    else:
        print("  [WARN] scipy not available; using equal weights for convex")
        weights = initial

    # Clamp near-zero weights
    weights = np.maximum(weights, 0.0)
    weights = weights / weights.sum()
    return dict(zip(models, weights))


def fit_ridge_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
    alpha: float = 1.0,
) -> dict[str, float]:
    """Fit weights via Ridge regression (L2 penalty, no non-negativity)."""
    n_models = len(models)
    if n_models == 0:
        return {}
    if n_models == 1:
        return {models[0]: 1.0}

    Y = y_true_series.values
    X = train_preds[models].fillna(0).values

    if HAS_SKLEARN:
        reg = Ridge(alpha=alpha, fit_intercept=False)
        reg.fit(X, Y)
        weights = reg.coef_
    else:
        # Fallback: OLS via numpy
        try:
            weights = np.linalg.lstsq(X, Y, rcond=None)[0]
        except np.linalg.LinAlgError:
            weights = np.ones(n_models) / n_models

    return dict(zip(models, weights))


def fit_softmax_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
    temperature: float = 0.1,
) -> dict[str, float]:
    """Fit weights via softmax of negative per-model RMSE."""
    weights: dict[str, float] = {}
    for model in models:
        resid = y_true_series.values - train_preds[model].fillna(0).values
        rmse = float(np.sqrt(np.mean(resid ** 2))) if len(resid) > 0 else 1e10
        weights[model] = -rmse  # lower RMSE → higher score

    # Softmax
    scores = np.array(list(weights.values()))
    scores_norm = scores / (np.abs(scores).max() + 1e-10)
    exp_scores = np.exp(scores_norm / max(temperature, 1e-10))
    softmax = exp_scores / exp_scores.sum()
    return dict(zip(models, softmax))


def fit_anchor_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
    anchor_model: str = ANCHOR_MODEL,
    anchor_weight: float = ANCHOR_WEIGHT,
) -> dict[str, float]:
    """Anchor weight on anchor_model, split remainder convexly."""
    if anchor_model not in models:
        print(f"  [WARN] anchor_model='{anchor_model}' not in models; using equal weights")
        return {m: 1.0 / len(models) for m in models}

    other_models = [m for m in models if m != anchor_model]
    if len(other_models) == 0:
        return {anchor_model: 1.0}

    # Split remaining weight (1 - anchor_weight) among other models via convex fit
    if len(other_models) > 1:
        other_weights = fit_convex_weights(train_preds, y_true_series, other_models)
    else:
        other_weights = {other_models[0]: 1.0}

    result = {anchor_model: anchor_weight}
    remaining = 1.0 - anchor_weight
    for m, w in other_weights.items():
        result[m] = remaining * w
    return result


# ── P3.1 Severe-underestimate-aware weight fitters ─────────────────────


def compute_severe_rate(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """Fraction of timestamps where y_true - y_pred > 200."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true - y_pred > 200))


def compute_underprediction_mae(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> float:
    """MAE on timestamps where y_pred < y_true (underprediction only)."""
    mask = y_pred < y_true
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def fit_severe_softmax_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
    temperature: float = 0.1,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict[str, float]:
    """Softmax weights with severe-underestimate penalty.

    score_i = recent_smape_i + alpha * severe_rate_i
              + beta * underprediction_mae_i / 200
    weight_i ∝ exp(-temperature * normalized_score_i)
    """
    ytv = y_true_series.values
    scores: dict[str, float] = {}
    for model in models:
        preds = train_preds[model].fillna(0).values

        # sMAPE penalty
        smape = float(np.nanmean(
            compute_smape_floor50(y_true_series, pd.Series(preds))
        ))

        # Severe underestimate rate
        severe_rate = compute_severe_rate(ytv, preds)

        # Underprediction MAE (normalized by 200 to be comparable to sMAPE)
        under_mae = compute_underprediction_mae(ytv, preds) / 200.0

        composite = smape + alpha * severe_rate * 100.0 + beta * under_mae
        scores[model] = composite  # lower is better

    # Softmax over negative scores (lower composite -> higher weight)
    arr = np.array([scores[m] for m in models])
    s_min, s_max = arr.min(), arr.max()
    if s_max - s_min > 1e-10:
        arr_norm = (arr - s_min) / (s_max - s_min)
    else:
        arr_norm = np.zeros_like(arr)
    exp_neg = np.exp(-arr_norm / max(temperature, 1e-10))
    softmax = exp_neg / exp_neg.sum()
    return dict(zip(models, softmax))


def fit_severe_anchor_weights(
    train_preds: pd.DataFrame,
    y_true_series: pd.Series,
    models: list[str],
    anchor_model: str = ANCHOR_MODEL,
    min_anchor_weight: float = SEVERE_ANCHOR_MIN,
) -> dict[str, float]:
    """LightGBM anchor >= min_anchor_weight; remaining weight only allocated
    to baselines that reduce recent severe underestimate rate vs anchor alone.

    For each baseline model, check if its severe_underestimate_rate is
    <= the anchor's rate. If yes, share remaining weight proportionally
    by inverse RMSE. If not, that baseline's share goes to the anchor.
    """
    if anchor_model not in models:
        print(f"  [WARN] anchor_model='{anchor_model}' not in models; using equal weights")
        return {m: 1.0 / len(models) for m in models}

    ytv = y_true_series.values
    anchor_preds = train_preds[anchor_model].fillna(0).values
    anchor_severe_rate = compute_severe_rate(ytv, anchor_preds)
    anchor_rmse = float(np.sqrt(np.mean((ytv - anchor_preds) ** 2)))

    other_models = [m for m in models if m != anchor_model]
    if len(other_models) == 0:
        return {anchor_model: 1.0}

    # Evaluate each baseline: does it reduce severe underestimates?
    qualifying: list[str] = []
    qualifying_rmse: dict[str, float] = {}
    for m in other_models:
        mp = train_preds[m].fillna(0).values
        m_severe_rate = compute_severe_rate(ytv, mp)
        m_rmse = float(np.sqrt(np.mean((ytv - mp) ** 2)))
        if m_severe_rate <= anchor_severe_rate * 1.05:  # within 5% of anchor rate
            qualifying.append(m)
            qualifying_rmse[m] = m_rmse

    remaining = 1.0 - min_anchor_weight
    result = {anchor_model: min_anchor_weight}

    if not qualifying or remaining <= 0:
        result[anchor_model] = 1.0
        for m in other_models:
            result[m] = 0.0
        return result

    # Distribute remaining weight among qualifying baselines by inverse RMSE
    inv_rmse = {m: 1.0 / max(qualifying_rmse[m], 1e-10) for m in qualifying}
    total_inv = sum(inv_rmse.values())
    for m in qualifying:
        result[m] = remaining * inv_rmse[m] / total_inv
    for m in other_models:
        if m not in result:
            result[m] = 0.0

    return result


def apply_quantile_guard(
    predictions: pd.DataFrame,
    pack: pd.DataFrame,
    weights_df: pd.DataFrame,
    models: list[str],
    risk_df: pd.DataFrame | None = None,
    risk_threshold: float = DEFAULT_RISK_THRESHOLD,
    severe_rate_threshold: float = DEFAULT_SEVERE_RATE_THRESHOLD,
    anchor_model: str = ANCHOR_MODEL,
    p75_lookback: int = P75_LOOKBACK,
) -> pd.DataFrame:
    """Post-process predictions with upward quantile guard.

    For high-risk hours (spike risk high AND recent severe rate high),
    override base_fused_pred:
        final_pred = max(base_fused_pred, lightgbm_pred, recent_p75_prediction)
    """
    preds = predictions.copy()
    preds["guarded"] = 0
    preds["guard_source"] = "none"

    # Pivot pack predictions
    pred_pivot = pack.pivot_table(
        index=["business_day", "hour_business"],
        columns="model_name", values="y_pred", aggfunc="first",
    )

    # Merge risk scores if available
    has_risk = risk_df is not None and not risk_df.empty
    if has_risk:
        risk_df = risk_df.copy()
        risk_df["business_day"] = risk_df["business_day"].astype(str)

    # Compute recent p75 and severe rate for each business_day
    business_days = sorted(pack["business_day"].unique())
    bd_to_p75: dict[str, float] = {}
    bd_to_severe_rate: dict[str, float] = {}
    for day in business_days:
        day_dt = pd.to_datetime(day)
        lookback_start = day_dt - timedelta(days=p75_lookback)
        lookback_end = day_dt - timedelta(days=1)
        mask = (
            (pack["business_day"] >= lookback_start.strftime("%Y-%m-%d"))
            & (pack["business_day"] <= lookback_end.strftime("%Y-%m-%d"))
        )
        recent = pack[mask]
        if not recent.empty and anchor_model in recent.columns and "y_true" in recent.columns:
            anchor_vals = recent[anchor_model].dropna().values
            bd_to_p75[day] = float(np.percentile(anchor_vals, 75)) if len(anchor_vals) > 0 else 0.0
            yt = recent["y_true"].values
            yp = recent[anchor_model].values
            bd_to_severe_rate[day] = compute_severe_rate(yt, yp)
        else:
            bd_to_p75[day] = 0.0
            bd_to_severe_rate[day] = 0.0

    # Apply guard
    n_guarded = 0
    for idx, row in preds.iterrows():
        bd = row["business_day"]
        hb = row["hour_business"]

        # Check risk
        risk_high = False
        if has_risk:
            rmatch = risk_df[
                (risk_df["business_day"] == bd)
                & (risk_df["hour_business"] == hb)
            ]
            if not rmatch.empty:
                risk_score = float(rmatch.iloc[0].get("spike_risk_score", 0))
                risk_high = risk_score >= risk_threshold

        # Check recent severe rate
        severe_high = bd_to_severe_rate.get(bd, 0) >= severe_rate_threshold

        if risk_high and severe_high:
            try:
                anchor_pred = float(pred_pivot.loc[(bd, hb), anchor_model])
            except (KeyError, TypeError):
                anchor_pred = 0.0

            p75_val = bd_to_p75.get(bd, 0.0)
            base_val = float(row["base_fused_pred"])
            guarded_val = max(base_val, anchor_pred, p75_val)

            if guarded_val > base_val + 1e-6:
                preds.at[idx, "base_fused_pred"] = round(guarded_val, 4)
                preds.at[idx, "guarded"] = 1
                preds.at[idx, "guard_source"] = "quantile_guard"
                n_guarded += 1

    print(f"  Quantile guard applied: {n_guarded} timestamp(s) lifted")
    return preds


# ── Aggregators ──────────────────────────────────────────────────────────

def compute_per_day_weights(
    pack: pd.DataFrame,
    models: list[str],
    fusion_mode: str,
    train_window_days: int = DEFAULT_TRAIN_WINDOW_DAYS,
    min_history_days: int = 10,
    anchor_model: str = ANCHOR_MODEL,
    anchor_weight: float = ANCHOR_WEIGHT,
    temperature: float = 0.1,
    ridge_alpha: float = 1.0,
    severe_alpha: float = DEFAULT_ALPHA,
    severe_beta: float = DEFAULT_BETA,
    severe_anchor_min: float = SEVERE_ANCHOR_MIN,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute rolling weights per business_day.

    Returns DataFrame with columns: business_day, model_name, weight.
    1 row per (business_day, model).
    """
    business_days = sorted(pack["business_day"].unique())
    weight_rows: list[dict] = []

    for i, day in enumerate(business_days):
        day_dt = pd.to_datetime(day)
        train_start = day_dt - timedelta(days=train_window_days)
        train_end = day_dt - timedelta(days=1)

        # Training window: [D-lookback, D-1]
        train_mask = (
            (pack["business_day"] >= train_start.strftime("%Y-%m-%d"))
            & (pack["business_day"] <= train_end.strftime("%Y-%m-%d"))
        )
        train = pack[train_mask].copy()

        # Prediction day: D
        pred = pack[pack["business_day"] == day].copy()

        # Check minimum history
        n_train_days = train["business_day"].nunique() if not train.empty else 0
        min_days_threshold = min(min_history_days, train_window_days // 2)
        use_fallback = train.empty or pred.empty or n_train_days < min_days_threshold

        if use_fallback:
            # Not enough history → equal weights
            for m in models:
                weight_rows.append({"business_day": day, "model_name": m, "weight": 1.0 / len(models)})
            if verbose:
                print(f"  {day}: insufficient history → equal weights")
            continue

        # Pivot train and pred to 1 row per timestamp
        train_pivot = train.pivot_table(
            index=["business_day", "hour_business"],
            columns="model_name", values="y_pred", aggfunc="first",
        ).reset_index()
        train_ytrue = train.groupby(["business_day", "hour_business"])["y_true"].first()

        # Align indices
        common_idx = train_pivot.set_index(["business_day", "hour_business"]).index
        train_ytrue_aligned = train_ytrue.reindex(common_idx)

        if len(train_pivot) < min_history_days * 4:
            if verbose:
                print(f"  {day}: only {len(train_pivot)} train rows (need {min_history_days*4}) → equal weights")
            for m in models:
                weight_rows.append({"business_day": day, "model_name": m, "weight": 1.0 / len(models)})
            continue

        # Fit weights
        available_models = [m for m in models if m in train_pivot.columns]
        if fusion_mode == "convex":
            w = fit_convex_weights(train_pivot, train_ytrue_aligned, available_models)
        elif fusion_mode == "ridge":
            w = fit_ridge_weights(train_pivot, train_ytrue_aligned, available_models, alpha=ridge_alpha)
        elif fusion_mode == "softmax":
            w = fit_softmax_weights(train_pivot, train_ytrue_aligned, available_models, temperature=temperature)
        elif fusion_mode == "anchor":
            w = fit_anchor_weights(train_pivot, train_ytrue_aligned, available_models,
                                   anchor_model=anchor_model, anchor_weight=anchor_weight)
        elif fusion_mode == "severe_softmax":
            w = fit_severe_softmax_weights(train_pivot, train_ytrue_aligned, available_models,
                                           temperature=temperature, alpha=severe_alpha, beta=severe_beta)
        elif fusion_mode == "severe_anchor":
            w = fit_severe_anchor_weights(train_pivot, train_ytrue_aligned, available_models,
                                          anchor_model=anchor_model, min_anchor_weight=severe_anchor_min)
        elif fusion_mode == "quantile_guarded":
            # quantile_guarded uses softmax base weights
            w = fit_softmax_weights(train_pivot, train_ytrue_aligned, available_models, temperature=temperature)
        else:
            w = {m: 1.0 / len(available_models) for m in available_models}

        # Fill missing models with 0
        for m in models:
            weight_rows.append({"business_day": day, "model_name": m, "weight": w.get(m, 0.0)})

        if verbose and (i % 10 == 0 or i == len(business_days) - 1):
            top = sorted(w.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{m}={v:.3f}" for m, v in top)
            print(f"  {day}: weights={top_str}")

    return pd.DataFrame(weight_rows)


def apply_weights(
    pack: pd.DataFrame,
    weights_df: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    """Apply rolling weights to produce fused predictions.

    Returns DataFrame with 1 row per (business_day, hour_business),
    columns: business_day, hour_business, timestamp, period,
             model_weights, base_fused_pred, y_true, residual, smape_floor50.
    """
    # Get 1 row per timestamp with all model predictions
    ts = pack.drop_duplicates(subset=["business_day", "hour_business"]).copy()
    ts = ts.set_index(["business_day", "hour_business"])

    # Pivot predictions
    pred_pivot = pack.pivot_table(
        index=["business_day", "hour_business"],
        columns="model_name", values="y_pred", aggfunc="first",
    )

    # Pivot weights
    weight_pivot = weights_df.pivot_table(
        index="business_day",
        columns="model_name", values="weight", aggfunc="first",
    )

    # Build results
    results: list[dict] = []
    for (bd, hb), row in ts.iterrows():
        if bd not in weight_pivot.index:
            continue

        day_weights = weight_pivot.loc[bd]
        # Look up exact (bd, hb) tuple — avoids DataFrame vs Series ambiguity
        try:
            day_preds = pred_pivot.loc[(bd, hb)]
        except KeyError:
            continue

        fused = 0.0
        weight_dict: dict[str, float] = {}
        for m in models:
            if m in day_weights.index and m in day_preds.index:
                w = float(day_weights[m])
                p = float(day_preds[m]) if pd.notna(day_preds[m]) else 0.0
                fused += w * p
                weight_dict[m] = w

        y_true = float(row.get("y_true", np.nan))
        residual = y_true - fused if pd.notna(y_true) else np.nan
        smape = float(compute_smape_floor50(
            pd.Series([y_true]), pd.Series([fused])
        )[0]) if pd.notna(y_true) else np.nan

        results.append({
            "business_day": bd,
            "hour_business": hb,
            "timestamp": row.get("timestamp", ""),
            "period": row.get("period", get_period(int(hb))),
            "model_weights": json.dumps(weight_dict),
            "base_fused_pred": round(fused, 4),
            "y_true": y_true,
            "residual": round(residual, 4) if pd.notna(residual) else None,
            "smape_floor50": round(smape, 4) if pd.notna(smape) else None,
            "metric_level": "timestamp",
        })

    return pd.DataFrame(results)


# ── Metrics ──────────────────────────────────────────────────────────────

def compute_per_day_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute sMAPE and severe_underestimate per business_day."""
    rows: list[dict] = []
    for bd, grp in predictions.groupby("business_day"):
        valid = grp.dropna(subset=["y_true", "base_fused_pred"])
        if len(valid) == 0:
            continue
        smape = float(np.nanmean(compute_smape_floor50(valid["y_true"], valid["base_fused_pred"])))
        severe = int((valid["y_true"] - valid["base_fused_pred"] > 200).sum())
        rows.append({
            "business_day": bd,
            "n_hours": len(valid),
            "smape_floor50": round(smape, 4),
            "severe_underestimate": severe,
        })
    return pd.DataFrame(rows)


def compute_overall_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Compute overall timestamp-level metrics."""
    valid = predictions.dropna(subset=["y_true", "base_fused_pred"])
    smape = float(np.nanmean(compute_smape_floor50(valid["y_true"], valid["base_fused_pred"])))
    severe = int((valid["y_true"] - valid["base_fused_pred"] > 200).sum())

    # 9_16 subset
    p9 = valid[valid["period"] == "9_16"]
    smape_9_16 = float(np.nanmean(compute_smape_floor50(p9["y_true"], p9["base_fused_pred"]))) if len(p9) > 0 else None

    # Severe_underestimate_delta vs base
    severe_base = int((valid["y_true"] > valid["base_fused_pred"] + 200).sum())

    return {
        "n_timestamps": len(valid),
        "smape_floor50": round(smape, 4),
        "smape_9_16": round(smape_9_16, 4) if smape_9_16 is not None else None,
        "severe_underestimate": severe,
        "severe_underestimate_base": severe_base,
        "metric_level": "timestamp",
        "note": "All metrics computed on deduplicated (business_day, hour_business) rows.",
    }


# ── Output writers ───────────────────────────────────────────────────────

def write_summary_md(
    out_dir: Path,
    fusion_mode: str,
    models: list[str],
    train_window_days: int,
    overall: dict[str, Any],
    per_day_metrics: pd.DataFrame,
    n_weight_rows: int,
    n_pred_rows: int,
) -> Path:
    """Write rolling_summary.md."""
    lines = [
        "# Rolling 30D Fusion Summary",
        "",
        f"**Fusion mode**: {fusion_mode}",
        f"**Models**: {', '.join(models)}",
        f"**Train window**: {train_window_days} days",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Overall Metrics (timestamp-level, deduplicated)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| sMAPE (floor50) | {overall['smape_floor50']} |",
        f"| sMAPE 9_16 | {overall['smape_9_16']} |",
        f"| Severe underestimates | {overall['severe_underestimate']} |",
        f"| Timestamps | {overall['n_timestamps']} |",
        f"| Metric level | {overall['metric_level']} |",
        "",
        "## Per-Day Metrics (top/bottom 5)",
        "",
    ]

    if not per_day_metrics.empty:
        sorted_m = per_day_metrics.sort_values("smape_floor50")
        lines.append("### Best 5 days")
        lines.append("| Day | sMAPE | Severe | Hours |")
        lines.append("|-----|-------|--------|-------|")
        for _, r in sorted_m.head(5).iterrows():
            lines.append(f"| {r['business_day']} | {r['smape_floor50']} | {r['severe_underestimate']} | {r['n_hours']} |")

        lines.append("")
        lines.append("### Worst 5 days")
        lines.append("| Day | sMAPE | Severe | Hours |")
        lines.append("|-----|-------|--------|-------|")
        for _, r in sorted_m.tail(5).iterrows():
            lines.append(f"| {r['business_day']} | {r['smape_floor50']} | {r['severe_underestimate']} | {r['n_hours']} |")

    lines += [
        "",
        "## Output Files",
        "",
        f"| File | Rows |",
        f"|------|------|",
        f"| `rolling_weights.csv` | {n_weight_rows} |",
        f"| `rolling_predictions.csv` | {n_pred_rows} |",
        f"| `rolling_metrics.csv` | {len(per_day_metrics)} |",
    ]

    path = out_dir / "rolling_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── CLI ──────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rolling 30-day weight fusion for P3 multi-model pack.",
    )
    parser.add_argument("--data-path", default=None,
                        help="Ignored, kept for orchestrator compatibility")
    parser.add_argument("--target", default="realtime",
                        choices=["realtime", "dayahead", "both"],
                        help="Market target")
    parser.add_argument("--start-date", default="2025-11-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-02-28",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="reports/local/p3_rolling_fusion",
                        help="Output directory")
    parser.add_argument(
        "--prediction-pack", required=True,
        help="Path to multi-candidate prediction pack CSV "
             "(columns: business_day, hour_business, model_name, y_pred, y_true)",
    )
    parser.add_argument(
        "--risk-predictions", default=None,
        help="Optional risk predictions CSV for correction evaluation",
    )

    # Weight mode (supports --weight-mode and --fusion-mode aliases)
    parser.add_argument(
        "--weight-mode", "--fusion-mode", default="convex",
        dest="fusion_mode",
        choices=FUSION_MODES,
        help=f"Weight fitting mode (default: convex). Choices: {FUSION_MODES}",
    )

    # Lookback window
    parser.add_argument(
        "--lookback-days", "--train-window-days", type=int,
        default=DEFAULT_TRAIN_WINDOW_DAYS, dest="train_window_days",
        help=f"Training window in days before prediction day (default: {DEFAULT_TRAIN_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--min-history-days", type=int, default=10,
        help="Minimum history days required before using trained weights (default: 10)",
    )
    parser.add_argument(
        "--anchor-model", default=ANCHOR_MODEL,
        help=f"Anchor model name for 'anchor' mode (default: {ANCHOR_MODEL})",
    )
    parser.add_argument(
        "--anchor-weight", type=float, default=ANCHOR_WEIGHT,
        help=f"Anchor weight for 'anchor' mode (default: {ANCHOR_WEIGHT})",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Softmax temperature (default: 0.1; lower = more peaky)",
    )
    parser.add_argument(
        "--ridge-alpha", type=float, default=1.0,
        help="Ridge L2 regularisation alpha (default: 1.0)",
    )

    # P3.1 Severe-aware params
    parser.add_argument(
        "--severe-alpha", type=float, default=DEFAULT_ALPHA,
        help="Severe_softmax: severe underestimate penalty weight (default: 1.0)",
    )
    parser.add_argument(
        "--severe-beta", type=float, default=DEFAULT_BETA,
        help="Severe_softmax: underprediction MAE penalty weight (default: 0.5)",
    )
    parser.add_argument(
        "--severe-anchor-min", type=float, default=SEVERE_ANCHOR_MIN,
        help="Severe_anchor: minimum LightGBM anchor weight (default: 0.85)",
    )
    parser.add_argument(
        "--risk-threshold", type=float, default=DEFAULT_RISK_THRESHOLD,
        help="Quantile_guarded: spike risk probability threshold (default: 0.6)",
    )
    parser.add_argument(
        "--severe-rate-threshold", type=float, default=DEFAULT_SEVERE_RATE_THRESHOLD,
        help="Quantile_guarded: recent severe rate threshold (default: 0.10)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print(f"  Rolling {args.train_window_days}D Fusion  |  mode={args.fusion_mode}")
    print("=" * 60)

    # Resolve paths
    pack_path = Path(args.prediction_pack)
    if not pack_path.exists():
        print(f"  [ERR] Prediction pack not found: {pack_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load prediction pack ──────────────────────────────────────────
    print(f"\n  Loading prediction pack: {pack_path.name}")
    pack = pd.read_csv(pack_path)
    pack["business_day"] = pack["business_day"].astype(str)

    models_in_pack = pack["model_name"].unique().tolist()
    models = [m for m in ALL_MODELS if m in models_in_pack]
    print(f"  Models found: {models}")
    print(f"  Date range: {pack['business_day'].min()} ~ {pack['business_day'].max()}")
    print(f"  Total rows: {len(pack)}")

    # ── Compute rolling weights ───────────────────────────────────────
    print(f"\n  Computing rolling weights (train_window={args.train_window_days}d)...")
    weights_df = compute_per_day_weights(
        pack, models,
        fusion_mode=args.fusion_mode,
        train_window_days=args.train_window_days,
        min_history_days=args.min_history_days,
        anchor_model=args.anchor_model,
        anchor_weight=args.anchor_weight,
        temperature=args.temperature,
        ridge_alpha=args.ridge_alpha,
        severe_alpha=args.severe_alpha,
        severe_beta=args.severe_beta,
        severe_anchor_min=args.severe_anchor_min,
    )
    print(f"  -> {len(weights_df)} weight rows ({weights_df['business_day'].nunique()} days x {len(models)} models)")

    # ── Apply weights to produce fused predictions ────────────────────
    print(f"\n  Applying weights...")
    predictions = apply_weights(pack, weights_df, models)
    print(f"  -> {len(predictions)} prediction rows, 1 per timestamp")

    # ── Quantile guard post-processing (quantile_guarded mode) ────────
    if args.fusion_mode == "quantile_guarded":
        print(f"\n  Applying quantile guard post-processing...")
        risk_df = None
        if args.risk_predictions:
            risk_path = Path(args.risk_predictions)
            if risk_path.exists():
                risk_df = pd.read_csv(risk_path)
                print(f"  Loaded risk predictions: {len(risk_df)} rows")
            else:
                print(f"  [WARN] Risk predictions not found: {risk_path}")

        predictions = apply_quantile_guard(
            predictions, pack, weights_df, models,
            risk_df=risk_df,
            risk_threshold=args.risk_threshold,
            severe_rate_threshold=args.severe_rate_threshold,
        )

    # ── Compute metrics ───────────────────────────────────────────────
    print(f"\n  Computing metrics (timestamp-level, deduplicated)...")
    per_day = compute_per_day_metrics(predictions)
    overall = compute_overall_metrics(predictions)
    print(f"  sMAPE_floor50:     {overall['smape_floor50']}")
    print(f"  sMAPE_9_16:        {overall['smape_9_16']}")
    print(f"  Severe underest:   {overall['severe_underestimate']}")
    print(f"  Timestamps:        {overall['n_timestamps']}")
    print(f"  Metric level:      {overall['metric_level']}")

    # ── Write outputs ─────────────────────────────────────────────────
    print(f"\n  Writing outputs to {out_dir}...")
    weights_df.to_csv(out_dir / "rolling_weights.csv", index=False, encoding="utf-8")
    predictions.to_csv(out_dir / "rolling_predictions.csv", index=False, encoding="utf-8")
    per_day.to_csv(out_dir / "rolling_metrics.csv", index=False, encoding="utf-8")

    summary_path = write_summary_md(
        out_dir, args.fusion_mode, models, args.train_window_days,
        overall, per_day, len(weights_df), len(predictions),
    )
    print(f"  [OK] Summary: {summary_path}")

    # ── Write manifest ────────────────────────────────────────────────
    manifest = {
        "script": "scripts/run_rolling_30d_fusion.py",
        "fusion_mode": args.fusion_mode,
        "models": models,
        "train_window_days": args.train_window_days,
        "anchor_model": args.anchor_model if args.fusion_mode in ("anchor", "severe_anchor") else None,
        "anchor_weight": args.anchor_weight if args.fusion_mode == "anchor" else None,
        "severe_anchor_min": args.severe_anchor_min if args.fusion_mode == "severe_anchor" else None,
        "severe_alpha": args.severe_alpha if args.fusion_mode == "severe_softmax" else None,
        "severe_beta": args.severe_beta if args.fusion_mode == "severe_softmax" else None,
        "risk_threshold": args.risk_threshold if args.fusion_mode == "quantile_guarded" else None,
        "severe_rate_threshold": args.severe_rate_threshold if args.fusion_mode == "quantile_guarded" else None,
        "date_range": {"start": args.start_date, "end": args.end_date},
        "overall_metrics": overall,
        "n_weight_rows": len(weights_df),
        "n_prediction_rows": len(predictions),
        "n_business_days": int(per_day["business_day"].nunique()) if not per_day.empty else 0,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "leakage_safe": True,
        "note": (
            f"Rolling {args.train_window_days}D fusion: for each business_day D, "
            f"weights are fitted on [D-{args.train_window_days}, D-1] using y_true. "
            f"Day D y_true never participates in fitting day D weights. "
            f"Mode: {args.fusion_mode}."
        ),
    }
    manifest_path = out_dir / "rolling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] Manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
