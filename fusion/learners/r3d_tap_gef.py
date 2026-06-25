"""R3D-Tap-GEF: Rolling 3-Day Validation Tap with Gated Expert Fusion.

Learner that updates model weights from the most recent fold backwards,
using recency and horizon gates, then optionally refits via constrained
convex optimization (Weighted Convex Refit).

Input
-----
validation_tap_long_table.csv with columns:
  task, model_name, tap_fold_id, age_block, horizon_day, period,
  ds, y_true, y_pred, ...

Output
------
weights, routing_table, dynamic_weight_trace, candidate_metrics
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .metrics import smape_floor50 as _smape_floor50, mae as _mae, rmse as _rmse

logger = logging.getLogger(__name__)

# ── Defaults ──────────────────────────────────────────────────────────
DEFAULT_TAU_BLOCK = 3.0
DEFAULT_TAU_HORIZON = 2.0
DEFAULT_ETA = 0.8
DEFAULT_WEIGHT_FLOOR = 0.03
DEFAULT_LAMBDA_REFIT = 0.05
VALID_PERIODS = ("1_8", "9_16", "17_24")


@dataclass
class R3DTapGEFOutput:
    """Output container for the R3D-Tap-GEF learner."""
    weights: pd.DataFrame
    routing_table: pd.DataFrame
    dynamic_weight_trace: pd.DataFrame
    candidate_metrics: pd.DataFrame
    coverage_report: pd.DataFrame
    manifest: dict = field(default_factory=dict)


# ── Weighted sMAPE floor50 ───────────────────────────────────────────
def _weighted_smape_floor50(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weights: np.ndarray,
) -> float:
    """Weighted sMAPE with floor-50 clipping.

    sample_weights: per-sample gate values (recency × horizon).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return float("nan")

    yt = y_true[mask]
    yp = y_pred[mask]
    sw = sample_weights[mask]

    true_clip = np.where(yt < 50.0, 50.0, yt)
    pred_clip = np.where(yp < 50.0, 50.0, yp)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < 1e-6, 1e-6, denom)
    errors = np.abs(pred_clip - true_clip) / denom

    if sw.sum() < 1e-9:
        return float(np.mean(errors) * 100.0)
    return float(np.average(errors, weights=sw) * 100.0)


# ── BGEW update per (task, period) ──────────────────────────────────
def _run_bgew_update(
    tap_df: pd.DataFrame,
    task: str,
    period: str,
    models: list[str],
    *,
    tau_block: float,
    tau_horizon: float,
    eta: float,
    weight_floor: float,
) -> tuple[dict[str, float], list[dict]]:
    """Run BGEW update from most recent fold to oldest.

    Returns (final_weights, trace_rows).
    """
    n_models = len(models)
    weights = {m: 1.0 / n_models for m in models}
    trace_rows: list[dict] = []

    # Get unique fold IDs sorted from most recent (lowest age_block) to oldest
    fold_info = (
        tap_df.groupby("tap_fold_id")
        .agg(age_block=("age_block", "first"))
        .sort_values("age_block", ascending=True)
    )
    fold_ids = fold_info.index.tolist()

    for fold_id in fold_ids:
        fold_data = tap_df[tap_df["tap_fold_id"] == fold_id]
        if fold_data.empty:
            continue

        age_block = int(fold_data["age_block"].iloc[0])
        recency_gate = float(np.exp(-age_block / tau_block))

        # Process each horizon_day in the fold
        for horizon_day in sorted(fold_data["horizon_day"].unique()):
            hday_data = fold_data[fold_data["horizon_day"] == horizon_day]
            horizon_gate = float(np.exp(-(horizon_day - 1) / tau_horizon))
            sample_gate = recency_gate * horizon_gate

            # Get y_true (same for all models, take from first available)
            truth = hday_data.drop_duplicates(subset=["ds"])[["ds", "y_true"]]
            y_true = truth["y_true"].values.astype(float)
            ds_vals = truth["ds"].values

            # Compute per-model weighted loss
            model_losses: dict[str, float] = {}
            for m in models:
                m_data = hday_data[hday_data["model_name"] == m]
                if m_data.empty:
                    continue
                # Align by ds
                m_aligned = m_data.set_index("ds").reindex(ds_vals)
                y_pred_m = m_aligned["y_pred"].values.astype(float)
                gate_arr = np.full(len(y_true), sample_gate)

                loss = _weighted_smape_floor50(y_true, y_pred_m, gate_arr)
                if np.isfinite(loss):
                    model_losses[m] = loss

            if len(model_losses) < 2:
                # Not enough models to update
                continue

            # Normalize losses by median
            loss_values = list(model_losses.values())
            median_loss = float(np.median(loss_values))
            if median_loss < 1e-6:
                median_loss = 1e-6

            for m in model_losses:
                norm_loss = model_losses[m] / median_loss
                norm_loss = np.clip(norm_loss, 0.25, 4.0)

                # BGEW update
                old_w = weights[m]
                weights[m] = old_w * np.exp(-eta * recency_gate * norm_loss)
                weights[m] = max(weights[m], weight_floor)

                trace_rows.append({
                    "task": task,
                    "period": period,
                    "tap_fold_id": fold_id,
                    "age_block": age_block,
                    "recency_gate": recency_gate,
                    "horizon_gate_mean": horizon_gate,
                    "model_name": m,
                    "weight_before": old_w,
                    "loss": model_losses[m],
                    "normalized_loss": norm_loss,
                    "weight_after": weights[m],  # before renormalization
                })

        # Renormalize after processing all horizons in this fold
        total = sum(weights.values())
        if total > 0:
            weights = {m: w / total for m, w in weights.items()}

    # Update trace with final normalized weights (post-fold normalization)
    # (The weight_after in trace is pre-normalization; that's fine for debugging)

    return weights, trace_rows


# ── Weighted Convex Refit ────────────────────────────────────────────
def _convex_refit(
    tap_df: pd.DataFrame,
    task: str,
    period: str,
    models: list[str],
    w_bgew: dict[str, float],
    *,
    lam: float,
    weight_floor: float,
) -> tuple[dict[str, float], str]:
    """Constrained convex optimization to refine BGEW weights.

    min_w  Σ gate_i * Loss(y_i, Σ_m w_m * pred_{m,i}) + λ * ||w - w_bgew||²
    s.t.   w_m >= weight_floor, Σ w_m = 1

    Returns (refit_weights, source) where source is "convex_refit" or "bgew_fallback".
    """
    try:
        from scipy.optimize import minimize

        n = len(models)
        w0 = np.array([w_bgew.get(m, 1.0 / n) for m in models])

        # Build aligned arrays for optimization
        # Pivot: rows = samples (ds), cols = models, values = y_pred
        all_ds = tap_df["ds"].unique()
        y_true_map = (
            tap_df.drop_duplicates(subset=["ds"])
            .set_index("ds")["y_true"]
        )

        model_pred_maps: dict[str, pd.Series] = {}
        for m in models:
            m_data = tap_df[tap_df["model_name"] == m].drop_duplicates(subset=["ds"])
            model_pred_maps[m] = m_data.set_index("ds")["y_pred"]

        # Compute sample gates
        gate_map = {}
        for _, row in tap_df.drop_duplicates(subset=["ds"]).iterrows():
            ds = row["ds"]
            age = row.get("age_block", 0)
            horizon = row.get("horizon_day", 1)
            gate_map[ds] = np.exp(-age / 3.0) * np.exp(-(horizon - 1) / 2.0)

        # Filter to ds where all models have predictions
        valid_ds = []
        for ds in all_ds:
            if ds in y_true_map.index and pd.notna(y_true_map[ds]):
                all_present = all(
                    ds in model_pred_maps[m].index and pd.notna(model_pred_maps[m].get(ds, np.nan))
                    for m in models
                )
                if all_present:
                    valid_ds.append(ds)

        if len(valid_ds) < 3:
            return w_bgew, "bgew_fallback"

        y_true_arr = np.array([float(y_true_map[ds]) for ds in valid_ds])
        pred_matrix = np.column_stack([
            [float(model_pred_maps[m][ds]) for ds in valid_ds] for m in models
        ])
        gate_arr = np.array([gate_map.get(ds, 1.0) for ds in valid_ds])

        def objective(w):
            y_fused = pred_matrix @ w
            # sMAPE floor50
            tc = np.where(y_true_arr < 50, 50, y_true_arr)
            pc = np.where(y_fused < 50, 50, y_fused)
            denom = (np.abs(tc) + np.abs(pc)) / 2.0
            denom = np.where(denom < 1e-6, 1e-6, denom)
            smape_err = np.abs(pc - tc) / denom
            # Normalized MAE
            mae_err = np.abs(y_true_arr - y_fused) / denom
            # Combined loss
            loss = 0.7 * smape_err + 0.3 * mae_err
            weighted_loss = np.average(loss, weights=gate_arr)
            # Regularization
            reg = lam * np.sum((w - w0) ** 2)
            return weighted_loss + reg

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(weight_floor, 1.0)] * n

        result = minimize(
            objective, w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-8},
        )

        if result.success:
            w_refit = result.x
            w_refit = np.maximum(w_refit, weight_floor)
            w_refit = w_refit / w_refit.sum()
            return {m: float(w_refit[i]) for i, m in enumerate(models)}, "convex_refit"
        else:
            logger.warning("Convex refit failed for %s/%s: %s", task, period, result.message)
            return w_bgew, "bgew_fallback"

    except ImportError:
        logger.warning("scipy not available, falling back to BGEW weights")
        return w_bgew, "bgew_fallback"
    except Exception as exc:
        logger.warning("Convex refit error for %s/%s: %s", task, period, exc)
        return w_bgew, "bgew_fallback"


# ── Main entry point ─────────────────────────────────────────────────
def run_r3d_tap_gef(
    tap_df: pd.DataFrame,
    *,
    tau_block: float = DEFAULT_TAU_BLOCK,
    tau_horizon: float = DEFAULT_TAU_HORIZON,
    eta: float = DEFAULT_ETA,
    weight_floor: float = DEFAULT_WEIGHT_FLOOR,
    lambda_refit: float = DEFAULT_LAMBDA_REFIT,
) -> R3DTapGEFOutput:
    """Run R3D-Tap-GEF learner on validation tap data.

    Parameters
    ----------
    tap_df : pd.DataFrame
        Validation tap long table with columns:
        task, model_name, tap_fold_id, age_block, horizon_day,
        period, ds, y_true, y_pred, ...
    tau_block, tau_horizon, eta, weight_floor, lambda_refit :
        Hyperparameters (see module docstring).

    Returns
    -------
    R3DTapGEFOutput
    """
    all_weights_rows: list[dict] = []
    all_routing_rows: list[dict] = []
    all_trace_rows: list[dict] = []
    all_metrics_rows: list[dict] = []
    all_coverage_rows: list[dict] = []
    warnings: list[str] = []

    # Group by (task, period)
    groups = tap_df.groupby(["task", "period"])

    for (task, period), group_df in groups:
        models = sorted(group_df["model_name"].unique().tolist())
        n_models = len(models)

        if n_models == 0:
            warnings.append(f"{task}/{period}: no models found")
            continue

        # Coverage check
        total_samples = len(group_df.drop_duplicates(subset=["ds"]))
        max_possible = total_samples * n_models
        actual_samples = len(group_df)
        coverage = actual_samples / max_possible if max_possible > 0 else 0.0

        all_coverage_rows.append({
            "task": task,
            "period": period,
            "n_models": n_models,
            "n_samples": total_samples,
            "coverage": coverage,
        })

        # Step 1: BGEW update
        w_bgew, trace = _run_bgew_update(
            group_df, task, period, models,
            tau_block=tau_block,
            tau_horizon=tau_horizon,
            eta=eta,
            weight_floor=weight_floor,
        )
        all_trace_rows.extend(trace)

        # Step 2: Weighted Convex Refit
        w_final, source = _convex_refit(
            group_df, task, period, models, w_bgew,
            lam=lambda_refit,
            weight_floor=weight_floor,
        )

        # Record weights
        for m in models:
            all_weights_rows.append({
                "task": task,
                "period": period,
                "model_name": m,
                "weight": w_final.get(m, 1.0 / n_models),
                "weight_source": source,
                "learner_mode": "r3d_tap_gef",
            })

        # Record routing
        all_routing_rows.append({
            "task": task,
            "period": period,
            "selected_mode": "fusion",
            "fallback_reason": "" if source == "convex_refit" else "refit_failed",
        })

        # Compute per-model metrics
        for m in models:
            m_data = group_df[group_df["model_name"] == m]
            yt = m_data["y_true"].values.astype(float)
            yp = m_data["y_pred"].values.astype(float)
            mask = ~(np.isnan(yt) | np.isnan(yp))
            yt_c, yp_c = yt[mask], yp[mask]

            all_metrics_rows.append({
                "task": task,
                "period": period,
                "model_name": m,
                "MAE": _mae(yt_c, yp_c) if len(yt_c) > 0 else float("nan"),
                "RMSE": _rmse(yt_c, yp_c) if len(yt_c) > 0 else float("nan"),
                "sMAPE_floor50": _smape_floor50(yt_c, yp_c) if len(yt_c) > 0 else float("nan"),
                "weighted_sMAPE_floor50": float("nan"),  # computed below
                "coverage": coverage,
            })

        # Compute weighted sMAPE for fused prediction
        wide = group_df.pivot_table(
            index=["ds"],
            columns="model_name",
            values="y_pred",
            aggfunc="first",
        )
        truth = group_df.drop_duplicates(subset=["ds"]).set_index("ds")["y_true"]
        gate = group_df.drop_duplicates(subset=["ds"]).set_index("ds")["age_block"]
        gate = gate.apply(lambda a: np.exp(-a / tau_block))

        y_fused = np.zeros(len(wide))
        for m in models:
            if m in wide.columns:
                y_fused += w_final.get(m, 0) * wide[m].fillna(0).values

        common_ds = wide.index.intersection(truth.index)
        if len(common_ds) > 0:
            yt_fused = truth.loc[common_ds].values.astype(float)
            yp_fused = y_fused[wide.index.isin(common_ds)]
            gate_fused = gate.loc[common_ds].values.astype(float)
            wsmape = _weighted_smape_floor50(yt_fused, yp_fused, gate_fused)
        else:
            wsmape = float("nan")

        # Update metrics rows with weighted sMAPE
        for row in all_metrics_rows:
            if row["task"] == task and row["period"] == period:
                row["weighted_sMAPE_floor50"] = wsmape

    return R3DTapGEFOutput(
        weights=pd.DataFrame(all_weights_rows),
        routing_table=pd.DataFrame(all_routing_rows),
        dynamic_weight_trace=pd.DataFrame(all_trace_rows),
        candidate_metrics=pd.DataFrame(all_metrics_rows),
        coverage_report=pd.DataFrame(all_coverage_rows),
        manifest={
            "learner_mode": "r3d_tap_gef",
            "tau_block": tau_block,
            "tau_horizon": tau_horizon,
            "eta": eta,
            "weight_floor": weight_floor,
            "lambda_refit": lambda_refit,
            "warnings": warnings,
        },
    )
