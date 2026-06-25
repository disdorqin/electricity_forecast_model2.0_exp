"""Comprehensive metrics for OOF learner evaluation.

Implements MAE, RMSE, sMAPE_floor50, bias_mean, bias_median, and high-price MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-6) -> float:
    """sMAPE with floor-50 clipping (percentage, 0-100 scale).

    Re-exports from fusion.metrics for convenience; identical logic.
    """
    true_clip = np.where(y_true < 50.0, 50.0, y_true)
    pred_clip = np.where(y_pred < 50.0, 50.0, y_pred)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Positive = over-prediction, negative = under-prediction."""
    return float(np.mean(y_pred - y_true))


def bias_median(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.median(y_pred - y_true))


def high_price_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.90,
) -> float:
    """MAE on the top-quantile price samples.

    Parameters
    ----------
    quantile : float
        Threshold quantile (e.g. 0.90 means top 10% prices).
    """
    if len(y_true) == 0:
        return float("nan")
    threshold = np.quantile(y_true, quantile)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute full metric suite for a single (task, period) group."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {
            "MAE": float("nan"),
            "RMSE": float("nan"),
            "sMAPE_floor50": float("nan"),
            "bias_mean": float("nan"),
            "bias_median": float("nan"),
            "q90_high_price_MAE": float("nan"),
            "q95_high_price_MAE": float("nan"),
            "n": 0,
        }
    return {
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "sMAPE_floor50": smape_floor50(yt, yp),
        "bias_mean": bias_mean(yt, yp),
        "bias_median": bias_median(yt, yp),
        "q90_high_price_MAE": high_price_mae(yt, yp, 0.90),
        "q95_high_price_MAE": high_price_mae(yt, yp, 0.95),
        "n": int(mask.sum()),
    }


def compute_candidate_metrics_df(
    oof_df: pd.DataFrame,
    weights_dict: dict[str, float],
    *,
    task: str,
    period: str,
    candidate_name: str,
    selected_mode: str,
    model_name_or_fusion: str,
) -> dict[str, object]:
    """Compute metrics for one candidate on one (task, period) slice.

    Returns a flat dict matching candidate_metrics.csv schema.
    """
    sub = oof_df[
        (oof_df["task"] == task) & (oof_df["period"] == period)
    ].copy()

    if model_name_or_fusion == "fusion":
        # Weighted combination
        sub["_y_pred_fused"] = 0.0
        total_w = 0.0
        for m, w in weights_dict.items():
            if w == 0.0:
                continue
            model_rows = sub[sub["model_name"] == m]
            if model_rows.empty:
                continue
            # Align by index
            sub = sub.set_index("ds", append=True)
            model_rows = model_rows.set_index("ds", append=True)
            sub.loc[model_rows.index, "_y_pred_fused"] = (
                sub.loc[model_rows.index, "_y_pred_fused"].fillna(0.0)
                + w * model_rows["y_pred"].values
            )
            sub = sub.reset_index().set_index(sub.columns[0])  # restore
            # Simpler approach: pivot-based
            break  # fallback to pivot approach

        # Use pivot approach for robustness
        sub = sub.reset_index(drop=True)
        wide = sub.pivot_table(
            index=["target_day", "ds", "hour_business"],
            columns="model_name",
            values="y_pred",
            aggfunc="first",
        )
        y_pred_fused = np.zeros(len(wide))
        for m, w in weights_dict.items():
            if m in wide.columns:
                y_pred_fused += w * wide[m].fillna(0.0).values
        y_true_vals = wide.index.get_level_values(0)

        # Get y_true from any available model
        truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
            ["target_day", "ds", "hour_business", "y_true"]
        ]
        truth = truth.set_index(["target_day", "ds", "hour_business"])
        wide["_fusion"] = y_pred_fused
        wide = wide.join(truth, how="left")

        yt = wide["y_true"].values.astype(float)
        yp = wide["_fusion"].values.astype(float)
    else:
        # Single model
        model_sub = sub[sub["model_name"] == model_name_or_fusion]
        yt = model_sub["y_true"].values.astype(float)
        yp = model_sub["y_pred"].values.astype(float)

    metrics = compute_all_metrics(yt, yp)
    return {
        "task": task,
        "period": period,
        "candidate_name": candidate_name,
        "selected_mode": selected_mode,
        "model_name_or_fusion": model_name_or_fusion,
        **metrics,
    }
