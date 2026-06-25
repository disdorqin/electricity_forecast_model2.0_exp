"""Apply trained OOF learner to final forecast predictions.

Reads routing_table + weights from learner artifacts and applies them
to a forecast long-table to produce final fused predictions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def apply_learner_to_forecast(
    forecast_df: pd.DataFrame,
    routing_table: pd.DataFrame,
    weights_df: pd.DataFrame,
    *,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Apply learner to forecast long-table.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Forecast long-table (from escort or daily_run).
    routing_table : pd.DataFrame
        Routing table from learner training.
    weights_df : pd.DataFrame
        Weights from learner training.
    output_path : str or Path, optional
        If provided, save output CSV here.

    Returns
    -------
    pd.DataFrame
        Final fused predictions with columns:
        task, target_day, ds, business_day, hour_business, period,
        y_pred, selected_mode, selected_model, available_models, weight_summary
    """
    rows = []

    # Group forecast by (task, target_day, period)
    for (task, target_day, period), grp in forecast_df.groupby(["task", "target_day", "period"]):
        # Find routing for this (task, period)
        route_match = routing_table[
            (routing_table["task"] == task) & (routing_table["period"] == period)
        ]
        if route_match.empty:
            logger.warning("No routing for task=%s, period=%s; skipping", task, period)
            continue

        route = route_match.iloc[0]
        selected_mode = route["selected_mode"]
        selected_model = route["selected_model"]

        # Get weights for this (task, period)
        w_match = weights_df[
            (weights_df["task"] == task) & (weights_df["period"] == period)
        ]
        weight_dict = {r["model_name"]: r["weight"] for _, r in w_match.iterrows()}

        # Get available models in this forecast group
        available_models = grp["model_name"].unique().tolist()

        if selected_mode == "single_model":
            # Use selected model
            if selected_model in available_models:
                model_rows = grp[grp["model_name"] == selected_model]
                for _, row in model_rows.iterrows():
                    rows.append({
                        "task": task,
                        "target_day": target_day,
                        "ds": row["ds"],
                        "business_day": row.get("business_day"),
                        "hour_business": row["hour_business"],
                        "period": period,
                        "y_pred": row["y_pred"],
                        "selected_mode": selected_mode,
                        "selected_model": selected_model,
                        "available_models": ",".join(sorted(available_models)),
                        "weight_summary": f"{selected_model}=1.0",
                    })
            else:
                # Fallback: use equal weight on available models
                logger.warning(
                    "Selected model %s missing for task=%s, period=%s; falling back to equal_weight",
                    selected_model, task, period,
                )
                n_avail = len(available_models)
                if n_avail == 0:
                    raise ValueError(f"No models available for task={task}, period={period}")
                eq_w = 1.0 / n_avail
                wide = grp.pivot_table(
                    index=["ds", "hour_business"],
                    columns="model_name",
                    values="y_pred",
                    aggfunc="first",
                )
                for idx, row in wide.iterrows():
                    y_pred = sum(eq_w * row[m] for m in available_models if m in row.index and pd.notna(row[m]))
                    rows.append({
                        "task": task,
                        "target_day": target_day,
                        "ds": idx[0],
                        "business_day": None,
                        "hour_business": idx[1],
                        "period": period,
                        "y_pred": y_pred,
                        "selected_mode": "equal_weight_fallback",
                        "selected_model": "fallback",
                        "available_models": ",".join(sorted(available_models)),
                        "weight_summary": ",".join(f"{m}={eq_w:.3f}" for m in sorted(available_models)),
                    })
        else:
            # Fusion mode
            wide = grp.pivot_table(
                index=["ds", "hour_business"],
                columns="model_name",
                values="y_pred",
                aggfunc="first",
            )

            # Renormalize weights for available models
            avail_weights = {m: weight_dict.get(m, 0.0) for m in available_models if m in weight_dict}
            total_w = sum(avail_weights.values())
            if total_w < 1e-9:
                # No weights available; fall back to equal
                avail_weights = {m: 1.0 / len(available_models) for m in available_models}
                total_w = 1.0
            else:
                avail_weights = {m: w / total_w for m, w in avail_weights.items()}

            for idx, row in wide.iterrows():
                y_pred = 0.0
                used_models = []
                for m, w in avail_weights.items():
                    if m in row.index and pd.notna(row[m]):
                        y_pred += w * row[m]
                        used_models.append(f"{m}={w:.3f}")

                rows.append({
                    "task": task,
                    "target_day": target_day,
                    "ds": idx[0],
                    "business_day": None,
                    "hour_business": idx[1],
                    "period": period,
                    "y_pred": y_pred,
                    "selected_mode": selected_mode,
                    "selected_model": "fusion",
                    "available_models": ",".join(sorted(available_models)),
                    "weight_summary": ",".join(used_models),
                })

    result = pd.DataFrame(rows)

    # Validation: each (task, target_day) should have 24 rows
    if not result.empty:
        counts = result.groupby(["task", "target_day"]).size()
        bad = counts[counts != 24]
        if not bad.empty:
            logger.warning("Some (task, target_day) groups don't have 24 rows:\n%s", bad)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("Saved final fused predictions to %s", output_path)

    return result


def load_learner_artifacts(artifact_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load learner artifacts from directory.

    Returns
    -------
    routing_table, weights_df, manifest
    """
    artifact_dir = Path(artifact_dir)
    routing_table = pd.read_csv(artifact_dir / "routing_table.csv")
    weights_df = pd.read_csv(artifact_dir / "weights.csv")
    with open(artifact_dir / "learner_manifest.json") as f:
        manifest = json.load(f)
    return routing_table, weights_df, manifest
