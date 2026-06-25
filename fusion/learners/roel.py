"""ROEL-BGEW: Rolling-Origin Expert Learner with Backward-Gated Expert Weighting.

Top-level orchestrator that:
1. Runs meta-validation (last_block_holdout) to select best candidate per (task, period)
2. Refits final weights on full OOF data
3. Produces routing table, weights, and all output artifacts
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .data_checks import compute_coverage_report, get_eligible_models
from .candidate_selector import select_best_candidate, CandidateResult
from .bgew import fit_bgew
from .static_convex import fit_static_convex
from .metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ROELOutput:
    """Complete output of ROEL-BGEW learner."""
    weights: pd.DataFrame  # weights.csv
    routing_table: pd.DataFrame  # routing_table.csv
    candidate_metrics: pd.DataFrame  # candidate_metrics.csv
    coverage_report: pd.DataFrame  # coverage_report.csv
    dynamic_weight_trace: pd.DataFrame  # dynamic_weight_trace.csv
    oof_backtest_predictions: pd.DataFrame  # oof_backtest_predictions.csv
    oof_backtest_metrics: pd.DataFrame  # oof_backtest_metrics.csv
    manifest: dict  # learner_manifest.json


def _get_month_label(ds_series: pd.Series) -> pd.Series:
    """Extract YYYY-MM from ds."""
    return pd.to_datetime(ds_series).dt.to_period("M").astype(str)


def _compute_backtest_predictions(
    oof_df: pd.DataFrame,
    routing_table: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute OOF backtest predictions using selected routing."""
    rows = []
    for (task, period), route_row in routing_table.groupby(["task", "period"]):
        route = route_row.iloc[0]
        selected_mode = route["selected_mode"]
        selected_model = route["selected_model"]

        sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)].copy()

        if selected_mode == "single_model":
            # Use single model predictions
            model_sub = sub[sub["model_name"] == selected_model]
            for _, row in model_sub.iterrows():
                rows.append({
                    "task": task,
                    "period": period,
                    "target_day": row["target_day"],
                    "ds": row["ds"],
                    "y_true": row["y_true"],
                    "y_pred_learner": row["y_pred"],
                    "selected_mode": selected_mode,
                    "selected_model": selected_model,
                    "available_models": selected_model,
                })
        else:
            # Fusion: use weights
            w_row = weights_df[(weights_df["task"] == task) & (weights_df["period"] == period)]
            weight_dict = {r["model_name"]: r["weight"] for _, r in w_row.iterrows()}

            # Pivot to wide
            wide = sub.pivot_table(
                index=["target_day", "ds", "hour_business"],
                columns="model_name",
                values="y_pred",
                aggfunc="first",
            )
            truth = sub.drop_duplicates(subset=["target_day", "ds", "hour_business"])[
                ["target_day", "ds", "hour_business", "y_true"]
            ].set_index(["target_day", "ds", "hour_business"])
            wide = wide.join(truth, how="inner")

            available_models = [m for m in weight_dict.keys() if m in wide.columns]
            for idx, row in wide.iterrows():
                y_pred_fused = 0.0
                total_w = 0.0
                for m, w in weight_dict.items():
                    if m in wide.columns and pd.notna(row.get(m)):
                        y_pred_fused += w * row[m]
                        total_w += w
                if total_w > 1e-9 and abs(total_w - 1.0) > 0.01:
                    y_pred_fused /= total_w

                rows.append({
                    "task": task,
                    "period": period,
                    "target_day": idx[0],
                    "ds": idx[1],
                    "y_true": row["y_true"],
                    "y_pred_learner": y_pred_fused,
                    "selected_mode": selected_mode,
                    "selected_model": selected_model if selected_mode == "single_model" else "fusion",
                    "available_models": ",".join(available_models),
                })

    return pd.DataFrame(rows)


def run_roel_bgew_fallback(
    oof_df: pd.DataFrame,
    *,
    metric_name: str = "sMAPE_floor50",
    tau: float = 30.0,
    eta: float = 0.5,
    coverage_threshold: float = 0.95,
    meta_eval_scheme: str = "last_block_holdout",
) -> ROELOutput:
    """Run ROEL-BGEW fallback learner on OOF data.

    Parameters
    ----------
    oof_df : pd.DataFrame
        Normalized OOF long-table.
    metric_name : str
        Metric to optimize ("sMAPE_floor50" or "MAE").
    tau : float
        BGEW time constant.
    eta : float
        BGEW learning rate.
    coverage_threshold : float
        Minimum coverage for model eligibility.
    meta_eval_scheme : str
        Meta-validation scheme (currently only "last_block_holdout").

    Returns
    -------
    ROELOutput
        All output artifacts.
    """
    # 1. Coverage report
    coverage_report = compute_coverage_report(oof_df, coverage_threshold=coverage_threshold)

    # 2. Get all (task, period) groups
    task_period_groups = oof_df.groupby(["task", "period"]).groups.keys()

    all_routing_rows = []
    all_candidate_metrics = []
    all_weights_rows = []
    all_trace_rows = []
    warnings = []

    for (task, period) in task_period_groups:
        eligible = get_eligible_models(coverage_report, task=task, period=period)
        if not eligible:
            logger.warning("No eligible models for task=%s, period=%s", task, period)
            warnings.append(f"No eligible models for {task}/{period}")
            continue

        sub = oof_df[(oof_df["task"] == task) & (oof_df["period"] == period)]
        months = sorted(sub["target_day"].apply(lambda x: x[:7]).unique())

        # Meta-validation: fit on fit_df, evaluate on eval_df, select best
        if meta_eval_scheme == "last_block_holdout" and len(months) > 1:
            fit_months = months[:-1]
            eval_months = months[-1:]
            fit_mask = sub["target_day"].apply(lambda x: x[:7]).isin(fit_months)
            eval_mask = sub["target_day"].apply(lambda x: x[:7]).isin(eval_months)
            fit_df = sub[fit_mask]
            eval_df = sub[eval_mask]

            # Fit candidates on fit_df
            best_fit, cand_metrics_fit = select_best_candidate(
                fit_df, task, period, eligible,
                metric_name=metric_name, tau=tau, eta=eta,
            )

            # Evaluate all candidates on eval_df
            from .candidate_selector import _evaluate_candidate
            eval_scores = []
            for _, row in cand_metrics_fit.iterrows():
                cand_name = row["candidate_name"]
                sel_mode = row["selected_mode"]
                sel_model = row["model_name_or_fusion"]

                if sel_mode == "single_model":
                    w_eval = {m: 1.0 if m == sel_model else 0.0 for m in eligible}
                elif sel_mode == "equal_weight":
                    w_eval = {m: 1.0 / len(eligible) for m in eligible}
                else:
                    # For fusion candidates, use equal weight as proxy (refit later)
                    w_eval = {m: 1.0 / len(eligible) for m in eligible}

                score_eval, metrics_eval = _evaluate_candidate(
                    eval_df, task, period, w_eval, metric_name=metric_name
                )
                eval_scores.append((cand_name, sel_mode, sel_model, score_eval, metrics_eval))

            # Select best based on eval_df
            best_eval = min(eval_scores, key=lambda x: x[3])
            best_name, best_mode, best_model, best_score, best_metrics = best_eval

            # Check retention rate
            retention = best_metrics.get("intersection_retention_rate", 1.0)
            if retention < 0.90:
                warn_msg = f"Low retention for {task}/{period}: {retention:.1%} < 90%"
                logger.warning(warn_msg)
                warnings.append(warn_msg)

            # Map back to CandidateResult for refit
            from .candidate_selector import CandidateResult
            best = CandidateResult(
                candidate_name=best_name,
                selected_mode=best_mode,
                selected_model=best_model if best_mode == "single_model" else None,
                weights={},  # Will be refit
                metric_value=best_score,
                metric_name=metric_name,
                full_metrics=best_metrics,
            )
            all_candidate_metrics.append(cand_metrics_fit)
        else:
            if len(months) <= 1:
                warnings.append(f"OOF span too short for {task}/{period}; candidate selection is in-sample")
            # Select best candidate on full data
            best, cand_metrics = select_best_candidate(
                sub, task, period, eligible,
                metric_name=metric_name, tau=tau, eta=eta,
            )
            all_candidate_metrics.append(cand_metrics)

            # Check retention rate
            retention = best.full_metrics.get("intersection_retention_rate", 1.0)
            if retention < 0.90:
                warn_msg = f"Low retention for {task}/{period}: {retention:.1%} < 90%"
                logger.warning(warn_msg)
                warnings.append(warn_msg)

        # Refit on full OOF data for final weights
        if best.selected_mode == "single_model":
            final_weights = {m: 1.0 if m == best.selected_model else 0.0 for m in eligible}
        elif best.selected_mode == "bgew":
            bgew_result = fit_bgew(oof_df, task, period, eligible, metric_name=metric_name, tau=tau, eta=eta)
            final_weights = bgew_result.weights
            all_trace_rows.extend(bgew_result.trace)
        elif best.selected_mode == "static_convex":
            sc_result = fit_static_convex(oof_df, task, period, eligible, metric_name=metric_name)
            final_weights = sc_result.weights
        else:  # equal_weight
            final_weights = {m: 1.0 / len(eligible) for m in eligible}

        # Build routing row
        all_routing_rows.append({
            "task": task,
            "period": period,
            "selected_mode": best.selected_mode,
            "selected_model": best.selected_model,
            "candidate_name": best.candidate_name,
            "metric_name": metric_name,
            "candidate_score": best.metric_value,
            "best_fusion_score": best.metric_value if best.selected_mode != "single_model" else float("nan"),
            "best_single_model": best.selected_model if best.selected_mode == "single_model" else float("nan"),
            "best_single_score": best.metric_value if best.selected_mode == "single_model" else float("nan"),
            "use_one_hot": best.selected_mode == "single_model",
            "reason": f"Best {metric_name} on OOF",
        })

        # Build weights rows
        for m, w in final_weights.items():
            all_weights_rows.append({
                "task": task,
                "period": period,
                "model_name": m,
                "weight": w,
                "learner_mode": "roel_bgew_fallback",
                "selected_mode": best.selected_mode,
                "selected_model": best.selected_model,
            })

    # Assemble outputs
    routing_table = pd.DataFrame(all_routing_rows)
    weights = pd.DataFrame(all_weights_rows)
    candidate_metrics = pd.concat(all_candidate_metrics, ignore_index=True) if all_candidate_metrics else pd.DataFrame()
    trace = pd.DataFrame(all_trace_rows) if all_trace_rows else pd.DataFrame()

    # Backtest predictions
    backtest_preds = _compute_backtest_predictions(oof_df, routing_table, weights)

    # Backtest metrics
    backtest_metrics_rows = []
    for (task, period), grp in backtest_preds.groupby(["task", "period"]):
        if "y_true" in grp.columns and grp["y_true"].notna().any():
            metrics = compute_all_metrics(grp["y_true"].values, grp["y_pred_learner"].values)
            backtest_metrics_rows.append({
                "task": task,
                "period": period,
                **metrics,
            })
    backtest_metrics = pd.DataFrame(backtest_metrics_rows) if backtest_metrics_rows else pd.DataFrame()

    # Manifest
    manifest = {
        "pool_id": "oof_learner",
        "generated_at": datetime.now().isoformat(),
        "learner_mode": "roel_bgew_fallback",
        "metric": metric_name,
        "tau": tau,
        "eta": eta,
        "coverage_threshold": coverage_threshold,
        "meta_eval_scheme": meta_eval_scheme,
        "oof_date_range": {
            "start": oof_df["target_day"].min(),
            "end": oof_df["target_day"].max(),
        },
        "models": sorted(oof_df["model_name"].unique().tolist()),
        "tasks": sorted(oof_df["task"].unique().tolist()),
        "n_routing_entries": len(routing_table),
        "warnings": warnings,
    }

    return ROELOutput(
        weights=weights,
        routing_table=routing_table,
        candidate_metrics=candidate_metrics,
        coverage_report=coverage_report,
        dynamic_weight_trace=trace,
        oof_backtest_predictions=backtest_preds,
        oof_backtest_metrics=backtest_metrics,
        manifest=manifest,
    )
