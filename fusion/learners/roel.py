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

            # Step 1: Fit all candidates on fit_df — get TRUE weights per candidate
            from .candidate_selector import fit_all_candidates, _evaluate_candidate, CandidateResult
            fitted_fit = fit_all_candidates(
                fit_df, task, period, eligible,
                metric_name=metric_name, tau=tau, eta=eta,
            )

            # Step 2: Evaluate each candidate with its TRUE fit_df weights on eval_df
            eval_results = []
            for cand_name, sel_mode, sel_model, w_fit, fit_metric in fitted_fit:
                eval_score, eval_metrics = _evaluate_candidate(
                    eval_df, task, period, w_fit, metric_name=metric_name,
                )
                eval_results.append((cand_name, sel_mode, sel_model, w_fit, fit_metric, eval_score, eval_metrics))

            # Step 3: Select best candidate based on eval_df metric
            # Simplicity tiebreaker (same as select_best_candidate)
            _complexity = {"single_model": 0, "equal_weight": 1, "static_convex": 2, "bgew": 3}
            best_eval_score = min(r[5] for r in eval_results)
            tol = max(abs(best_eval_score) * 1e-3, 1e-9)

            def _eval_sort_key(r):
                return (0 if r[5] <= best_eval_score + tol else 1, _complexity.get(r[1], 9))

            best_eval = min(eval_results, key=_eval_sort_key)
            best_name, best_mode, best_model, best_w_fit, best_fit_metric, best_eval_metric, best_eval_metrics = best_eval

            # Check retention rate
            retention = best_eval_metrics.get("intersection_retention_rate", 1.0)
            if retention < 0.90:
                warn_msg = f"Low retention for {task}/{period}: {retention:.1%} < 90%"
                logger.warning(warn_msg)
                warnings.append(warn_msg)

            # Build CandidateResult for refit
            best = CandidateResult(
                candidate_name=best_name,
                selected_mode=best_mode,
                selected_model=best_model if best_mode == "single_model" else None,
                weights={},  # Will be refit on full OOF
                metric_value=best_eval_metric,
                metric_name=metric_name,
                full_metrics=best_eval_metrics,
            )

            # Step 4: Refit on full OOF for final weights (done below)
            # Step 5: Build candidate_metrics with fit_metric, eval_metric, selected_by
            cand_metric_rows = []
            for cand_name, sel_mode, sel_model, w_fit, fit_metric, eval_score, eval_metrics in eval_results:
                cand_metric_rows.append({
                    "task": task,
                    "period": period,
                    "candidate_name": cand_name,
                    "selected_mode": sel_mode,
                    "model_name_or_fusion": sel_model if sel_model else "fusion",
                    "fit_metric": fit_metric,
                    "eval_metric": eval_score,
                    "sMAPE_floor50": eval_score if metric_name == "sMAPE_floor50" else float("nan"),
                    "MAE": eval_score if metric_name == "MAE" else float("nan"),
                    "intersection_retention_rate": eval_metrics.get("intersection_retention_rate", float("nan")),
                    "selected_by": "eval_holdout",
                })
            all_candidate_metrics.append(pd.DataFrame(cand_metric_rows))
        else:
            if len(months) <= 1:
                warnings.append(f"OOF span too short for {task}/{period}; candidate selection is in-sample")
            # Select best candidate on full data
            best, cand_metrics = select_best_candidate(
                sub, task, period, eligible,
                metric_name=metric_name, tau=tau, eta=eta,
            )

            # Add fit_metric/eval_metric/selected_by columns for consistency
            cand_metrics = cand_metrics.copy()
            cand_metrics["fit_metric"] = cand_metrics["sMAPE_floor50"] if metric_name == "sMAPE_floor50" else cand_metrics["MAE"]
            cand_metrics["eval_metric"] = cand_metrics["fit_metric"]  # No holdout, same as fit
            cand_metrics["selected_by"] = "in_sample"
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

        # Compute final_refit_metric: evaluate refitted weights on full OOF
        from .candidate_selector import _evaluate_candidate
        final_refit_metric, _ = _evaluate_candidate(
            oof_df, task, period, final_weights, metric_name=metric_name,
        )

        # Build routing row
        all_routing_rows.append({
            "task": task,
            "period": period,
            "selected_mode": best.selected_mode,
            "selected_model": best.selected_model,
            "candidate_name": best.candidate_name,
            "metric_name": metric_name,
            "candidate_score": best.metric_value,
            "final_refit_metric": final_refit_metric,
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

    # Backfill final_refit_metric into candidate_metrics for the winning candidate
    if not candidate_metrics.empty and not routing_table.empty:
        refit_map = routing_table.set_index(["task", "period"])["final_refit_metric"].to_dict()
        candidate_metrics["final_refit_metric"] = float("nan")
        for idx, row in candidate_metrics.iterrows():
            key = (row["task"], row["period"])
            cand = row["candidate_name"]
            # Only the winning candidate has a final_refit_metric
            route = routing_table[
                (routing_table["task"] == key[0]) & (routing_table["period"] == key[1])
            ]
            if not route.empty and route.iloc[0]["candidate_name"] == cand:
                candidate_metrics.at[idx, "final_refit_metric"] = refit_map.get(key, float("nan"))

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
