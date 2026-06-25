"""CLI runner for OOF learner pipeline.

Provides run_oof_learner() and run_apply_oof_learner() entry points.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from fusion.learners.oof_contracts import load_and_normalize_oof_table, load_forecast_long
from fusion.learners.roel import run_roel_bgew_fallback
from fusion.learners.apply_learner import apply_learner_to_forecast, load_learner_artifacts

logger = logging.getLogger(__name__)


def run_oof_learner(args) -> dict:
    """Run OOF learner training pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including:
        - oof_path: path to oof_long_table.csv
        - learner_mode: "roel_bgew_fallback" (default)
        - metric: "sMAPE_floor50" or "MAE"
        - tau: BGEW tau parameter
        - eta: BGEW eta parameter
        - coverage_threshold: minimum coverage for model eligibility
        - output_root: output directory

    Returns
    -------
    dict
        Manifest with output paths.
    """
    oof_path = getattr(args, "oof_path", None)
    if not oof_path:
        raise ValueError("--oof-path is required for oof_learner pipeline")

    learner_mode = getattr(args, "learner_mode", "roel_bgew_fallback")
    metric_name = getattr(args, "metric", "sMAPE_floor50")
    tau = getattr(args, "tau", 30.0)
    eta = getattr(args, "eta", 0.5)
    coverage_threshold = getattr(args, "coverage_threshold", 0.95)
    output_root = Path(getattr(args, "output_root", "learner_runs"))

    logger.info("=" * 80)
    logger.info("OOF LEARNER PIPELINE")
    logger.info("=" * 80)
    logger.info("OOF path: %s", oof_path)
    logger.info("Learner mode: %s", learner_mode)
    logger.info("Metric: %s", metric_name)
    logger.info("Tau: %.1f, Eta: %.2f", tau, eta)
    logger.info("Coverage threshold: %.2f", coverage_threshold)
    logger.info("Output root: %s", output_root)

    # Load and normalize OOF data
    logger.info("Loading OOF long-table...")
    oof_df = load_and_normalize_oof_table(oof_path)

    # Run ROEL-BGEW fallback
    logger.info("Running ROEL-BGEW fallback learner...")
    output = run_roel_bgew_fallback(
        oof_df,
        metric_name=metric_name,
        tau=tau,
        eta=eta,
        coverage_threshold=coverage_threshold,
    )

    # Save outputs
    output_root.mkdir(parents=True, exist_ok=True)

    output.weights.to_csv(output_root / "weights.csv", index=False)
    output.routing_table.to_csv(output_root / "routing_table.csv", index=False)
    output.candidate_metrics.to_csv(output_root / "candidate_metrics.csv", index=False)
    output.coverage_report.to_csv(output_root / "coverage_report.csv", index=False)
    if not output.dynamic_weight_trace.empty:
        output.dynamic_weight_trace.to_csv(output_root / "dynamic_weight_trace.csv", index=False)
    output.oof_backtest_predictions.to_csv(output_root / "oof_backtest_predictions.csv", index=False)
    output.oof_backtest_metrics.to_csv(output_root / "oof_backtest_metrics.csv", index=False)

    # Update manifest with output paths
    output.manifest["output_root"] = str(output_root)
    output.manifest["oof_path"] = str(oof_path)
    output.manifest["output_files"] = {
        "weights": "weights.csv",
        "routing_table": "routing_table.csv",
        "candidate_metrics": "candidate_metrics.csv",
        "coverage_report": "coverage_report.csv",
        "dynamic_weight_trace": "dynamic_weight_trace.csv" if not output.dynamic_weight_trace.empty else None,
        "oof_backtest_predictions": "oof_backtest_predictions.csv",
        "oof_backtest_metrics": "oof_backtest_metrics.csv",
    }

    with open(output_root / "learner_manifest.json", "w") as f:
        json.dump(output.manifest, f, indent=2)

    logger.info("=" * 80)
    logger.info("OOF LEARNER COMPLETE")
    logger.info("=" * 80)
    logger.info("Outputs saved to: %s", output_root)
    logger.info("Routing table: %d entries", len(output.routing_table))
    logger.info("Backtest metrics:\n%s", output.oof_backtest_metrics.to_string())

    # If forecast_path provided, also apply learner
    forecast_path = getattr(args, "forecast_path", None)
    if forecast_path:
        logger.info("Applying learner to forecast: %s", forecast_path)
        forecast_df = load_forecast_long(forecast_path)
        final_df = apply_learner_to_forecast(
            forecast_df,
            output.routing_table,
            output.weights,
            output_path=output_root / "final_fused_predictions.csv",
        )
        logger.info("Final fused predictions: %d rows", len(final_df))
        output.manifest["forecast_path"] = str(forecast_path)
        output.manifest["output_files"]["final_fused_predictions"] = "final_fused_predictions.csv"
        with open(output_root / "learner_manifest.json", "w") as f:
            json.dump(output.manifest, f, indent=2)

    return output.manifest


def run_apply_oof_learner(args) -> dict:
    """Apply trained OOF learner to forecast.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including:
        - forecast_path: path to forecast long-table
        - learner_artifact: path to learner artifact directory
        - output_root: output directory

    Returns
    -------
    dict
        Summary with output path.
    """
    forecast_path = getattr(args, "forecast_path", None)
    learner_artifact = getattr(args, "learner_artifact", None)
    output_root = Path(getattr(args, "output_root", "learner_runs"))

    if not forecast_path:
        raise ValueError("--forecast-path is required for apply_oof_learner pipeline")
    if not learner_artifact:
        raise ValueError("--learner-artifact is required for apply_oof_learner pipeline")

    logger.info("=" * 80)
    logger.info("APPLY OOF LEARNER")
    logger.info("=" * 80)
    logger.info("Forecast path: %s", forecast_path)
    logger.info("Learner artifact: %s", learner_artifact)
    logger.info("Output root: %s", output_root)

    # Load artifacts
    routing_table, weights_df, manifest = load_learner_artifacts(learner_artifact)

    # Load forecast
    forecast_df = load_forecast_long(forecast_path)

    # Apply learner
    output_root.mkdir(parents=True, exist_ok=True)
    final_df = apply_learner_to_forecast(
        forecast_df,
        routing_table,
        weights_df,
        output_path=output_root / "final_fused_predictions.csv",
    )

    logger.info("=" * 80)
    logger.info("APPLY COMPLETE")
    logger.info("=" * 80)
    logger.info("Final fused predictions: %d rows", len(final_df))
    logger.info("Output: %s", output_root / "final_fused_predictions.csv")

    return {
        "output_path": str(output_root / "final_fused_predictions.csv"),
        "n_rows": len(final_df),
    }
