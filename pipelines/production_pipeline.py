"""Production pipeline: end-to-end electricity price forecasting.

Orchestrates:
1. Rolling-origin OOF pool generation
2. Per-model forecast for target date
3. ROEL-BGEW OOF learner training
4. Learner-based fusion
5. Negative price classifier (realtime only)

Usage:
    python main.py 2026-06-25
    python main.py --start 2026-06-01 --end 2026-06-07
    python main.py 2026-06-25 --force
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Formal model lists ──────────────────────────────────────────────
FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["sgdfnet", "timemixer", "rt916", "timesfm"]

FORMAL_MODELS_BY_TASK = {
    "dayahead": FORMAL_DAYAHEAD_MODELS,
    "realtime": FORMAL_REALTIME_MODELS,
}

ALL_FORMAL_MODELS = sorted(set(FORMAL_DAYAHEAD_MODELS + FORMAL_REALTIME_MODELS))


# ── Directory helpers ────────────────────────────────────────────────
def _output_root(args) -> Path:
    return Path("outputs")


def _date_dir(args, date: str) -> Path:
    return _output_root(args) / date


def _manifest_path(date_dir: Path) -> Path:
    return date_dir / "run_manifest.json"


def _step_dir(date_dir: Path, target: str, step: str) -> Path:
    """Return path for a pipeline step directory, creating it."""
    p = date_dir / target / step
    p.mkdir(parents=True, exist_ok=True)
    return p


def _file_nonempty(path: Path) -> bool:
    """Check if file exists and is non-empty."""
    return path.exists() and path.stat().st_size > 0


# ── Manifest I/O ────────────────────────────────────────────────────
def _write_manifest(date_dir: Path, manifest: dict):
    path = _manifest_path(date_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


def _load_manifest(date_dir: Path) -> dict | None:
    path = _manifest_path(date_dir)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _init_manifest(date: str, targets: list[str]) -> dict:
    return {
        "date": date,
        "status": "running",
        "pipeline_version": "production_v1",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "targets": targets,
        "dayahead_models": FORMAL_DAYAHEAD_MODELS.copy(),
        "realtime_models": FORMAL_REALTIME_MODELS.copy(),
        "steps": {},
        "final_outputs": {},
        "warnings": [],
    }


# ── Cache / skip logic ──────────────────────────────────────────────
def _should_skip_entire(date_dir: Path, force: bool) -> bool:
    """Check if we can skip the entire run for this date."""
    if force:
        return False
    manifest = _load_manifest(date_dir)
    if manifest is None:
        return False
    status = manifest.get("status", "")
    if status not in ("complete", "complete_with_warnings"):
        return False
    # Check final output files exist and are non-empty
    final_dir = date_dir / "final"
    for f in ["dayahead_final_predictions.csv", "realtime_final_predictions.csv"]:
        if not _file_nonempty(final_dir / f):
            return False
    # If classifier was successful, check corrected output too
    clf_status = manifest.get("steps", {}).get("realtime_classifier", "")
    if "complete" in str(clf_status):
        if not _file_nonempty(final_dir / "realtime_final_predictions_corrected.csv"):
            return False
    return True


def _should_skip_step(step_dir: Path, key_files: list[str], force: bool) -> bool:
    """Check if a pipeline step can be skipped."""
    if force:
        return False
    return all(_file_nonempty(step_dir / f) for f in key_files)


# ── OOF month range ─────────────────────────────────────────────────
def _compute_oof_months(date_str: str) -> tuple[str, str]:
    """Compute OOF start/end month for a given prediction date.

    end_month = prediction date's month - 1
    start_month = end_month - 4
    """
    dt = pd.Timestamp(date_str)
    end_month_dt = dt.replace(day=1) - pd.Timedelta(days=1)
    end_month = end_month_dt.strftime("%Y-%m")
    start_month_dt = end_month_dt - pd.DateOffset(months=4)
    start_month = start_month_dt.strftime("%Y-%m")
    return start_month, end_month


# ── Step 1: OOF pool generation ─────────────────────────────────────
def _step_oof_pool(args, date_dir: Path, target: str, manifest: dict) -> Path:
    """Generate or reuse rolling-origin OOF pool.

    Returns path to oof_long_table.csv.
    """
    oof_dir = _step_dir(date_dir, target, "01_model_oof")
    oof_table = oof_dir / "oof_long_table.csv"

    if _should_skip_step(oof_dir, ["oof_long_table.csv"], getattr(args, "force", False)):
        logger.info("SKIP %s OOF pool — already exists", target)
        manifest["steps"][f"{target}_oof"] = "skipped"
        _write_manifest(date_dir, manifest)
        return oof_table

    date_str = manifest["date"]
    oof_start, oof_end = _compute_oof_months(date_str)

    # Check shared cache
    pool_id = f"oof_{oof_start}_to_{oof_end}_expanding"
    shared_root = Path(getattr(args, "oof_output_root", "oof_runs"))
    shared_pool_dir = shared_root / pool_id
    shared_table = shared_pool_dir / "oof_long_table.csv"

    if _file_nonempty(shared_table):
        logger.info("REUSE shared OOF pool: %s", shared_table)
        shutil.copy2(shared_table, oof_table)
        manifest["steps"][f"{target}_oof"] = "complete"
        manifest.setdefault("oof_pool_source", str(shared_pool_dir))
        _write_manifest(date_dir, manifest)
        return oof_table

    # Generate OOF pool
    logger.info("GENERATING OOF pool for %s: %s to %s", target, oof_start, oof_end)
    try:
        from rolling_oof.contracts import RollingOriginConfig
        from rolling_oof.scheduler import RollingOriginOrchestrator

        models = FORMAL_MODELS_BY_TASK[target]
        tasks = [target]

        config = RollingOriginConfig(
            data_path=args.data_path,
            output_root=str(shared_root),
            start_month=oof_start,
            end_month=oof_end,
            models=models,
            tasks=tasks,
            expanding=True,
            max_cpu_workers=getattr(args, "max_cpu_workers", 2),
            max_gpu_workers=getattr(args, "max_gpu_workers", 1),
            skip_audit=getattr(args, "skip_oof_audit", False),
        )

        orchestrator = RollingOriginOrchestrator(config)
        orchestrator.run_all()

        if _file_nonempty(shared_table):
            shutil.copy2(shared_table, oof_table)
            manifest["steps"][f"{target}_oof"] = "complete"
            manifest["oof_pool_source"] = str(shared_pool_dir)
        else:
            raise FileNotFoundError(f"OOF pool not generated: {shared_table}")

    except Exception as exc:
        logger.error("OOF pool generation failed for %s: %s", target, exc)
        manifest["steps"][f"{target}_oof"] = f"failed: {exc}"
        manifest["warnings"].append(f"OOF pool generation failed for {target}: {exc}")
        _write_manifest(date_dir, manifest)
        raise

    _write_manifest(date_dir, manifest)
    return oof_table


# ── Step 2: Model forecasts ─────────────────────────────────────────
def _step_model_forecasts(args, date_dir: Path, target: str, manifest: dict) -> Path:
    """Generate per-model forecasts for the target date.

    Returns path to all_model_forecasts_long.csv.
    """
    forecast_dir = _step_dir(date_dir, target, "02_model_forecasts")
    all_forecasts = forecast_dir / "all_model_forecasts_long.csv"

    date_str = manifest["date"]
    models = FORMAL_MODELS_BY_TASK[target]
    force = getattr(args, "force", False)

    frames = []
    for model_name in models:
        model_dir = forecast_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pred_file = model_dir / "forecast_predictions.csv"

        if _should_skip_step(model_dir, ["forecast_predictions.csv"], force):
            logger.info("SKIP %s/%s forecast — already exists", target, model_name)
            df = pd.read_csv(pred_file)
            frames.append(df)
            continue

        logger.info("RUNNING %s/%s forecast for %s", target, model_name, date_str)
        try:
            from runners.registry import get_model_pipeline

            pipeline = get_model_pipeline(model_name)
            result = pipeline.predict_range(
                target=target,
                predict_date=date_str,
                start=date_str,
                end=date_str,
                data_path=args.data_path,
                output_root=str(_output_root(args)),
                training_months=getattr(args, "training_months", 12),
                val_ratio=getattr(args, "val_ratio", 0.2),
                use_predicted_temp=getattr(args, "use_predicted_temp", False),
                segment_count=getattr(args, "segment_count", 3),
                seed=getattr(args, "seed", 42),
                deterministic=getattr(args, "deterministic", False),
            )

            if result is None or result.frame is None or result.frame.empty:
                logger.warning("No prediction from %s/%s", target, model_name)
                continue

            # Normalize to long-table format
            frame = result.frame.copy()
            ts_col = frame.columns[0]
            pred_col = frame.columns[1]

            long_df = pd.DataFrame({
                "task": target,
                "model_name": model_name,
                "target_day": pd.to_datetime(frame[ts_col]).dt.normalize().where(
                    pd.to_datetime(frame[ts_col]).dt.hour != 0,
                    pd.to_datetime(frame[ts_col]).dt.normalize() - pd.Timedelta(days=1),
                ),
                "ds": pd.to_datetime(frame[ts_col]),
                "y_pred": pd.to_numeric(frame[pred_col], errors="coerce"),
            })
            long_df["target_day"] = pd.to_datetime(long_df["target_day"]).dt.strftime("%Y-%m-%d")
            long_df["hour_business"] = long_df["ds"].dt.hour.replace({0: 24}).astype(int)

            from fusion.contracts import infer_period
            long_df["period"] = long_df["hour_business"].apply(
                lambda h: infer_period(h) if pd.notna(h) else "unknown"
            )
            long_df["business_day"] = long_df["ds"].apply(
                lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if pd.notna(t) and t.hour == 0
                else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
            )
            long_df["source"] = model_name
            long_df["run_mode"] = "forecast"

            long_df.to_csv(pred_file, index=False)
            frames.append(long_df)
            logger.info("  %s/%s: %d rows", target, model_name, len(long_df))

        except Exception as exc:
            logger.error("Model forecast failed for %s/%s: %s", target, model_name, exc)
            manifest["warnings"].append(f"Forecast failed: {target}/{model_name}: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(all_forecasts, index=False)
        manifest["steps"][f"{target}_forecast"] = "complete"
    else:
        manifest["steps"][f"{target}_forecast"] = "failed: no model predictions"
        manifest["warnings"].append(f"No model forecasts for {target}")

    _write_manifest(date_dir, manifest)
    return all_forecasts


# ── Step 3: OOF learner training ────────────────────────────────────
def _step_learner(args, date_dir: Path, target: str, manifest: dict) -> Path:
    """Train ROEL-BGEW learner on OOF pool.

    Returns path to learner output directory.
    """
    learner_dir = _step_dir(date_dir, target, "03_learner")
    manifest_file = learner_dir / "learner_manifest.json"

    if _should_skip_step(learner_dir, ["learner_manifest.json"], getattr(args, "force", False)):
        # Check if manifest says complete
        try:
            with open(manifest_file, encoding="utf-8") as f:
                lm = json.load(f)
            if lm.get("status") == "complete" or lm.get("learner_mode"):
                logger.info("SKIP %s learner — already trained", target)
                manifest["steps"][f"{target}_learner"] = "skipped"
                _write_manifest(date_dir, manifest)
                return learner_dir
        except Exception:
            pass

    oof_table = _step_dir(date_dir, target, "01_model_oof") / "oof_long_table.csv"
    if not _file_nonempty(oof_table):
        raise FileNotFoundError(f"OOF table not found: {oof_table}")

    logger.info("TRAINING %s learner on %s", target, oof_table)
    try:
        from fusion.learners.oof_contracts import load_and_normalize_oof_table
        from fusion.learners.roel import run_roel_bgew_fallback

        oof_df = load_and_normalize_oof_table(oof_table)

        coverage_threshold = getattr(args, "coverage_threshold", 0.95)
        metric_name = getattr(args, "metric", "sMAPE_floor50")
        tau = getattr(args, "tau", 30.0)
        eta = getattr(args, "eta", 0.5)

        output = run_roel_bgew_fallback(
            oof_df,
            metric_name=metric_name,
            tau=tau,
            eta=eta,
            coverage_threshold=coverage_threshold,
        )

        # Save all artifacts
        output.weights.to_csv(learner_dir / "weights.csv", index=False)
        output.routing_table.to_csv(learner_dir / "routing_table.csv", index=False)
        output.candidate_metrics.to_csv(learner_dir / "candidate_metrics.csv", index=False)
        output.coverage_report.to_csv(learner_dir / "coverage_report.csv", index=False)

        if not output.dynamic_weight_trace.empty:
            output.dynamic_weight_trace.to_csv(learner_dir / "dynamic_weight_trace.csv", index=False)
        else:
            pd.DataFrame().to_csv(learner_dir / "dynamic_weight_trace.csv", index=False)

        if not output.oof_backtest_predictions.empty:
            output.oof_backtest_predictions.to_csv(learner_dir / "oof_backtest_predictions.csv", index=False)
        else:
            pd.DataFrame().to_csv(learner_dir / "oof_backtest_predictions.csv", index=False)

        if not output.oof_backtest_metrics.empty:
            output.oof_backtest_metrics.to_csv(learner_dir / "oof_backtest_metrics.csv", index=False)
        else:
            pd.DataFrame().to_csv(learner_dir / "oof_backtest_metrics.csv", index=False)

        # Write learner manifest
        learner_manifest = {
            "status": "complete",
            "learner_mode": "roel_bgew_fallback",
            "metric": metric_name,
            "tau": tau,
            "eta": eta,
            "coverage_threshold": coverage_threshold,
            "generated_at": datetime.now().isoformat(),
            "warnings": output.manifest.get("warnings", []),
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(learner_manifest, f, indent=2, ensure_ascii=False)

        manifest["steps"][f"{target}_learner"] = "complete"

    except Exception as exc:
        logger.error("Learner training failed for %s: %s", target, exc)
        manifest["steps"][f"{target}_learner"] = f"failed: {exc}"
        _write_manifest(date_dir, manifest)
        raise

    _write_manifest(date_dir, manifest)
    return learner_dir


# ── Step 4: Learner fusion ──────────────────────────────────────────
def _step_fusion(args, date_dir: Path, target: str, manifest: dict) -> Path:
    """Apply learner weights to fuse model forecasts.

    Returns path to fused_predictions.csv.
    """
    fusion_dir = _step_dir(date_dir, target, "04_fusion")
    fused_csv = fusion_dir / "fused_predictions.csv"

    if _should_skip_step(fusion_dir, ["fused_predictions.csv"], getattr(args, "force", False)):
        logger.info("SKIP %s fusion — already exists", target)
        manifest["steps"][f"{target}_fusion"] = "skipped"
        _write_manifest(date_dir, manifest)
        return fused_csv

    forecast_file = _step_dir(date_dir, target, "02_model_forecasts") / "all_model_forecasts_long.csv"
    learner_dir = _step_dir(date_dir, target, "03_learner")

    if not _file_nonempty(forecast_file):
        raise FileNotFoundError(f"Forecast long-table not found: {forecast_file}")

    logger.info("FUSING %s predictions for %s", target, manifest["date"])
    try:
        from fusion.learners.apply_learner import apply_learner_to_forecast, load_learner_artifacts
        from fusion.learners.oof_contracts import load_forecast_long

        forecast_df = load_forecast_long(forecast_file)
        routing_table, weights_df, _ = load_learner_artifacts(learner_dir)

        result = apply_learner_to_forecast(
            forecast_df, routing_table, weights_df, output_path=fused_csv,
        )

        # Ensure y_fused column exists (classifier needs it)
        if "y_pred" in result.columns and "y_fused" not in result.columns:
            result["y_fused"] = result["y_pred"]

        # Re-save with y_fused included
        result.to_csv(fused_csv, index=False)

        manifest["steps"][f"{target}_fusion"] = "complete"

        # Validation
        if not result.empty:
            counts = result.groupby(["task", "target_day"]).size()
            if not all(counts == 24):
                manifest["warnings"].append(f"{target} fusion: some days != 24 rows")
            if result["business_day"].isna().any():
                manifest["warnings"].append(f"{target} fusion: some business_day are None")

    except Exception as exc:
        logger.error("Fusion failed for %s: %s", target, exc)
        manifest["steps"][f"{target}_fusion"] = f"failed: {exc}"
        _write_manifest(date_dir, manifest)
        raise

    _write_manifest(date_dir, manifest)
    return fused_csv


# ── Step 5: Negative price classifier (realtime only) ───────────────
def _step_classifier(args, date_dir: Path, manifest: dict) -> Path | None:
    """Run negative price classifier on realtime fusion output.

    Creates a compat directory structure that run_classifier_pipeline expects,
    calls it, then copies results back to canonical locations.

    Returns path to corrected CSV, or None if failed/skipped.
    """
    clf_dir = _step_dir(date_dir, "realtime", "05_classifier")
    corrected_csv = clf_dir / "fused_predictions_corrected.csv"

    if _should_skip_step(clf_dir, ["fused_predictions_corrected.csv"], getattr(args, "force", False)):
        logger.info("SKIP classifier — already exists")
        manifest["steps"]["realtime_classifier"] = "skipped"
        _write_manifest(date_dir, manifest)
        return corrected_csv

    fused_csv = _step_dir(date_dir, "realtime", "04_fusion") / "fused_predictions.csv"
    if not _file_nonempty(fused_csv):
        logger.warning("No fusion output for classifier")
        manifest["steps"]["realtime_classifier"] = "skipped: no fusion"
        _write_manifest(date_dir, manifest)
        return None

    logger.info("RUNNING negative price classifier")
    try:
        from fusion.classifier_bridge import run_classifier_pipeline

        # Build compat directory structure: compat_fusion/realtime/fused_predictions.csv
        compat_root = clf_dir / "compat_fusion"
        compat_rt_dir = compat_root / "realtime"
        compat_rt_dir.mkdir(parents=True, exist_ok=True)
        compat_fused = compat_rt_dir / "fused_predictions.csv"

        # Copy fusion output to compat location, ensure y_fused column exists
        fused_df = pd.read_csv(fused_csv)
        if "y_pred" in fused_df.columns and "y_fused" not in fused_df.columns:
            fused_df["y_fused"] = fused_df["y_pred"]
        fused_df.to_csv(compat_fused, index=False)

        # Resolve project root and classifier data path
        project_root = Path(__file__).resolve().parents[1]
        default_clf_data = project_root / "ExtremPriceClf" / "data" / "260525.xlsx"
        clf_data_path = Path(args.clf_data) if getattr(args, "clf_data", None) else default_clf_data

        date_str = manifest["date"]
        result = run_classifier_pipeline(
            fusion_work_dir=compat_root,
            project_root=project_root,
            start_date=date_str,
            end_date=date_str,
            clf_data_path=clf_data_path,
        )

        status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"

        if status == "completed":
            # Copy corrected CSV from compat location to canonical locations
            compat_corrected = compat_rt_dir / "fused_predictions_corrected.csv"
            if compat_corrected.exists():
                shutil.copy2(compat_corrected, corrected_csv)
                manifest["steps"]["realtime_classifier"] = "complete"
                manifest["classifier_corrections"] = result.get("corrected_hours", 0)
            else:
                manifest["steps"]["realtime_classifier"] = "failed: corrected output not found"
                manifest["warnings"].append("Classifier completed but corrected CSV missing")
        elif status == "skipped":
            reason = result.get("reason", "unknown")
            manifest["steps"]["realtime_classifier"] = f"skipped: {reason}"
            manifest["warnings"].append(f"Classifier skipped: {reason}")
        else:
            manifest["steps"]["realtime_classifier"] = f"failed: {status}"
            manifest["warnings"].append(f"Classifier failed: {result}")

    except Exception as exc:
        logger.warning("Classifier failed (non-fatal): %s", exc)
        manifest["steps"]["realtime_classifier"] = "failed"
        manifest["warnings"].append(f"Classifier failed: {exc}")

    _write_manifest(date_dir, manifest)
    return corrected_csv if _file_nonempty(corrected_csv) else None


# ── Final output assembly ───────────────────────────────────────────
def _assemble_final_outputs(args, date_dir: Path, manifest: dict):
    """Copy/merge results into final/ directory."""
    final_dir = _step_dir(date_dir, "final", "")
    date_str = manifest["date"]
    targets = manifest["targets"]

    # Dayahead final
    if "dayahead" in targets:
        da_fused = date_dir / "dayahead" / "04_fusion" / "fused_predictions.csv"
        da_final = final_dir / "dayahead_final_predictions.csv"
        if _file_nonempty(da_fused):
            df = pd.read_csv(da_fused)
            # Ensure standard columns
            out_cols = ["target_day", "ds", "business_day", "hour_business", "period", "y_pred"]
            available = [c for c in out_cols if c in df.columns]
            df[available].to_csv(da_final, index=False)
            manifest["final_outputs"]["dayahead"] = str(da_final)

    # Realtime final
    if "realtime" in targets:
        rt_fused = date_dir / "realtime" / "04_fusion" / "fused_predictions.csv"
        rt_final = final_dir / "realtime_final_predictions.csv"
        if _file_nonempty(rt_fused):
            df = pd.read_csv(rt_fused)
            out_cols = ["target_day", "ds", "business_day", "hour_business", "period", "y_pred"]
            available = [c for c in out_cols if c in df.columns]
            df[available].to_csv(rt_final, index=False)
            manifest["final_outputs"]["realtime"] = str(rt_final)

        # Realtime corrected
        rt_corrected = date_dir / "realtime" / "05_classifier" / "fused_predictions_corrected.csv"
        rt_corr_final = final_dir / "realtime_final_predictions_corrected.csv"
        if _file_nonempty(rt_corrected):
            shutil.copy2(rt_corrected, rt_corr_final)
            manifest["final_outputs"]["realtime_corrected"] = str(rt_corr_final)

    # Submission-ready combined file
    submission = final_dir / "submission_ready.csv"
    frames = {}
    da_final = final_dir / "dayahead_final_predictions.csv"
    rt_final = final_dir / "realtime_final_predictions.csv"
    rt_corr_final = final_dir / "realtime_final_predictions_corrected.csv"

    if _file_nonempty(da_final):
        da_df = pd.read_csv(da_final)
        da_df = da_df.rename(columns={"y_pred": "dayahead_pred"})
        frames["dayahead"] = da_df

    if _file_nonempty(rt_final):
        rt_df = pd.read_csv(rt_final)
        rt_df = rt_df.rename(columns={"y_pred": "realtime_pred"})
        frames["realtime"] = rt_df

    if frames:
        # Merge on common keys
        merged = None
        for key, df in frames.items():
            merge_cols = ["target_day", "ds", "business_day", "hour_business", "period"]
            available = [c for c in merge_cols if c in df.columns]
            if merged is None:
                merged = df
            else:
                merged = merged.merge(df, on=available, how="outer", suffixes=("", f"_{key}"))

        if merged is not None:
            # Add corrected if available
            if _file_nonempty(rt_corr_final):
                corr_df = pd.read_csv(rt_corr_final)
                # Classifier corrected CSV has y_fused_corrected column
                pred_col = None
                for c in ["y_fused_corrected", "y_pred"]:
                    if c in corr_df.columns:
                        pred_col = c
                        break
                if pred_col:
                    corr_df = corr_df.rename(columns={pred_col: "realtime_pred_corrected"})
                    merge_keys = ["target_day", "ds"]
                    available_keys = [k for k in merge_keys if k in corr_df.columns]
                    if available_keys:
                        merged = merged.merge(
                            corr_df[available_keys + ["realtime_pred_corrected"]],
                            on=available_keys, how="left",
                        )

            merged.to_csv(submission, index=False)
            manifest["final_outputs"]["submission"] = str(submission)

    _write_manifest(date_dir, manifest)


# ── Single date pipeline ────────────────────────────────────────────
def run_production_for_date(args, date: str) -> dict:
    """Run full production pipeline for a single date."""
    date = pd.Timestamp(date).strftime("%Y-%m-%d")
    date_dir = _date_dir(args, date)
    date_dir.mkdir(parents=True, exist_ok=True)

    force = getattr(args, "force", False)
    targets = ["dayahead", "realtime"] if getattr(args, "target", "both") == "both" else [args.target]

    # Setup logging to file
    log_dir = date_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    try:
        # Check cache
        if _should_skip_entire(date_dir, force):
            msg = f"SKIP {date} — final outputs already exist"
            logger.info(msg)
            print(msg)
            manifest = _load_manifest(date_dir) or {}
            return manifest

        if force:
            logger.info("FORCED RERUN for %s", date)

        # Init manifest
        manifest = _init_manifest(date, targets)
        _write_manifest(date_dir, manifest)

        logger.info("=" * 60)
        logger.info("PRODUCTION PIPELINE: %s", date)
        logger.info("=" * 60)

        for target in targets:
            logger.info("--- %s ---", target)

            # Step 1: OOF pool
            try:
                _step_oof_pool(args, date_dir, target, manifest)
            except Exception as exc:
                logger.error("OOF step failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _write_manifest(date_dir, manifest)
                return manifest

            # Step 2: Model forecasts
            try:
                _step_model_forecasts(args, date_dir, target, manifest)
            except Exception as exc:
                logger.error("Forecast step failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _write_manifest(date_dir, manifest)
                return manifest

            # Step 3: Learner training
            try:
                _step_learner(args, date_dir, target, manifest)
            except Exception as exc:
                logger.error("Learner step failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _write_manifest(date_dir, manifest)
                return manifest

            # Step 4: Fusion
            try:
                _step_fusion(args, date_dir, target, manifest)
            except Exception as exc:
                logger.error("Fusion step failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _write_manifest(date_dir, manifest)
                return manifest

        # Step 5: Classifier (realtime only)
        if "realtime" in targets:
            _step_classifier(args, date_dir, manifest)

        # Final assembly
        _assemble_final_outputs(args, date_dir, manifest)

        # Validation
        from pipelines.output_validator import run_all_validations
        models_by_task = {t: FORMAL_MODELS_BY_TASK[t] for t in targets}
        passed, errors = run_all_validations(date_dir, tasks=targets, models_by_task=models_by_task)
        if errors:
            for err in errors:
                logger.warning("VALIDATION: %s", err)
            manifest["validation_errors"] = errors

            # Classify errors: critical vs non-fatal
            critical_keywords = [
                "!=24 rows", "hour_business missing", "null business_day",
                "negative weights", "Weight sums not", "Missing routing entries",
                "File not found", "File is empty",
            ]
            non_fatal_keywords = [
                "classifier", "Classifier", "Row count mismatch",
            ]

            critical_errors = []
            non_fatal_errors = []
            for err in errors:
                if any(kw in err for kw in non_fatal_keywords):
                    non_fatal_errors.append(err)
                elif any(kw in err for kw in critical_keywords):
                    critical_errors.append(err)
                else:
                    # Unknown error type → treat as non-fatal
                    non_fatal_errors.append(err)

            if critical_errors:
                manifest["status"] = "failed"
                manifest["critical_errors"] = critical_errors
            elif non_fatal_errors:
                manifest["status"] = "complete_with_warnings"
            else:
                manifest["status"] = "complete"
        else:
            manifest["status"] = "complete"

        manifest["finished_at"] = datetime.now().isoformat()
        _write_manifest(date_dir, manifest)

        logger.info("PRODUCTION PIPELINE %s: %s", manifest["status"], date)
        return manifest

    finally:
        logging.getLogger().removeHandler(file_handler)


# ── Date range pipeline ─────────────────────────────────────────────
def run_production_for_range(args, start: str, end: str) -> list[dict]:
    """Run production pipeline for each date in [start, end]."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    results = []

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        logger.info("─── Processing date: %s ───", date_str)
        result = run_production_for_date(args, date_str)
        results.append(result)
        current += pd.Timedelta(days=1)

    return results


# ── Top-level entry ─────────────────────────────────────────────────
def run_production_pipeline(args) -> dict | list[dict]:
    """Entry point called from main.py when --pipeline full."""
    # Determine date(s)
    if getattr(args, "start", None) and getattr(args, "end", None):
        return run_production_for_range(args, args.start, args.end)
    elif getattr(args, "date", None):
        return run_production_for_date(args, args.date)
    elif getattr(args, "pos_date", None):
        return run_production_for_date(args, args.pos_date)
    else:
        # Default: today
        today = datetime.now().strftime("%Y-%m-%d")
        logger.warning("No date specified, using today: %s", today)
        return run_production_for_date(args, today)
