"""R3D-Tap-GEF Production Pipeline.

6-step pipeline for electricity price forecasting:
  Step 0: Create/check outputs/{date}
  Step 1: Rolling 3-Day Validation Tap
  Step 2: Real Forecast (model predictions for target date)
  Step 3: R3D-Tap-GEF Learner (weight learning)
  Step 4: Fusion (weighted combination)
  Step 5: Classifier (realtime negative price)
  Step 6: Final Outputs

Usage:
    python main.py 2026-02-01
    python main.py 2026-02-01 --force
    python main.py 2026.2.1-2026.2.28
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Formal model lists ────────────────────────────────────────────────
FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["sgdfnet", "timemixer", "rt916", "timesfm"]
FORMAL_MODELS_BY_TASK = {
    "dayahead": FORMAL_DAYAHEAD_MODELS,
    "realtime": FORMAL_REALTIME_MODELS,
}


# ── Helpers ───────────────────────────────────────────────────────────
def _output_root() -> Path:
    return Path("outputs")


def _date_dir(dt: str) -> Path:
    return _output_root() / dt


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _months_back(d, months: int) -> date:
    """Return date that is `months` months before d."""
    if hasattr(d, 'date') and callable(d.date):
        d = d.date()
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, 28)
    return date(year, month, day)


# ── Manifest I/O ─────────────────────────────────────────────────────
def _save_manifest(ddir: Path, manifest: dict):
    path = ddir / "run_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)


# ── Step 0: Directory setup ──────────────────────────────────────────
def _step0_setup(dt: str, force: bool) -> Path | None:
    """Create or check the date directory.

    Returns the date_dir if we should proceed, None if already done.
    """
    ddir = _date_dir(dt)

    if ddir.exists():
        manifest_file = ddir / "run_manifest.json"
        if not force and manifest_file.exists():
            print(f"该日期已经预测过。如需重跑，请使用 --force。")
            return None
        if force:
            shutil.rmtree(ddir)
            logger.info("FORCE: removed existing %s", ddir)

    ddir.mkdir(parents=True, exist_ok=True)
    _ensure_dir(ddir / "logs")
    return ddir


# ── Step 1: Validation Tap ───────────────────────────────────────────
def _step1_validation_tap(args, ddir: Path, target: str, manifest: dict) -> Path:
    """Run 10-fold rolling 3-day validation tap."""
    from pipelines.validation_tap import run_validation_tap

    val_dir = _ensure_dir(ddir / target / "validation")
    tap_table = val_dir / "validation_tap_long_table.csv"

    if _file_nonempty(tap_table) and not getattr(args, "force", False):
        logger.info("SKIP %s validation tap — cached", target)
        manifest["steps"][f"{target}_validation"] = "skipped"
        return tap_table

    logger.info("STEP 1: %s validation tap", target)
    date_str = manifest["date"]

    extra_kwargs = {}
    if getattr(args, "fast_dev_run", False):
        extra_kwargs["fast_dev_run"] = True
    if getattr(args, "skip_rt916_validation", False):
        extra_kwargs["skip_models"] = ["rt916"]

    # --models filtering: compute skip_models for models NOT in the list
    specified_models = getattr(args, "models", None)
    if specified_models and specified_models not in (None, "all", ""):
        model_list = [m.strip() for m in specified_models.split(",") if m.strip()]
        all_target_models = FORMAL_MODELS_BY_TASK[target]
        skip = [m for m in all_target_models if m not in model_list]
        if skip:
            existing_skip = extra_kwargs.get("skip_models", [])
            extra_kwargs["skip_models"] = existing_skip + skip
            logger.info("  --models=%s -> skipping %s for %s", specified_models, skip, target)
        # Override fast_dev model limitation with explicit --models
        if extra_kwargs.get("fast_dev_run"):
            extra_kwargs["fast_dev_target_models"] = model_list

    # New: pass online update parameters
    extra_kwargs["timemixer_online_epochs"] = getattr(args, "timemixer_online_epochs", 3)
    extra_kwargs["timemixer_online_lr"] = getattr(args, "timemixer_online_lr", None)
    extra_kwargs["rt916_online_epochs"] = getattr(args, "rt916_online_epochs", 3)
    extra_kwargs["rt916_online_lr"] = getattr(args, "rt916_online_lr", None)
    extra_kwargs["timesfm_inference_mode"] = getattr(args, "timesfm_inference_mode", "daily")
    extra_kwargs["sgdfnet_fold_strategy"] = getattr(args, "sgdfnet_fold_strategy", "3x10")

    result_path = run_validation_tap(
        predict_date=date_str,
        target=target,
        data_path=args.data_path,
        output_dir=val_dir,
        force=getattr(args, "force", False),
        extra_kwargs=extra_kwargs if extra_kwargs else None,
    )

    if _file_nonempty(result_path):
        manifest["steps"][f"{target}_validation"] = "complete"
    else:
        manifest["steps"][f"{target}_validation"] = "failed"
        manifest["warnings"].append(f"{target} validation tap produced no data")

    _save_manifest(ddir, manifest)
    return result_path


# ── Step 2: Real Forecast ────────────────────────────────────────────
def _step2_real_forecast(args, ddir: Path, target: str, manifest: dict) -> Path:
    """Generate per-model forecasts for the target date."""
    from rolling_oof.contracts import FoldSpec
    from rolling_oof.scheduler import ADAPTER_REGISTRY, _init_registry

    _init_registry()

    real_dir = _ensure_dir(ddir / target / "real")
    all_forecasts = real_dir / "all_model_forecasts_long.csv"

    if _file_nonempty(all_forecasts) and not getattr(args, "force", False):
        logger.info("SKIP %s real forecast — cached", target)
        manifest["steps"][f"{target}_forecast"] = "skipped"
        return all_forecasts

    logger.info("STEP 2: %s real forecast", target)
    date_str = manifest["date"]
    models = FORMAL_MODELS_BY_TASK[target]

    # Fast dev: limit to models that have validation tap data
    fast_dev = getattr(args, "fast_dev_run", False)
    if fast_dev:
        # In fast_dev mode, only forecast with models that participated in validation tap
        val_dir = ddir / target / "validation"
        val_folds = val_dir / "folds"
        available = []
        for m in models:
            fold_dir = val_folds / "fold_09" / f"{m}_predictions.csv"
            if fold_dir.exists():
                available.append(m)
        if available:
            models = available
            logger.info("  fast_dev: limiting models to %s", models)
        else:
            models = ["lightgbm"]  # fallback
            logger.info("  fast_dev: no validation data, using lightgbm only")
    D = pd.Timestamp(date_str).date()

    fs = FoldSpec(
        fold_id=99,
        train_start=_months_back(D - timedelta(days=1), 6),
        train_end=D - timedelta(days=1),
        test_start=D,
        test_end=D,
        target_month="",
    )

    frames: list[pd.DataFrame] = []
    kwargs = {
        "training_months": 6,
        "val_ratio": 0.2,
        "seed": 42,
        "rolling_mode": "block",
        "block_days": 1,
    }

    for model_name in models:
        model_dir = _ensure_dir(real_dir / model_name)
        pred_file = model_dir / "forecast_predictions.csv"

        if _file_nonempty(pred_file) and not getattr(args, "force", False):
            logger.info("  SKIP %s/%s — cached", target, model_name)
            df = pd.read_csv(pred_file)
            frames.append(df)
            continue

        logger.info("  %s/%s: predicting...", target, model_name)
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            logger.warning("  Unknown model: %s", model_name)
            continue

        adapter = adapter_cls()
        if target not in adapter.supported_tasks:
            continue

        # For online models, check if checkpoint exists from validation tap
        model_kwargs = dict(kwargs)
        if model_name in ("timemixer", "rt916"):
            checkpoint_dir = ddir / target / "validation" / f"{model_name}_checkpoints"
            if checkpoint_dir.exists():
                model_kwargs["rolling_mode"] = "online"
                model_kwargs["checkpoint_dir"] = str(checkpoint_dir)
                logger.info("  %s: using checkpoint from validation tap", model_name)

        try:
            result = adapter.fold_train_predict(
                task=target, fold_spec=fs, data_path=args.data_path, **model_kwargs,
            )

            if not result.success or result.predictions_df is None:
                logger.warning("  %s/%s: %s", target, model_name, result.error_message)
                manifest["warnings"].append(f"{target}/{model_name}: {result.error_message}")
                continue

            df = _normalize_real_forecast(
                result.predictions_df, task=target, model_name=model_name, date_str=date_str,
            )
            df.to_csv(pred_file, index=False)
            frames.append(df)
            logger.info("  %s/%s: %d rows", target, model_name, len(df))

        except Exception as exc:
            logger.error("  %s/%s FAILED: %s", target, model_name, exc)
            manifest["warnings"].append(f"Forecast failed: {target}/{model_name}: {exc}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(all_forecasts, index=False)
        manifest["steps"][f"{target}_forecast"] = "complete"
    else:
        manifest["steps"][f"{target}_forecast"] = "failed: no predictions"

    _save_manifest(ddir, manifest)
    return all_forecasts


def _normalize_real_forecast(raw_df, *, task, model_name, date_str):
    """Normalize raw forecast to real forecast long table."""
    df = raw_df.copy()

    if "ds" not in df.columns:
        for col in ["timestamp", "datetime", "time", "date"]:
            if col in df.columns:
                df = df.rename(columns={col: "ds"})
                break

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if "hour_business" not in df.columns:
        df["hour_business"] = df["ds"].dt.hour.replace({0: 24}).astype(int)

    if "period" not in df.columns:
        df["period"] = df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8
            else ("9_16" if 9 <= h <= 16 else "17_24") if h <= 24 else "unknown"
        )

    if "business_day" not in df.columns:
        df["business_day"] = df["ds"].apply(
            lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.notna(t) and t.hour == 0
            else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
        )

    if "target_day" not in df.columns:
        df["target_day"] = df["business_day"]

    if "y_pred" not in df.columns:
        for col in ["pred", "prediction", "y", "price_pred"]:
            if col in df.columns:
                df["y_pred"] = pd.to_numeric(df[col], errors="coerce")
                break
    if "y_pred" not in df.columns:
        df["y_pred"] = float("nan")

    df["task"] = task
    df["model_name"] = model_name
    df["source"] = model_name
    df["run_mode"] = "real_forecast"
    df["created_at"] = datetime.now().isoformat()

    out_cols = [
        "task", "model_name", "target_day", "business_day", "ds",
        "hour_business", "period", "y_pred", "source", "run_mode", "created_at",
    ]
    available = [c for c in out_cols if c in df.columns]
    return df[available].copy()


# ── Step 3: R3D-Tap-GEF Learner ─────────────────────────────────────
def _load_previous_weights(ddir: Path, target: str) -> dict[str, float] | None:
    """Try to load previous day's weights for evidence shrinkage prior."""
    yesterday = (pd.Timestamp(ddir.name) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    yesterday_weights = Path("outputs") / yesterday / target / "fused" / "weights.csv"
    if yesterday_weights.exists() and yesterday_weights.stat().st_size > 0:
        try:
            df = pd.read_csv(yesterday_weights)
            df = df[df["task"] == target] if "task" in df.columns else df
            return dict(zip(df["model_name"], df["weight"]))
        except Exception:
            return None
    return None


def _step3_learner(args, ddir: Path, target: str, manifest: dict) -> Path:
    """Train the R3D-Tap-GEF learner on validation tap data."""
    from fusion.learners.r3d_tap_gef import run_r3d_tap_gef

    fused_dir = _ensure_dir(ddir / target / "fused")
    weights_csv = fused_dir / "weights.csv"

    if _file_nonempty(weights_csv) and not getattr(args, "force", False):
        logger.info("SKIP %s learner — cached", target)
        manifest["steps"][f"{target}_learner"] = "skipped"
        return fused_dir

    logger.info("STEP 3: %s R3D-Tap-GEF learner", target)

    tap_table = ddir / target / "validation" / "validation_tap_long_table.csv"
    if not _file_nonempty(tap_table):
        raise FileNotFoundError(f"Validation tap table not found: {tap_table}")

    tap_df = pd.read_csv(tap_table)
    if "task" in tap_df.columns:
        tap_df = tap_df[tap_df["task"] == target]

    output = run_r3d_tap_gef(
        tap_df,
        tau_block=getattr(args, "tau_block", 3.0),
        tau_horizon=getattr(args, "tau_horizon", 2.0),
        tau_days=getattr(args, "tau_days", 14.0),
        eta=getattr(args, "eta", 0.8),
        weight_floor=getattr(args, "weight_floor", 0.03),
        lambda_refit=getattr(args, "lambda_refit", 0.05),
        evidence_prior=getattr(args, "evidence_prior", 5.0),
        previous_weights=_load_previous_weights(ddir, target),
    )

    output.weights.to_csv(fused_dir / "weights.csv", index=False)
    output.routing_table.to_csv(fused_dir / "routing_table.csv", index=False)
    output.dynamic_weight_trace.to_csv(fused_dir / "dynamic_weight_trace.csv", index=False)
    output.candidate_metrics.to_csv(fused_dir / "candidate_metrics.csv", index=False)
    output.coverage_report.to_csv(fused_dir / "coverage_report.csv", index=False)

    debug = {
        "learner_mode": "r3d_tap_gef",
        "tau_block": getattr(args, "tau_block", 3.0),
        "tau_horizon": getattr(args, "tau_horizon", 2.0),
        "tau_days": getattr(args, "tau_days", 14.0),
        "eta": getattr(args, "eta", 0.8),
        "weight_floor": getattr(args, "weight_floor", 0.03),
        "lambda_refit": getattr(args, "lambda_refit", 0.05),
        "evidence_prior": getattr(args, "evidence_prior", 5.0),
        "warnings": output.manifest.get("warnings", []),
        "generated_at": datetime.now().isoformat(),
    }
    with open(fused_dir / "fused_debug.csv", "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, ensure_ascii=False)

    manifest["steps"][f"{target}_learner"] = "complete"
    _save_manifest(ddir, manifest)
    return fused_dir


# ── Step 4: Fusion ───────────────────────────────────────────────────
def _step4_fusion(args, ddir: Path, target: str, manifest: dict) -> Path:
    """Apply learned weights to fuse real forecasts."""
    fused_dir = ddir / target / "fused"
    fused_csv = fused_dir / "fused_predictions.csv"

    if _file_nonempty(fused_csv) and not getattr(args, "force", False):
        logger.info("SKIP %s fusion — cached", target)
        manifest["steps"][f"{target}_fusion"] = "skipped"
        return fused_csv

    logger.info("STEP 4: %s fusion", target)

    forecast_file = ddir / target / "real" / "all_model_forecasts_long.csv"
    weights_file = fused_dir / "weights.csv"

    if not _file_nonempty(forecast_file):
        raise FileNotFoundError(f"Forecast file not found: {forecast_file}")
    if not _file_nonempty(weights_file):
        raise FileNotFoundError(f"Weights file not found: {weights_file}")

    forecast_df = pd.read_csv(forecast_file)
    weights_df = pd.read_csv(weights_file)

    if "task" in weights_df.columns:
        weights_df = weights_df[weights_df["task"] == target]

    result, debug_df = _apply_weights(forecast_df, weights_df, target)

    if "y_pred" in result.columns and "y_fused" not in result.columns:
        result["y_fused"] = result["y_pred"]

    result.to_csv(fused_csv, index=False)

    # Write fused_debug.csv
    if not debug_df.empty:
        debug_df.to_csv(fused_dir / "fused_debug.csv", index=False)

    if not result.empty:
        day_counts = result.groupby("target_day").size()
        if not all(day_counts == 24):
            manifest["warnings"].append(f"{target} fusion: some days != 24 rows")

    manifest["steps"][f"{target}_fusion"] = "complete"
    _save_manifest(ddir, manifest)
    return fused_csv


def _apply_weights(forecast_df, weights_df, target):
    """Apply fusion weights to forecasts with per-hour re-normalization.

    For each (ds, hour_business):
      - Find non-NaN model predictions
      - Re-normalize their weights
      - y_fused = sum(w_m * y_pred_m)
    If all models missing for an hour: raise error.
    """
    result_frames: list[pd.DataFrame] = []
    debug_frames: list[pd.DataFrame] = []

    for (task, target_day, period), group in forecast_df.groupby(
        ["task", "target_day", "period"]
    ):
        w_mask = (weights_df["task"] == task) & (weights_df["period"] == period)
        period_weights = weights_df[w_mask]

        if period_weights.empty:
            models = group["model_name"].unique()
            n = len(models)
            w_dict = {m: 1.0 / n for m in models}
        else:
            w_dict = dict(zip(period_weights["model_name"], period_weights["weight"]))

        pivot = group.pivot_table(
            index=["ds", "hour_business", "business_day"],
            columns="model_name",
            values="y_pred",
            aggfunc="first",
        )

        # Per-hour re-normalization
        y_fused = np.full(len(pivot), np.nan)
        debug_rows = []

        for idx_pos in range(len(pivot)):
            row = pivot.iloc[idx_pos]
            ds_val = pivot.index[idx_pos][0]
            hb_val = pivot.index[idx_pos][1]

            # Find available (non-NaN) models
            available = [m for m in row.index if pd.notna(row.get(m, np.nan))]
            missing = [m for m in w_dict if m not in available]

            if not available:
                logger.error(
                    "FUSION ERROR: all models missing for %s/%s ds=%s hour=%s",
                    task, period, ds_val, hb_val,
                )
                raise ValueError(
                    f"All models missing for {task}/{period} ds={ds_val} hour={hb_val}"
                )

            # Re-normalize weights for available models
            avail_w = {m: w_dict.get(m, 0) for m in available if m in w_dict}
            total_w = sum(avail_w.values())
            if total_w > 0:
                avail_w = {m: w / total_w for m, w in avail_w.items()}
            else:
                n = len(available)
                avail_w = {m: 1.0 / n for m in available}

            y_fused[idx_pos] = sum(
                avail_w.get(m, 0) * row[m] for m in available
            )

            debug_rows.append({
                "task": task,
                "target_day": target_day,
                "period": period,
                "ds": ds_val,
                "hour_business": hb_val,
                "available_models": str(sorted(available)),
                "missing_models": str(sorted(missing)),
                "weight_summary": str({m: round(w, 4) for m, w in avail_w.items()}),
                "renormalized": len(missing) > 0,
            })

        result = pivot.reset_index()
        result["task"] = task
        result["target_day"] = target_day
        result["period"] = period
        result["y_pred"] = y_fused
        result["y_fused"] = y_fused
        result["available_models"] = [r["available_models"] for r in debug_rows]
        result["weight_summary"] = [r["weight_summary"] for r in debug_rows]

        result_frames.append(result)
        if debug_rows:
            debug_frames.append(pd.DataFrame(debug_rows))

    if result_frames:
        result_df = pd.concat(result_frames, ignore_index=True)
        debug_df = pd.concat(debug_frames, ignore_index=True) if debug_frames else pd.DataFrame()
        return result_df, debug_df
    return pd.DataFrame(), pd.DataFrame()


# ── Step 5: Classifier ───────────────────────────────────────────────
def _step5_classifier(args, ddir: Path, manifest: dict) -> Path | None:
    """Run negative price classifier on realtime fusion output."""
    final_dir = _ensure_dir(ddir / "realtime" / "final")
    corrected_csv = final_dir / "realtime_final_predictions_corrected.csv"
    report_json = final_dir / "classifier_report.json"

    if _file_nonempty(corrected_csv) and not getattr(args, "force", False):
        logger.info("SKIP classifier — cached")
        manifest["steps"]["realtime_classifier"] = "skipped"
        return corrected_csv

    fused_csv = ddir / "realtime" / "fused" / "fused_predictions.csv"
    if not _file_nonempty(fused_csv):
        logger.warning("No fusion output for classifier")
        manifest["steps"]["realtime_classifier"] = "skipped: no fusion"
        return None

    logger.info("STEP 5: negative price classifier")
    try:
        from fusion.classifier_bridge import run_classifier_pipeline

        compat_root = ddir / "realtime" / "final" / "compat_fusion"
        compat_rt = _ensure_dir(compat_root / "realtime")
        compat_fused = compat_rt / "fused_predictions.csv"

        fused_df = pd.read_csv(fused_csv)
        if "y_pred" in fused_df.columns and "y_fused" not in fused_df.columns:
            fused_df["y_fused"] = fused_df["y_pred"]
        fused_df.to_csv(compat_fused, index=False)

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
            compat_corrected = compat_rt / "fused_predictions_corrected.csv"
            if compat_corrected.exists():
                shutil.copy2(compat_corrected, corrected_csv)
                manifest["steps"]["realtime_classifier"] = "complete"
                manifest["classifier_corrections"] = result.get("corrected_hours", 0)
            else:
                manifest["steps"]["realtime_classifier"] = "failed: output missing"
                manifest["warnings"].append("Classifier completed but output missing")
        elif status == "skipped":
            reason = result.get("reason", "unknown")
            manifest["steps"]["realtime_classifier"] = f"skipped: {reason}"
            manifest["warnings"].append(f"Classifier skipped: {reason}")
        else:
            manifest["steps"]["realtime_classifier"] = f"failed: {status}"
            manifest["warnings"].append(f"Classifier failed: {result}")

        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(result if isinstance(result, dict) else {"status": str(result)}, f, indent=2)

    except Exception as exc:
        logger.warning("Classifier failed (non-fatal): %s", exc)
        manifest["steps"]["realtime_classifier"] = "failed"
        manifest["warnings"].append(f"Classifier failed: {exc}")

    _save_manifest(ddir, manifest)
    return corrected_csv if _file_nonempty(corrected_csv) else None


# ── Step 6: Final Outputs ────────────────────────────────────────────
def _step6_final_outputs(ddir: Path, manifest: dict):
    """Assemble final output files."""
    targets = manifest["targets"]

    for target in targets:
        target_final = _ensure_dir(ddir / target / "final")
        fused_csv = ddir / target / "fused" / "fused_predictions.csv"

        if not _file_nonempty(fused_csv):
            continue

        df = pd.read_csv(fused_csv)
        out_cols = ["target_day", "ds", "business_day", "hour_business", "period", "y_pred"]
        available = [c for c in out_cols if c in df.columns]
        final_name = f"{target}_final_predictions.csv"
        df[available].to_csv(target_final / final_name, index=False)

    final_dir = _ensure_dir(ddir / "final")

    da_src = ddir / "dayahead" / "final" / "dayahead_final_predictions.csv"
    da_dst = final_dir / "dayahead_final_predictions.csv"
    if _file_nonempty(da_src):
        shutil.copy2(da_src, da_dst)
        manifest["final_outputs"]["dayahead"] = str(da_dst)

    rt_src = ddir / "realtime" / "final" / "realtime_final_predictions.csv"
    rt_dst = final_dir / "realtime_final_predictions.csv"
    if _file_nonempty(rt_src):
        shutil.copy2(rt_src, rt_dst)
        manifest["final_outputs"]["realtime"] = str(rt_dst)

    rt_corr_src = ddir / "realtime" / "final" / "realtime_final_predictions_corrected.csv"
    rt_corr_dst = final_dir / "realtime_final_predictions_corrected.csv"
    if _file_nonempty(rt_corr_src):
        shutil.copy2(rt_corr_src, rt_corr_dst)
        manifest["final_outputs"]["realtime_corrected"] = str(rt_corr_dst)

    clf_report_src = ddir / "realtime" / "final" / "classifier_report.json"
    clf_report_dst = final_dir / "classifier_report.json"
    if _file_nonempty(clf_report_src):
        shutil.copy2(clf_report_src, clf_report_dst)

    _build_submission_ready(ddir, final_dir, manifest)
    _save_manifest(ddir, manifest)


def _build_submission_ready(ddir: Path, final_dir: Path, manifest: dict):
    """Build submission_ready.csv."""
    da_file = final_dir / "dayahead_final_predictions.csv"
    rt_file = final_dir / "realtime_final_predictions.csv"
    rt_corr_file = final_dir / "realtime_final_predictions_corrected.csv"

    frames: dict[str, pd.DataFrame] = {}
    if _file_nonempty(da_file):
        da = pd.read_csv(da_file).rename(columns={"y_pred": "dayahead_pred"})
        frames["dayahead"] = da
    if _file_nonempty(rt_file):
        rt = pd.read_csv(rt_file).rename(columns={"y_pred": "realtime_pred"})
        frames["realtime"] = rt

    if not frames:
        return

    merge_cols = ["target_day", "ds", "business_day", "hour_business", "period"]
    merged = None
    for key, df in frames.items():
        available = [c for c in merge_cols if c in df.columns]
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on=available, how="outer", suffixes=("", f"_{key}"))

    if merged is not None and _file_nonempty(rt_corr_file):
        corr = pd.read_csv(rt_corr_file)
        pred_col = next((c for c in ["y_fused_corrected", "y_pred"] if c in corr.columns), None)
        if pred_col:
            corr = corr.rename(columns={pred_col: "realtime_pred_corrected"})
            merge_keys = [k for k in ["target_day", "ds"] if k in corr.columns]
            if merge_keys:
                merged = merged.merge(
                    corr[merge_keys + ["realtime_pred_corrected"]],
                    on=merge_keys, how="left",
                )
        merged.to_csv(final_dir / "submission_ready.csv", index=False)
        manifest["final_outputs"]["submission"] = str(final_dir / "submission_ready.csv")


# ── Output validation ────────────────────────────────────────────────
def _run_output_validation(ddir: Path, targets: list[str], manifest: dict):
    """Run output validation checks."""
    from pipelines.r3d_output_validator import run_all_validations

    passed, errors = run_all_validations(ddir, predict_date=manifest["date"], tasks=targets)

    if errors:
        for err in errors:
            logger.warning("VALIDATION: %s", err)
        manifest["validation_errors"] = errors

        critical_kw = ["!=24 rows", "null business_day", "File not found", "File is empty", "sum not"]
        if any(any(kw in e for kw in critical_kw) for e in errors):
            manifest["status"] = "failed"
        else:
            manifest["status"] = "complete_with_warnings"
    else:
        manifest["status"] = "complete"


# ── Single date pipeline ─────────────────────────────────────────────
def run_production_for_date(args, dt: str) -> dict:
    """Run full R3D-Tap-GEF pipeline for a single date."""
    dt = pd.Timestamp(dt).strftime("%Y-%m-%d")
    force = getattr(args, "force", False)
    fast_dev = getattr(args, "fast_dev_run", False)

    ddir = _step0_setup(dt, force)
    if ddir is None:
        manifest_file = _date_dir(dt) / "run_manifest.json"
        if manifest_file.exists():
            with open(manifest_file, encoding="utf-8") as f:
                return json.load(f)
        return {"date": dt, "status": "skipped"}

    log_file = ddir / "logs" / "pipeline.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Fast dev run overrides
    if fast_dev:
        # Respect explicit --target; default to dayahead for fast dev
        explicit_target = getattr(args, "target", "both")
        if explicit_target and explicit_target != "both":
            targets = [explicit_target]
        else:
            targets = ["dayahead"]
        logger.info("FAST DEV RUN: %s only, 1 fold, 1 model, no classifier", targets)
    else:
        targets = ["dayahead", "realtime"] if getattr(args, "target", "both") == "both" else [args.target]

    manifest = {
        "date": dt,
        "status": "running",
        "pipeline_version": "r3d_tap_gef_v1",
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "targets": targets,
        "dayahead_models": FORMAL_DAYAHEAD_MODELS,
        "realtime_models": FORMAL_REALTIME_MODELS,
        "steps": {},
        "timing": {},
        "final_outputs": {},
        "warnings": [],
    }
    _save_manifest(ddir, manifest)

    try:
        logger.info("=" * 60)
        logger.info("R3D-Tap-GEF PRODUCTION: %s", dt)
        logger.info("=" * 60)

        for target in targets:
            logger.info("--- %s ---", target)

            try:
                t_step = time.time()
                _step1_validation_tap(args, ddir, target, manifest)
                manifest["timing"][f"{target}_validation_tap_seconds"] = round(time.time() - t_step, 1)
            except Exception as exc:
                logger.error("Validation tap failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _save_manifest(ddir, manifest)
                return manifest

            try:
                t_step = time.time()
                _step2_real_forecast(args, ddir, target, manifest)
                manifest["timing"][f"{target}_real_forecast_seconds"] = round(time.time() - t_step, 1)
            except Exception as exc:
                logger.error("Real forecast failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _save_manifest(ddir, manifest)
                return manifest

            try:
                t_step = time.time()
                _step3_learner(args, ddir, target, manifest)
                manifest["timing"][f"{target}_learner_seconds"] = round(time.time() - t_step, 1)
            except Exception as exc:
                logger.error("Learner failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _save_manifest(ddir, manifest)
                return manifest

            try:
                t_step = time.time()
                _step4_fusion(args, ddir, target, manifest)
                manifest["timing"][f"{target}_fusion_seconds"] = round(time.time() - t_step, 1)
            except Exception as exc:
                logger.error("Fusion failed for %s: %s", target, exc)
                manifest["status"] = "failed"
                _save_manifest(ddir, manifest)
                return manifest

        if "realtime" in targets:
            t_step = time.time()
            _step5_classifier(args, ddir, manifest)
            manifest["timing"]["classifier_seconds"] = round(time.time() - t_step, 1)

        t_step = time.time()
        _step6_final_outputs(ddir, manifest)
        manifest["timing"]["final_outputs_seconds"] = round(time.time() - t_step, 1)

        t_step = time.time()
        _run_output_validation(ddir, targets, manifest)
        manifest["timing"]["output_validation_seconds"] = round(time.time() - t_step, 1)

        manifest["finished_at"] = datetime.now().isoformat()

        # Generate runtime report
        timing = manifest.get("timing", {})
        total_seconds = sum(v for v in timing.values() if isinstance(v, (int, float)))
        manifest["timing"]["total_seconds"] = round(total_seconds, 1)

        # Per-model timing breakdown
        for target in targets:
            for model_name in FORMAL_MODELS_BY_TASK[target]:
                key = f"{target}_{model_name}_seconds"
                if key not in timing:
                    timing[key] = 0.0

        _save_manifest(ddir, manifest)

        # Print summary
        logger.info("=" * 60)
        logger.info("RUNTIME SUMMARY for %s:", dt)
        logger.info("  Total: %.1f seconds (%.1f minutes)", total_seconds, total_seconds / 60)
        for key, val in sorted(timing.items()):
            if key.endswith("_seconds") and key != "total_seconds":
                logger.info("  %s: %.1f s", key, val)
        logger.info("=" * 60)

        logger.info("R3D-Tap-GEF %s: %s", manifest["status"], dt)
        return manifest

    finally:
        logging.getLogger().removeHandler(file_handler)


# ── Date range ────────────────────────────────────────────────────────
def run_production_for_range(args, start: str, end: str) -> list[dict]:
    """Run pipeline for each date in [start, end]."""
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


# ── Top-level entry ──────────────────────────────────────────────────
def run_production_pipeline(args) -> dict | list[dict]:
    """Entry point called from main.py."""
    if getattr(args, "start", None) and getattr(args, "end", None):
        return run_production_for_range(args, args.start, args.end)
    elif getattr(args, "date", None):
        return run_production_for_date(args, args.date)
    elif getattr(args, "pos_date", None):
        return run_production_for_date(args, args.pos_date)
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        logger.warning("No date specified, using today: %s", today)
        return run_production_for_date(args, today)
