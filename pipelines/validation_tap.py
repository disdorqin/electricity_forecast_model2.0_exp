"""Validation Tap: 10-fold rolling 3-day validation.

Generates 10 tap folds for a given prediction date D:
  Fold 0: train to D-31, predict D-30 ~ D-28
  Fold 1: train to D-28, predict D-27 ~ D-25
  ...
  Fold 9: train to D-4, predict D-3 ~ D-1

Calls the existing model adapters (fold_train_predict) for each fold.
Assembles results into validation_tap_long_table.csv.
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Fixed parameters ──────────────────────────────────────────────────
TRAINING_WINDOW_MONTHS = 6
ONLINE_BASE_TRAIN_MONTHS = 12        # TimeMixer / RT916: 12 个月 base train
LIGHT_MODEL_TRAIN_MONTHS = 6         # LightGBM / SGDFNet: 6 个月
TAP_FOLDS = 10
TAP_BLOCK_DAYS = 3

# Model lists
FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["sgdfnet", "timemixer", "rt916", "timesfm"]
FORMAL_MODELS_BY_TASK = {
    "dayahead": FORMAL_DAYAHEAD_MODELS,
    "realtime": FORMAL_REALTIME_MODELS,
}


def generate_tap_fold_specs(predict_date: str) -> list[dict]:
    """Generate 10 tap fold specs for a prediction date.

    Parameters
    ----------
    predict_date : str
        Target date in 'YYYY-MM-DD' format.

    Returns
    -------
    list[dict]
        Each dict has: fold_id, train_start, train_end, test_start, test_end,
        age_block, target_days (list of 3 dates).
    """
    D = pd.Timestamp(predict_date).date()
    folds: list[dict] = []

    for i in range(TAP_FOLDS):
        # Fold 0 oldest:  train_end = D-31, test D-30~D-28
        # Fold 9 newest:  train_end = D-4,  test D-3~D-1
        # General: train_end = D - 31 + 3*i
        train_end = D - timedelta(days=31 - 3 * i)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=TAP_BLOCK_DAYS - 1)

        # train_start = train_end - 6 months
        train_start = _months_back(train_end, TRAINING_WINDOW_MONTHS)

        age_block = TAP_FOLDS - 1 - i  # fold 9 is age 0, fold 0 is age 9

        target_days = [
            (test_start + timedelta(days=d)).isoformat()
            for d in range(TAP_BLOCK_DAYS)
        ]

        folds.append({
            "fold_id": i,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
            "age_block": age_block,
            "target_days": target_days,
        })

    return folds


def _months_back(d: date, months: int) -> date:
    """Return date that is `months` months before d."""
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, 28)  # safe day
    return date(year, month, day)


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def run_validation_tap(
    *,
    predict_date: str,
    target: str,
    data_path: str,
    output_dir: Path,
    force: bool = False,
    extra_kwargs: dict | None = None,
) -> Path:
    """Run the 10-fold validation tap for one target.

    Parameters
    ----------
    predict_date : str
        'YYYY-MM-DD' prediction date.
    target : str
        'dayahead' or 'realtime'.
    data_path : str
        Path to the raw data CSV.
    output_dir : Path
        e.g. outputs/{date}/{target}/validation/
    force : bool
        If True, rerun even if results exist.
    extra_kwargs : dict
        Extra keyword arguments for adapters.

    Returns
    -------
    Path to validation_tap_long_table.csv.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)

    tap_table_path = output_dir / "validation_tap_long_table.csv"
    manifest_path = output_dir / "tap_manifest.json"

    # Check cache
    if not force and _file_nonempty(tap_table_path):
        logger.info("SKIP validation tap for %s — already exists", target)
        return tap_table_path

    fold_specs = generate_tap_fold_specs(predict_date)
    models = list(FORMAL_MODELS_BY_TASK[target])

    # Handle extra_kwargs for dev mode and model skipping
    fast_dev = False
    skip_models = []
    if extra_kwargs:
        fast_dev = extra_kwargs.pop("fast_dev_run", False)
        skip_models = extra_kwargs.pop("skip_models", [])

    if fast_dev:
        fold_specs = fold_specs[-1:]  # only fold 9 (most recent)
        # If --models specified with fast-dev, use those; else first model
        fast_dev_models = extra_kwargs.get("fast_dev_target_models", None) if extra_kwargs else None
        if fast_dev_models:
            models = [m for m in fast_dev_models if m in models]
            logger.info("FAST DEV: %d fold, %d models (--models): %s", len(fold_specs), len(models), models)
        else:
            models = models[:1]  # only first model (lightgbm)
            logger.info("FAST DEV: %d fold, %d models: %s", len(fold_specs), len(models), models)

    if skip_models:
        models = [m for m in models if m not in skip_models]
        logger.info("Skipping models %s, remaining: %s", skip_models, models)

    all_frames: list[pd.DataFrame] = []
    fold_results: list[dict] = []

    # Initialize adapter registry
    from rolling_oof.adapters.base import BaseRollingAdapter
    from rolling_oof.scheduler import ADAPTER_REGISTRY, _init_registry
    _init_registry()

    kwargs = {
        "training_months": TRAINING_WINDOW_MONTHS,
        "val_ratio": 0.2,
        "seed": 42,
        "rolling_mode": "block",
        "block_days": TAP_BLOCK_DAYS,  # 3 — one train per fold
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    # ── Classify models by tap strategy ──────────────────────────────
    online_models = [m for m in models if m in ("timemixer", "rt916")]
    inference_models = [m for m in models if m in ("timesfm",)]
    rolling_models = [m for m in models if m in ("lightgbm",)]
    sgdfnet_models = [m for m in models if m in ("sgdfnet",)]

    logger.info(
        "Tap strategy — online: %s | inference: %s | rolling: %s | sgdfnet: %s",
        online_models, inference_models, rolling_models, sgdfnet_models,
    )

    # ── 1. Online models (timemixer, rt916): walk-forward across all folds ──
    # Build FoldSpec objects for all folds
    from rolling_oof.contracts import FoldSpec
    all_fold_specs = []
    for fold_info in fold_specs:
        fs = FoldSpec(
            fold_id=fold_info["fold_id"],
            train_start=date.fromisoformat(fold_info["train_start"]),
            train_end=date.fromisoformat(fold_info["train_end"]),
            test_start=date.fromisoformat(fold_info["test_start"]),
            test_end=date.fromisoformat(fold_info["test_end"]),
            target_month="",
        )
        all_fold_specs.append(fs)

    for model_name in online_models:
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            logger.warning("Unknown model: %s", model_name)
            continue
        adapter = adapter_cls()
        if target not in adapter.supported_tasks:
            logger.info("%s does not support %s", model_name, target)
            continue

        checkpoint_dir = output_dir / f"{model_name}_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Check cache for all folds
        all_cached = all(
            (folds_dir / f"fold_{fi['fold_id']:02d}" / f"{model_name}_predictions.csv").exists()
            and _file_nonempty(folds_dir / f"fold_{fi['fold_id']:02d}" / f"{model_name}_predictions.csv")
            for fi in fold_specs
        )
        if not force and all_cached:
            logger.info("  [online] %s: all folds cached, skipping", model_name)
            for fold_info in fold_specs:
                fold_id = fold_info["fold_id"]
                pred_file = folds_dir / f"fold_{fold_id:02d}" / f"{model_name}_predictions.csv"
                df = pd.read_csv(pred_file)
                all_frames.append(df)
                fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "cached"})
            continue

        logger.info("  [online] %s: base train once + %d block online updates", model_name, len(fold_specs))
        online_epochs = kwargs.get(f"{model_name}_online_epochs", 3)
        online_lr = kwargs.get(f"{model_name}_online_lr")

        try:
            if model_name == "timemixer":
                # New: buffered online — base train once + 3×10day blocks + seasonal replay
                from TimeMixer.buffered_online import run_online_month_buffered
                logger.info("    %s: calling run_online_month_buffered (3x10day + seasonal replay)...", model_name)
                combined_df = run_online_month_buffered(
                    task=target,
                    data_path=data_path,
                    predict_date=predict_date,
                    checkpoint_dir=str(checkpoint_dir),
                    train_months=ONLINE_BASE_TRAIN_MONTHS,  # P0-1: 12 months
                    online_epochs=online_epochs,
                    online_lr=online_lr,
                    lambda_seasonal=kwargs.get("lambda_seasonal", 0.3),
                    lambda_anchor=kwargs.get("lambda_anchor", 1e-4),
                    replay_buffer_mode=kwargs.get("replay_buffer_mode", "rotating_year"),
                )
            elif model_name == "rt916":
                # New: buffered online — base train once + 3×10day blocks + seasonal replay
                from rolling_oof.buffered_rt916 import run_rt916_online_month_buffered
                logger.info("    %s: calling run_rt916_online_month_buffered (3x10day + seasonal replay)...", model_name)
                combined_df = run_rt916_online_month_buffered(
                    task=target,
                    data_path=data_path,
                    predict_date=predict_date,
                    checkpoint_dir=str(checkpoint_dir),
                    train_months=ONLINE_BASE_TRAIN_MONTHS,  # P0-1: 12 months
                    online_epochs=online_epochs,
                    online_lr=online_lr,
                    lambda_seasonal=kwargs.get("lambda_seasonal", 0.3),
                    lambda_anchor=kwargs.get("lambda_anchor", 1e-4),
                    replay_buffer_mode=kwargs.get("replay_buffer_mode", "rotating_year"),
                )
            else:
                logger.error("    %s: unknown online model", model_name)
                continue

            if combined_df is None or combined_df.empty:
                logger.error("    %s: batch online returned no data", model_name)
                fold_results.append({"model_name": model_name, "status": "failed", "error": "No data from batch"})
                continue

            # Split combined_df into per-fold DataFrames and save
            for fold_info in fold_specs:
                fold_id = fold_info["fold_id"]
                fold_dir = folds_dir / f"fold_{fold_id:02d}"
                fold_dir.mkdir(exist_ok=True)
                pred_file = fold_dir / f"{model_name}_predictions.csv"

                fold_df = _split_fold_predictions(combined_df, fold_info, fold_id)
                if fold_df.empty:
                    logger.warning("    %s: no data for fold %d", model_name, fold_id)
                    continue

                fold_df = _normalize_tap_predictions(
                    fold_df, task=target, model_name=model_name,
                    fold_id=fold_id, fold_info=fold_info,
                    predict_date=predict_date,
                )
                fold_df["tap_source"] = fold_df.get("tap_source", "online_update")
                fold_df["source_confidence"] = fold_df.get("source_confidence", 0.95)
                fold_df["checkpoint_path"] = fold_df.get("checkpoint_path", str(checkpoint_dir))

                fold_df.to_csv(pred_file, index=False)
                all_frames.append(fold_df)
                fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "complete", "n_rows": len(fold_df)})
                logger.info("    %s fold %d: %d rows (online batch)", model_name, fold_id, len(fold_df))

        except Exception as exc:
            # P0-4: Do NOT fall back to per-fold online (which would re-train base 10 times).
            # Instead, let the buffered function handle its own fallback (single_train_range).
            logger.error("    %s buffered online FAILED: %s. Falling back to single_train_range.", model_name, exc, exc_info=True)
            try:
                if model_name == "timemixer":
                    combined_df = _timemixer_single_train_range(
                        target=target, data_path=data_path,
                        predict_date=predict_date, checkpoint_dir=str(checkpoint_dir),
                        fold_specs=all_fold_specs, folds_dir=folds_dir,
                        training_months=ONLINE_BASE_TRAIN_MONTHS,
                    )
                elif model_name == "rt916":
                    combined_df = _rt916_single_train_range(
                        target=target, data_path=data_path,
                        predict_date=predict_date, checkpoint_dir=str(checkpoint_dir),
                        fold_specs=all_fold_specs, folds_dir=folds_dir,
                        training_months=ONLINE_BASE_TRAIN_MONTHS,
                    )
                else:
                    combined_df = None

                if combined_df is None or combined_df.empty:
                    logger.error("    %s: single_train_range fallback also failed", model_name)
                    fold_results.append({"model_name": model_name, "status": "failed", "error": str(exc)})
                    continue

                # Save fallback results
                for fold_info in fold_specs:
                    fold_id = fold_info["fold_id"]
                    fold_dir = folds_dir / f"fold_{fold_id:02d}"
                    fold_dir.mkdir(exist_ok=True)
                    pred_file = fold_dir / f"{model_name}_predictions.csv"

                    fold_df = _split_fold_predictions(combined_df, fold_info, fold_id)
                    if fold_df.empty:
                        continue
                    fold_df = _normalize_tap_predictions(
                        fold_df, task=target, model_name=model_name,
                        fold_id=fold_id, fold_info=fold_info, predict_date=predict_date,
                    )
                    fold_df["tap_source"] = "single_train_range"
                    fold_df["source_confidence"] = 0.70
                    fold_df["checkpoint_path"] = str(checkpoint_dir)
                    fold_df.to_csv(pred_file, index=False)
                    all_frames.append(fold_df)
                    fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "fallback", "n_rows": len(fold_df)})
                    logger.info("    %s fold %d: %d rows (single_train_range fallback)", model_name, fold_id, len(fold_df))

            except Exception as exc2:
                logger.error("    %s: single_train_range fallback exception: %s", model_name, exc2)

    # ── 2. Inference models (timesfm): 30 daily cutoff inference ─────
    # Import once at module level for efficiency
    from pipelines.validation_tap_light_sgdf_timesfm import (
        run_timesfm_daily_validation_tap,
        normalize_block_predictions_to_tap,
        save_per_fold_predictions,
        save_runtime_report,
    )
    all_runtime_rows: list[dict] = []

    for model_name in inference_models:
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            logger.warning("Unknown model: %s", model_name)
            continue
        adapter = adapter_cls()
        if target not in adapter.supported_tasks:
            logger.info("%s does not support %s", model_name, target)
            continue

        logger.info("  [inference] %s: 30 daily cutoff inference", model_name)

        combined_df, runtime_rows = run_timesfm_daily_validation_tap(
            predict_date=predict_date,
            target=target,
            data_path=data_path,
            output_dir=output_dir,
            force=force,
            segment_count=kwargs.get("segment_count", 3),
            seed=kwargs.get("seed", 42),
            deterministic=kwargs.get("deterministic", True),
        )

        if not combined_df.empty:
            tap_df = normalize_block_predictions_to_tap(
                combined_df, predict_date=predict_date,
                task=target, model_name=model_name,
            )
            save_per_fold_predictions(tap_df, folds_dir, model_name)
            all_frames.append(tap_df)

            for fold_id in range(TAP_FOLDS):
                fold_rows = len(tap_df[tap_df["tap_fold_id"] == fold_id])
                fold_results.append({
                    "fold_id": fold_id, "model_name": model_name,
                    "status": "complete" if fold_rows > 0 else "empty",
                    "n_rows": fold_rows,
                })
            logger.info("    %s: %d rows across %d folds", model_name, len(tap_df), TAP_FOLDS)
        else:
            logger.error("    %s: NO predictions generated", model_name)
            fold_results.append({"model_name": model_name, "status": "failed", "error": "No predictions"})

        all_runtime_rows.extend(runtime_rows)

    # ── 3. Rolling models (lightgbm): 3x10 true rolling ──────────────
    from pipelines.validation_tap_light_sgdf_timesfm import (
        run_lightgbm_3x10_validation_tap,
    )

    for model_name in rolling_models:
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            logger.warning("Unknown model: %s", model_name)
            continue
        adapter = adapter_cls()
        if target not in adapter.supported_tasks:
            logger.info("%s does not support %s", model_name, target)
            continue

        logger.info("  [rolling] %s: 3x10 true rolling", model_name)

        combined_df, runtime_rows = run_lightgbm_3x10_validation_tap(
            predict_date=predict_date,
            target=target,
            data_path=data_path,
            output_dir=output_dir,
            force=force,
            training_months=kwargs.get("training_months", TRAINING_WINDOW_MONTHS),
            val_ratio=kwargs.get("val_ratio", 0.2),
        )

        if not combined_df.empty:
            tap_df = normalize_block_predictions_to_tap(
                combined_df, predict_date=predict_date,
                task=target, model_name=model_name,
            )
            save_per_fold_predictions(tap_df, folds_dir, model_name)
            all_frames.append(tap_df)

            for fold_id in range(TAP_FOLDS):
                fold_rows = len(tap_df[tap_df["tap_fold_id"] == fold_id])
                fold_results.append({
                    "fold_id": fold_id, "model_name": model_name,
                    "status": "complete" if fold_rows > 0 else "empty",
                    "n_rows": fold_rows,
                })
            logger.info("    %s: %d rows across %d folds", model_name, len(tap_df), TAP_FOLDS)
        else:
            logger.error("    %s: NO predictions generated", model_name)
            fold_results.append({"model_name": model_name, "status": "failed", "error": "No predictions"})

        all_runtime_rows.extend(runtime_rows)

    # ── 4. SGDFNet models: 3x10 true rolling ─────────────────────────
    from pipelines.validation_tap_light_sgdf_timesfm import (
        run_sgdfnet_3x10_validation_tap,
    )

    for model_name in sgdfnet_models:
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            logger.warning("Unknown model: %s", model_name)
            continue
        adapter = adapter_cls()
        if target not in adapter.supported_tasks:
            logger.info("%s does not support %s", model_name, target)
            continue

        fold_strategy = kwargs.get("sgdfnet_fold_strategy", "3x10")
        logger.info("  [sgdfnet] %s: 3x10 true rolling, strategy=%s", model_name, fold_strategy)

        combined_df, runtime_rows = run_sgdfnet_3x10_validation_tap(
            predict_date=predict_date,
            target=target,
            data_path=data_path,
            output_dir=output_dir,
            force=force,
            fold_strategy=fold_strategy,
        )

        if not combined_df.empty:
            tap_df = normalize_block_predictions_to_tap(
                combined_df, predict_date=predict_date,
                task=target, model_name=model_name,
            )
            save_per_fold_predictions(tap_df, folds_dir, model_name)
            all_frames.append(tap_df)

            for fold_id in range(TAP_FOLDS):
                fold_rows = len(tap_df[tap_df["tap_fold_id"] == fold_id])
                fold_results.append({
                    "fold_id": fold_id, "model_name": model_name,
                    "status": "complete" if fold_rows > 0 else "empty",
                    "n_rows": fold_rows,
                })
            logger.info("    %s: %d rows across %d folds", model_name, len(tap_df), TAP_FOLDS)
        else:
            logger.error("    %s: NO predictions generated", model_name)
            fold_results.append({"model_name": model_name, "status": "failed", "error": "No predictions"})

        all_runtime_rows.extend(runtime_rows)

    # Save runtime report if any model produced timing data
    if all_runtime_rows:
        save_runtime_report(all_runtime_rows, output_dir)

    # Assemble tap long table
    if all_frames:
        try:
            # Debug: check for duplicate columns in each frame
            for i, frame in enumerate(all_frames):
                dup_cols = [c for c in frame.columns if list(frame.columns).count(c) > 1]
                if dup_cols:
                    logger.error("Frame %d has duplicate columns: %s", i, dup_cols)
                    logger.error("  Columns: %s", list(frame.columns))
            tap_df = pd.concat(all_frames, ignore_index=True)
        except Exception as exc:
            logger.error("Failed to concat %d frames: %s", len(all_frames), exc)
            for i, frame in enumerate(all_frames):
                logger.error("Frame %d: shape=%s, cols=%s", i, frame.shape, list(frame.columns))
            raise

        # P0-2: Fill y_true from raw data (TimeMixer/RT916 buffered output has NaN y_true)
        tap_df = fill_y_true_from_data(tap_df, data_path, target)

        tap_df.to_csv(tap_table_path, index=False)
        logger.info("Validation tap: %d rows, %d models", len(tap_df), len(models))
    else:
        tap_df = pd.DataFrame()
        logger.error("Validation tap: NO predictions generated!")

    # Write manifest
    manifest = {
        "predict_date": predict_date,
        "target": target,
        "n_folds": TAP_FOLDS,
        "tap_block_days": TAP_BLOCK_DAYS,
        "training_window_months": TRAINING_WINDOW_MONTHS,
        "models": models,
        "n_rows": len(tap_df) if not tap_df.empty else 0,
        "fold_results": fold_results,
        "generated_at": datetime.now().isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, default=str)

    return tap_table_path


# ── Helper: per-fold online fallback ────────────────────────────────
def _run_online_per_fold(
    adapter,
    model_name: str,
    target: str,
    fold_specs: list[dict],
    all_fold_specs: list[FoldSpec],
    data_path: str,
    output_dir: Path,
    folds_dir: Path,
    checkpoint_dir: Path,
    force: bool,
    kwargs: dict,
    all_frames: list[pd.DataFrame],
    fold_results: list[dict],
):
    """Fallback: run online models fold-by-fold (may re-train base each time).

    This is the original per-fold logic, kept as fallback when the adapter
    does not support batch run_online_validation_tap.
    """
    from rolling_oof.contracts import FoldSpec

    for fold_info in fold_specs:
        fold_id = fold_info["fold_id"]
        fold_dir = folds_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)
        pred_file = fold_dir / f"{model_name}_predictions.csv"

        # Cache check
        if not force and _file_nonempty(pred_file):
            logger.info("    SKIP %s fold %d — cached", model_name, fold_id)
            df = pd.read_csv(pred_file)
            all_frames.append(df)
            fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "cached"})
            continue

        fs = all_fold_specs[fold_id]

        try:
            online_kwargs = {
                "rolling_mode": "online",
                "training_months": kwargs.get("training_months", TRAINING_WINDOW_MONTHS),
                "block_days": 3,
                "online_epochs": kwargs.get(f"{model_name}_online_epochs", 3),
                "online_lr": kwargs.get(f"{model_name}_online_lr"),
                "checkpoint_dir": str(checkpoint_dir),
            }

            logger.info("    %s fold %d: online update (per-fold)...", model_name, fold_id)
            result = adapter.fold_train_predict(
                task=target, fold_spec=fs, data_path=data_path,
                **online_kwargs,
            )

            if not result.success:
                logger.error("    %s fold %d FAILED: %s", model_name, fold_id, result.error_message)
                fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "failed", "error": result.error_message})
                continue

            if result.predictions_df is None or result.predictions_df.empty:
                fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "empty"})
                continue

            df = _normalize_tap_predictions(
                result.predictions_df, task=target, model_name=model_name,
                fold_id=fold_id, fold_info=fold_info,
                predict_date=kwargs.get("predict_date", ""),
            )
            df["tap_source"] = df.get("tap_source", "online_update")
            df["source_confidence"] = df.get("source_confidence", 0.95)
            df["checkpoint_path"] = df.get("checkpoint_path", str(checkpoint_dir))

            df.to_csv(pred_file, index=False)
            all_frames.append(df)
            fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "complete", "n_rows": len(df)})
            logger.info("    %s fold %d: %d rows (online per-fold)", model_name, fold_id, len(df))

        except Exception as exc:
            logger.error("    %s fold %d exception: %s", model_name, fold_id, exc)
            fold_results.append({"fold_id": fold_id, "model_name": model_name, "status": "exception", "error": str(exc)})


# ── Batch online helpers: direct pipeline calls (P0-2, P0-3 fix) ────

def _timemixer_online_batch(
    target: str,
    data_path: str,
    fold_specs: list[FoldSpec],
    checkpoint_dir: str,
    training_months: int = 6,
    online_epochs: int = 3,
    online_lr: float | None = None,
) -> pd.DataFrame | None:
    """TimeMixer online: base train once + sequential block updates.

    Creates a merged fold (D-30 ~ D-1) and calls run_monthly_reproduction
    with training_mode='online'. This ensures base train only happens once,
    followed by 10 sequential block predict+update steps.
    """
    import os as _os
    from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction

    _os.environ["OPTIM_NUM_WORKERS"] = "0"
    _os.environ["OPTIM_PIN_MEMORY"] = "0"

    if not fold_specs:
        return None

    # Sort by fold_id to ensure correct order
    fold_specs = sorted(fold_specs, key=lambda f: f.fold_id)

    # merged: test_start = D-30 (first fold), test_end_exclusive = D (last fold + 1)
    merged_test_start = fold_specs[0].test_start
    merged_test_end = fold_specs[-1].test_end

    output_dir = f"oof_runs/timemixer_batch_online_{target}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "[timemixer_online_batch] base train to %s, test %s ~ %s, %d folds",
        fold_specs[0].train_end, merged_test_start, merged_test_end, len(fold_specs),
    )

    run_cfg = RunConfig(
        data_path=data_path,
        output_dir=output_dir,
        month="",
        test_start=merged_test_start.isoformat(),
        test_end_exclusive=(merged_test_end + timedelta(days=1)).isoformat(),
        training_mode="online",
        block_days=3,
        train_months=training_months,
        checkpoint_dir=checkpoint_dir,
        online_epochs=online_epochs,
        online_lr=online_lr,
    )

    result = run_monthly_reproduction(run_cfg)
    if result is None:
        return None

    pred_key = "da_predictions" if target == "dayahead" else "rt_predictions"
    predictions = result.get(pred_key)
    if predictions is None:
        return None

    if isinstance(predictions, pd.DataFrame):
        combined = predictions
    elif isinstance(predictions, list):
        combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    else:
        return None

    if combined.empty:
        return None

    # Ensure ds is datetime
    combined = combined.copy()
    combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce")

    # Add tap_fold_id and age_block
    date_to_fold = {}
    for fs in fold_specs:
        current = fs.test_start
        while current <= fs.test_end:
            date_to_fold[current] = fs.fold_id
            current += timedelta(days=1)

    combined["tap_fold_id"] = combined["ds"].apply(
        lambda d: date_to_fold.get(d.normalize(), -1) if pd.notna(d) else -1
    )
    combined["age_block"] = combined["tap_fold_id"].apply(
        lambda fid: 9 - fid if 0 <= fid <= 9 else -1
    )

    logger.info(
        "[timemixer_online_batch] %d rows, folds=%s",
        len(combined), sorted(combined["tap_fold_id"].unique().tolist()),
    )
    return combined


def _rt916_online_batch(
    target: str,
    data_path: str,
    fold_specs: list[FoldSpec],
    checkpoint_dir: str,
    online_epochs: int = 3,
    online_lr: float | None = None,
) -> pd.DataFrame | None:
    """RT916 online: base train once + sequential block updates.

    Calls pipeline.online_predict_range with all fold_specs, which
    correctly does walk-forward with single base train.
    """
    import os as _os
    from RT916_SpikeFusionNet.pipeline import ModelPipeline

    _os.environ["OPTIM_AMP"] = "0"  # RT916: disable AMP during inference
    _os.environ["OPTIM_NUM_WORKERS"] = "0"

    pipeline = ModelPipeline()
    logger.info("[rt916_online_batch] %d folds, checkpoint=%s", len(fold_specs), checkpoint_dir)

    results = pipeline.online_predict_range(
        target=target,
        fold_specs=fold_specs,
        checkpoint_root=checkpoint_dir,
        online_epochs=online_epochs,
        online_lr=online_lr,
        data_path=data_path,
    )

    if not results:
        return None

    valid = [r for r in results if r is not None and len(r) > 0]
    if not valid:
        return None

    merged = pd.concat(valid, ignore_index=True)

    # Handle RT916's column naming (时刻 column)
    if "ds" not in merged.columns and "时刻" in merged.columns:
        merged["ds"] = pd.to_datetime(merged["时刻"], errors="coerce")
    else:
        merged["ds"] = pd.to_datetime(merged["ds"], errors="coerce")

    merged = merged.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)

    # Add tap_fold_id and age_block
    date_to_fold = {}
    for fs in fold_specs:
        current = fs.test_start
        while current <= fs.test_end:
            date_to_fold[current] = fs.fold_id
            current += timedelta(days=1)

    merged["tap_fold_id"] = merged["ds"].apply(
        lambda d: date_to_fold.get(d.normalize(), -1) if pd.notna(d) else -1
    )
    merged["age_block"] = merged["tap_fold_id"].apply(
        lambda fid: 9 - fid if 0 <= fid <= 9 else -1
    )

    logger.info(
        "[rt916_online_batch] %d rows, folds=%s",
        len(merged), sorted(merged["tap_fold_id"].unique().tolist()),
    )
    return merged


def _split_fold_predictions(
    combined_df: pd.DataFrame,
    fold_info: dict,
    fold_id: int,
) -> pd.DataFrame:
    """Split combined batch predictions to per-fold DataFrame."""
    # Try tap_fold_id first
    if "tap_fold_id" in combined_df.columns:
        fold_df = combined_df[combined_df["tap_fold_id"] == fold_id].copy()
        if not fold_df.empty:
            return fold_df

    # Fallback: filter by ds range
    test_start = pd.Timestamp(fold_info["test_start"])
    test_end = pd.Timestamp(fold_info["test_end"])
    if "ds" in combined_df.columns:
        mask = (pd.to_datetime(combined_df["ds"], errors="coerce") >= test_start) & (
            pd.to_datetime(combined_df["ds"], errors="coerce") <= test_end
        )
        return combined_df[mask].copy()

    return pd.DataFrame()


def _normalize_tap_predictions(
    raw_df: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    fold_id: int,
    fold_info: dict,
    predict_date: str | None = None,
) -> pd.DataFrame:
    """Normalize raw adapter predictions to tap long table format.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw predictions from the adapter.
    task : str
        'dayahead' or 'realtime'.
    model_name : str
        Name of the model.
    fold_id : int
        Fold index.
    fold_info : dict
        Fold specification dict from generate_tap_fold_specs.
    predict_date : str, optional
        'YYYY-MM-DD' prediction date, used to compute age_days.
    """
    df = raw_df.copy()

    # Ensure ds column exists and is datetime
    if "ds" not in df.columns:
        # Try to find timestamp column
        for col in ["timestamp", "datetime", "time", "date"]:
            if col in df.columns:
                df = df.rename(columns={col: "ds"})
                break

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    # Compute derived columns
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

    # Ensure y_pred exists
    if "y_pred" not in df.columns:
        for col in ["pred", "prediction", "y", "price_pred"]:
            if col in df.columns:
                df["y_pred"] = pd.to_numeric(df[col], errors="coerce")
                break

    if "y_pred" not in df.columns:
        df["y_pred"] = float("nan")

    # Ensure y_true exists (may not be available for all adapters)
    if "y_true" not in df.columns:
        for col in ["actual", "y", "price", "price_actual"]:
            if col in df.columns:
                df["y_true"] = pd.to_numeric(df[col], errors="coerce")
                break
    if "y_true" not in df.columns:
        df["y_true"] = float("nan")

    # Add horizon_day: which day within the 3-day test window (1, 2, or 3)
    test_start = pd.Timestamp(fold_info["test_start"])
    df["horizon_day"] = ((df["ds"].dt.normalize() - test_start).dt.days + 1).astype(int)
    # Clip to valid range
    df["horizon_day"] = df["horizon_day"].clip(1, TAP_BLOCK_DAYS)

    # Add metadata columns
    df["task"] = task
    df["model_name"] = model_name
    df["tap_fold_id"] = fold_id
    df["age_block"] = fold_info["age_block"]
    df["train_start"] = fold_info["train_start"]
    df["train_end"] = fold_info["train_end"]
    df["test_start"] = fold_info["test_start"]
    df["test_end"] = fold_info["test_end"]
    df["cutoff_date"] = fold_info["train_end"]
    df["source"] = model_name
    if "run_mode" not in df.columns:
        df["run_mode"] = "rolling_3day_validation_tap"  # P1: preserve existing run_mode
    if "created_at" not in df.columns:
        df["created_at"] = datetime.now().isoformat()

    # ── New provenance columns ───────────────────────────────────────
    if "tap_source" not in df.columns:
        df["tap_source"] = "rolling_cutoff"
    if "source_confidence" not in df.columns:
        df["source_confidence"] = 1.0
    if "checkpoint_path" not in df.columns:
        df["checkpoint_path"] = ""

    # age_days: how many days between each prediction date and the predict_date
    if "age_days" not in df.columns:
        if predict_date is not None:
            pd_date = pd.Timestamp(predict_date)
            df["age_days"] = (pd_date - df["ds"].dt.normalize()).dt.days.astype(int)
        else:
            df["age_days"] = -1  # unknown

    # Select standard columns — P1: preserve all metadata
    out_cols = [
        "task", "model_name", "tap_fold_id", "train_start", "train_end",
        "test_start", "test_end", "cutoff_date", "target_day", "business_day",
        "ds", "hour_business", "period", "horizon_day", "age_block",
        "y_pred", "y_true", "source", "run_mode", "created_at",
        "tap_source", "source_confidence", "age_days", "checkpoint_path",
        # P1 metadata: preserve buffered online / daily inference provenance
        "model_update_block_id", "learner_tap_fold_id",
        "replay_year", "replay_start", "replay_end",
        "lambda_seasonal", "lambda_anchor",
        "fold_strategy", "daily_inference_day", "cache_path",
        "predict_day", "cutoff_date_daily",
    ]
    available = [c for c in out_cols if c in df.columns]
    return df[available].copy()


# ═══════════════════════════════════════════════════════════════════════
# P0-2: fill_y_true_from_data
# ═══════════════════════════════════════════════════════════════════════

def fill_y_true_from_data(
    pred_df: pd.DataFrame,
    data_path: str,
    task: str,
) -> pd.DataFrame:
    """Fill y_true from raw data by exact ds merge.

    Parameters
    ----------
    pred_df : pd.DataFrame
        Predictions with ds column (timestamp).
    data_path : str
        Path to raw data CSV.
    task : str
        "dayahead" → day_ahead_clearing_price, "realtime" → realtime_price.

    Returns
    -------
    pd.DataFrame
        pred_df with y_true filled where possible.
    """
    df = pred_df.copy()

    target_col = "day_ahead_clearing_price" if task == "dayahead" else "realtime_price"

    try:
        raw = pd.read_csv(data_path, parse_dates=["ds"])
    except Exception:
        logger.warning("fill_y_true: cannot read %s", data_path)
        return df

    if "ds" not in raw.columns or target_col not in raw.columns:
        logger.warning("fill_y_true: missing ds or %s in raw data", target_col)
        return df

    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    raw["ds"] = pd.to_datetime(raw["ds"], errors="coerce")

    # Left merge: keep all predictions, fill y_true where available
    raw_subset = raw[["ds", target_col]].rename(columns={target_col: "_y_true_filled"})
    merged = df.merge(raw_subset, on="ds", how="left")

    # Only fill where y_true is NaN
    if "y_true" not in merged.columns:
        merged["y_true"] = np.nan
    nan_before = merged["y_true"].isna().sum()
    merged["y_true"] = merged["y_true"].fillna(merged["_y_true_filled"])
    nan_after = merged["y_true"].isna().sum()
    filled = nan_before - nan_after

    merged.drop(columns=["_y_true_filled"], inplace=True)

    coverage = 1.0 - nan_after / max(len(merged), 1)
    if coverage < 0.5:
        logger.warning(
            "fill_y_true: LOW coverage %.1f%% for %s (%d/%d NaN remain)",
            coverage * 100, task, nan_after, len(merged),
        )

    logger.info("fill_y_true: %s filled %d/%d NaN (coverage %.1f%%)", task, filled, nan_before, coverage * 100)
    return merged


# ═══════════════════════════════════════════════════════════════════════
# P0-4: single_train_range fallback helpers
# ═══════════════════════════════════════════════════════════════════════

def _timemixer_single_train_range(
    target: str,
    data_path: str,
    predict_date: str,
    checkpoint_dir: str,
    fold_specs: list,
    folds_dir: Path,
    training_months: int = 12,
) -> pd.DataFrame | None:
    """TimeMixer fallback: single train to D-31, predict D-30~D-1, block=30 days.

    Returns DataFrame with predictions sliced into learner_tap_fold_id 0..9.
    """
    import os as _os
    from datetime import timedelta
    from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction

    _os.environ["OPTIM_NUM_WORKERS"] = "0"
    _os.environ["OPTIM_PIN_MEMORY"] = "0"

    D = pd.Timestamp(predict_date).date()
    test_start = D - timedelta(days=30)
    test_end_exclusive = D

    output_dir = f"oof_runs/timemixer_st_fallback_{target}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    run_cfg = RunConfig(
        data_path=data_path,
        output_dir=output_dir,
        month="",
        test_start=test_start.isoformat(),
        test_end_exclusive=test_end_exclusive.isoformat(),
        training_mode="rolling",   # single train, no online update
        block_days=30,
        train_months=training_months,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("[timemixer_st_fallback] training_mode=rolling, %s→%s", test_start, test_end_exclusive)
    result = run_monthly_reproduction(run_cfg)
    if result is None:
        return None

    pred_key = "da_predictions" if target == "dayahead" else "rt_predictions"
    predictions = result.get(pred_key)
    if predictions is None:
        return None

    if isinstance(predictions, list):
        combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    else:
        combined = predictions

    if combined.empty:
        return None

    combined = combined.copy()
    combined["ds"] = pd.to_datetime(combined["ds"], errors="coerce")

    # Slice into learner_tap_fold_id 0..9
    date_to_fold = {}
    for fs in fold_specs:
        current = fs.test_start
        while current <= fs.test_end:
            date_to_fold[current] = fs.fold_id
            current += timedelta(days=1)

    combined["tap_fold_id"] = combined["ds"].apply(
        lambda d: date_to_fold.get(d.normalize(), -1) if pd.notna(d) else -1
    )
    combined["learner_tap_fold_id"] = combined["tap_fold_id"]
    combined["age_block"] = combined["tap_fold_id"].apply(lambda fid: 9 - fid if 0 <= fid <= 9 else -1)
    combined["model_update_block_id"] = 0
    combined["tap_source"] = "single_train_range"
    combined["source_confidence"] = 0.70
    combined["run_mode"] = "single_train_range"

    logger.info("[timemixer_st_fallback] %d rows", len(combined))
    return combined


def _rt916_single_train_range(
    target: str,
    data_path: str,
    predict_date: str,
    checkpoint_dir: str,
    fold_specs: list,
    folds_dir: Path,
    training_months: int = 12,
) -> pd.DataFrame | None:
    """RT916 fallback: single train to D-31, predict D-30~D-1."""
    from rolling_oof.buffered_rt916 import run_rt916_online_month_buffered

    # RT916 already has fallback built into run_rt916_online_month_buffered.
    # But if that fails too, try a direct pipeline call.
    return run_rt916_online_month_buffered(
        task=target,
        data_path=data_path,
        predict_date=predict_date,
        checkpoint_dir=checkpoint_dir,
        train_months=training_months,
        online_epochs=2,
    )
