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
    models = FORMAL_MODELS_BY_TASK[target]

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
        "timemixer_rolling_mode": "daily",
        "timemixer_block_days": 7,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    for fold_info in fold_specs:
        fold_id = fold_info["fold_id"]
        fold_dir = folds_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(exist_ok=True)

        logger.info(
            "  Tap fold %d/10: train %s→%s, test %s→%s",
            fold_id, fold_info["train_start"], fold_info["train_end"],
            fold_info["test_start"], fold_info["test_end"],
        )

        # Build FoldSpec for adapter
        from rolling_oof.contracts import FoldSpec
        fs = FoldSpec(
            fold_id=fold_id,
            train_start=date.fromisoformat(fold_info["train_start"]),
            train_end=date.fromisoformat(fold_info["train_end"]),
            test_start=date.fromisoformat(fold_info["test_start"]),
            test_end=date.fromisoformat(fold_info["test_end"]),
            target_month="",  # not used for tap
        )

        for model_name in models:
            pred_file = fold_dir / f"{model_name}_predictions.csv"

            # Cache check
            if not force and _file_nonempty(pred_file):
                logger.info("    SKIP %s fold %d — cached", model_name, fold_id)
                df = pd.read_csv(pred_file)
                all_frames.append(df)
                fold_results.append({
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "status": "cached",
                })
                continue

            logger.info("    %s fold %d: training...", model_name, fold_id)
            adapter_cls = ADAPTER_REGISTRY.get(model_name)
            if adapter_cls is None:
                logger.warning("    Unknown model: %s", model_name)
                continue

            adapter = adapter_cls()
            if target not in adapter.supported_tasks:
                logger.info("    %s does not support %s", model_name, target)
                continue

            try:
                result = adapter.fold_train_predict(
                    task=target,
                    fold_spec=fs,
                    data_path=data_path,
                    **kwargs,
                )

                if not result.success:
                    logger.error(
                        "    %s fold %d FAILED: %s",
                        model_name, fold_id, result.error_message,
                    )
                    fold_results.append({
                        "fold_id": fold_id,
                        "model_name": model_name,
                        "status": "failed",
                        "error": result.error_message,
                    })
                    continue

                if result.predictions_df is None or result.predictions_df.empty:
                    logger.warning("    %s fold %d: no predictions", model_name, fold_id)
                    fold_results.append({
                        "fold_id": fold_id,
                        "model_name": model_name,
                        "status": "empty",
                    })
                    continue

                # Normalize to tap long table format
                df = _normalize_tap_predictions(
                    result.predictions_df,
                    task=target,
                    model_name=model_name,
                    fold_id=fold_id,
                    fold_info=fold_info,
                )

                df.to_csv(pred_file, index=False)
                all_frames.append(df)
                fold_results.append({
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "status": "complete",
                    "n_rows": len(df),
                })
                logger.info("    %s fold %d: %d rows", model_name, fold_id, len(df))

            except Exception as exc:
                logger.error("    %s fold %d exception: %s", model_name, fold_id, exc)
                fold_results.append({
                    "fold_id": fold_id,
                    "model_name": model_name,
                    "status": "exception",
                    "error": str(exc),
                })

    # Assemble tap long table
    if all_frames:
        tap_df = pd.concat(all_frames, ignore_index=True)
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


def _normalize_tap_predictions(
    raw_df: pd.DataFrame,
    *,
    task: str,
    model_name: str,
    fold_id: int,
    fold_info: dict,
) -> pd.DataFrame:
    """Normalize raw adapter predictions to tap long table format."""
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
    df["run_mode"] = "rolling_3day_validation_tap"
    df["created_at"] = datetime.now().isoformat()

    # Select standard columns
    out_cols = [
        "task", "model_name", "tap_fold_id", "train_start", "train_end",
        "test_start", "test_end", "cutoff_date", "target_day", "business_day",
        "ds", "hour_business", "period", "horizon_day", "age_block",
        "y_pred", "y_true", "source", "run_mode", "created_at",
    ]
    available = [c for c in out_cols if c in df.columns]
    return df[available].copy()
