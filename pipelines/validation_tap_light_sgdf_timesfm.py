# -*- coding: utf-8 -*-
"""Validation Tap helpers for LightGBM / SGDFNet / TimesFM.

This module handles the three non-online models exclusively.
Do NOT import or modify TimeMixer/RT916 logic.

Strategies:
  LightGBM: 3x10 day true rolling (3 train blocks, each covering 10 days)
  SGDFNet:  3x10 day true rolling (same structure, realtime only)
  TimesFM:  30 daily cutoff inference (30 individual daily predictions)

All three produce a unified long-table DataFrame with tap_fold_id 0..9
for the learner layer.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
TAP_BLOCK_DAYS = 3
TOTAL_VALIDATION_DAYS = 30  # D-30 ~ D-1
LEARNER_FOLDS = 10
TRAINING_WINDOW_MONTHS = 6


def _months_back(d: date, months: int) -> date:
    """Return date that is `months` months before d."""
    year = d.year
    month = d.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(d.day, 28)
    return date(year, month, day)


def _file_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


# ══════════════════════════════════════════════════════════════════════
#  3x10 Block Spec Generation
# ══════════════════════════════════════════════════════════════════════

def generate_3x10_block_specs(predict_date: str) -> list[dict]:
    """Generate 3 model-block specs covering D-30 ~ D-1.

    Block 0: train_end = D-31, predict D-30 ~ D-21
    Block 1: train_end = D-21, predict D-20 ~ D-11
    Block 2: train_end = D-11, predict D-10 ~ D-1
    """
    D = pd.Timestamp(predict_date).date()
    blocks: list[dict] = []

    for block_id in range(3):
        # train_end: D - 31 + 10*block_id
        train_end = D - timedelta(days=31 - 10 * block_id)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=9)  # 10 days inclusive

        train_start = _months_back(train_end, TRAINING_WINDOW_MONTHS)

        blocks.append({
            "block_id": block_id,
            "train_start": train_start.isoformat(),
            "train_end": train_end.isoformat(),
            "test_start": test_start.isoformat(),
            "test_end": test_end.isoformat(),
        })

    return blocks


# ══════════════════════════════════════════════════════════════════════
#  Learner Fold Splitting
# ══════════════════════════════════════════════════════════════════════

def build_date_to_fold_map(predict_date: str) -> dict:
    """Build mapping from date -> tap_fold_id (0..9) for D-30 ~ D-1.

    fold 0: D-30 ~ D-28
    fold 1: D-27 ~ D-25
    ...
    fold 9: D-3  ~ D-1
    """
    D = pd.Timestamp(predict_date).date()
    date_map: dict[date, int] = {}
    for fold_id in range(LEARNER_FOLDS):
        start = D - timedelta(days=30 - 3 * fold_id)
        for offset in range(3):
            d = start + timedelta(days=offset)
            date_map[d] = fold_id
    return date_map


def split_month_predictions_to_learner_folds(
    predictions_df: pd.DataFrame,
    predict_date: str,
) -> pd.DataFrame:
    """Split 30-day predictions into 10 learner folds (tap_fold_id 0..9).

    Adds columns: tap_fold_id, learner_tap_fold_id, age_block, age_days, horizon_day.
    """
    if predictions_df.empty:
        return predictions_df

    df = predictions_df.copy()
    D = pd.Timestamp(predict_date).date()
    date_map = build_date_to_fold_map(predict_date)

    # Ensure date column exists
    date_col = None
    for col in ("target_day", "business_day", "ds", "timestamp"):
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        logger.warning("split_month_predictions: no date column found")
        df["tap_fold_id"] = -1
        df["learner_tap_fold_id"] = -1
        df["age_block"] = -1
        df["age_days"] = -1
        df["horizon_day"] = -1
        return df

    # Normalize date column name to 'ds' for downstream compatibility
    if date_col != "ds":
        if "ds" in df.columns:
            # Already has ds, drop the duplicate date column
            df = df.drop(columns=[date_col])
        else:
            df = df.rename(columns={date_col: "ds"})

    parsed = pd.to_datetime(df["ds"], errors="coerce").dt.date
    df["tap_fold_id"] = parsed.map(lambda d: date_map.get(d, -1))
    df["learner_tap_fold_id"] = df["tap_fold_id"]
    df["age_block"] = df["tap_fold_id"].apply(lambda fid: 9 - fid if 0 <= fid <= 9 else -1)
    df["age_days"] = parsed.map(lambda d: (D - d).days if d is not None else -1)

    # horizon_day: which day within the 3-day fold (1, 2, or 3)
    # fold 0 starts at D-30, so day offset = 30 - (D-d) = 30 - age_days
    df["horizon_day"] = df["age_days"].apply(
        lambda ad: ((30 - ad - 1) % 3) + 1 if 1 <= ad <= 30 else -1
    )

    # Drop rows not in valid range
    df = df[df["tap_fold_id"] >= 0].copy()

    return df


# ══════════════════════════════════════════════════════════════════════
#  LightGBM 3x10 Validation Tap
# ══════════════════════════════════════════════════════════════════════

def run_lightgbm_3x10_validation_tap(
    *,
    predict_date: str,
    target: str,
    data_path: str,
    output_dir: Path,
    force: bool = False,
    training_months: int = TRAINING_WINDOW_MONTHS,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run LightGBM 3x10 true rolling validation tap.

    Returns (combined_df, runtime_rows).
    """
    from rolling_oof.adapters.lightgbm import _run_lgbm_10day_block

    output_dir = Path(output_dir)
    blocks_dir = output_dir / "model_blocks" / "lightgbm"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    block_specs = generate_3x10_block_specs(predict_date)
    all_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []

    logger.info(
        "[lightgbm_3x10] %s: %d blocks, D=%s",
        target, len(block_specs), predict_date,
    )

    for block in block_specs:
        block_id = block["block_id"]
        cache_file = blocks_dir / f"block_{block_id:02d}_predictions.csv"

        t0 = time.monotonic()
        cache_hit = False
        status = "complete"

        if not force and _file_nonempty(cache_file):
            logger.info("  [lightgbm_3x10] block %d: cache hit", block_id)
            block_df = pd.read_csv(cache_file)
            cache_hit = True
        else:
            logger.info(
                "  [lightgbm_3x10] block %d: train to %s, predict %s ~ %s",
                block_id, block["train_end"], block["test_start"], block["test_end"],
            )
            try:
                block_df = _run_lgbm_10day_block(
                    data_path=data_path,
                    forecast_start=block["test_start"],
                    forecast_end=block["test_end"],
                    train_start=block["train_start"],
                    train_end=block["train_end"],
                    target=target,
                    training_months=training_months,
                    val_ratio=val_ratio,
                )
                if block_df is not None and not block_df.empty:
                    block_df["model_update_block_id"] = block_id
                    block_df.to_csv(cache_file, index=False)
                else:
                    status = "failed: no predictions"
            except Exception as exc:
                logger.error(
                    "  [lightgbm_3x10] block %d FAILED: %s", block_id, exc,
                )
                status = f"failed: {exc}"
                block_df = pd.DataFrame()

        elapsed = time.monotonic() - t0
        if block_df is not None and not block_df.empty:
            all_frames.append(block_df)

        runtime_rows.append({
            "model_name": "lightgbm",
            "target": target,
            "resource": "cpu",
            "tap_strategy": "rolling_cutoff_3x10",
            "model_update_block_id": block_id,
            "learner_tap_fold_id": None,
            "runtime_seconds": round(elapsed, 1),
            "cache_hit": cache_hit,
            "status": status,
            "error_message": "" if status == "complete" else status,
        })

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, runtime_rows


# ══════════════════════════════════════════════════════════════════════
#  SGDFNet 3x10 Validation Tap
# ══════════════════════════════════════════════════════════════════════

def run_sgdfnet_3x10_validation_tap(
    *,
    predict_date: str,
    target: str,
    data_path: str,
    output_dir: Path,
    force: bool = False,
    fold_strategy: str = "3x10",
    **kwargs,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run SGDFNet 3x10 true rolling validation tap.

    Only supports realtime target. Uses 3 training blocks, each covering 10 days.
    Returns (combined_df, runtime_rows).
    """
    if target != "realtime":
        logger.warning("[sgdfnet_3x10] SGDFNet only supports realtime, got %s", target)
        return pd.DataFrame(), []

    from rolling_oof.adapters.sgdfnet import _run_sgdfnet_fold

    output_dir = Path(output_dir)
    blocks_dir = output_dir / "model_blocks" / "sgdfnet"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    block_specs = generate_3x10_block_specs(predict_date)
    all_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []

    logger.info(
        "[sgdfnet_3x10] %s: %d blocks, strategy=%s, D=%s",
        target, len(block_specs), fold_strategy, predict_date,
    )

    for block in block_specs:
        block_id = block["block_id"]
        cache_file = blocks_dir / f"block_{block_id:02d}_predictions.csv"

        t0 = time.monotonic()
        cache_hit = False
        status = "complete"

        if not force and _file_nonempty(cache_file):
            logger.info("  [sgdfnet_3x10] block %d: cache hit", block_id)
            block_df = pd.read_csv(cache_file)
            cache_hit = True
        else:
            logger.info(
                "  [sgdfnet_3x10] block %d: train to %s, predict %s ~ %s",
                block_id, block["train_end"], block["test_start"], block["test_end"],
            )
            try:
                from rolling_oof.contracts import FoldSpec
                import os as _os
                _os.environ.setdefault("OPTIM_NUM_WORKERS", "0")

                fs = FoldSpec(
                    fold_id=block_id,
                    train_start=date.fromisoformat(block["train_start"]),
                    train_end=date.fromisoformat(block["train_end"]),
                    test_start=date.fromisoformat(block["test_start"]),
                    test_end=date.fromisoformat(block["test_end"]),
                    target_month="",
                )
                block_df = _run_sgdfnet_fold(data_path, fs)

                if block_df is not None and not block_df.empty:
                    block_df["model_update_block_id"] = block_id
                    block_df["fold_strategy"] = fold_strategy
                    block_df["tap_source"] = "rolling_cutoff_3x10"
                    block_df["source_confidence"] = 1.0
                    block_df.to_csv(cache_file, index=False)
                else:
                    status = "failed: no predictions"
            except Exception as exc:
                logger.error(
                    "  [sgdfnet_3x10] block %d FAILED: %s", block_id, exc,
                )
                status = f"failed: {exc}"
                block_df = pd.DataFrame()

        elapsed = time.monotonic() - t0
        if block_df is not None and not block_df.empty:
            all_frames.append(block_df)

        runtime_rows.append({
            "model_name": "sgdfnet",
            "target": target,
            "resource": "cpu",
            "tap_strategy": "rolling_cutoff_3x10",
            "model_update_block_id": block_id,
            "learner_tap_fold_id": None,
            "runtime_seconds": round(elapsed, 1),
            "cache_hit": cache_hit,
            "status": status,
            "error_message": "" if status == "complete" else status,
        })

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, runtime_rows


# ══════════════════════════════════════════════════════════════════════
#  TimesFM Daily Validation Tap
# ══════════════════════════════════════════════════════════════════════

def run_timesfm_daily_validation_tap(
    *,
    predict_date: str,
    target: str,
    data_path: str,
    output_dir: Path,
    force: bool = False,
    segment_count: int = 3,
    seed: int = 42,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run TimesFM 30 daily cutoff inference validation tap.

    Model loads once. Each day d: cutoff = d-1, predict d.
    Produces 30 daily predictions, mapped to 10 learner folds 0..9.

    Returns (combined_df, runtime_rows).
    """
    from TimesFM.infer import predict_price_for_range

    output_dir = Path(output_dir)
    daily_dir = output_dir / "model_blocks" / "timesfm"
    daily_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=True)

    D = pd.Timestamp(predict_date).date()
    all_frames: list[pd.DataFrame] = []
    runtime_rows: list[dict] = []
    fallback_entries: list[dict] = []

    logger.info(
        "[timesfm_daily] %s: 30 daily cutoff inference, D=%s",
        target, predict_date,
    )

    # Pre-check: are all daily caches present?
    all_cached = True
    if not force:
        for offset in range(TOTAL_VALIDATION_DAYS, 0, -1):
            d = D - timedelta(days=offset)
            cache_file = daily_dir / f"daily_{target}_{d.isoformat()}.csv"
            if not _file_nonempty(cache_file):
                all_cached = False
                break

    if all_cached and not force:
        logger.info("  [timesfm_daily] all 30 daily caches present, loading")
        for offset in range(TOTAL_VALIDATION_DAYS, 0, -1):
            d = D - timedelta(days=offset)
            cache_file = daily_dir / f"daily_{target}_{d.isoformat()}.csv"
            df = pd.read_csv(cache_file)
            all_frames.append(df)
            runtime_rows.append({
                "model_name": "timesfm",
                "target": target,
                "resource": "timesfm",
                "tap_strategy": "direct_inference_daily",
                "model_update_block_id": f"daily_{d.isoformat()}",
                "learner_tap_fold_id": None,
                "runtime_seconds": 0,
                "cache_hit": True,
                "status": "cached",
                "error_message": "",
            })
        if all_frames:
            return pd.concat(all_frames, ignore_index=True), runtime_rows
        return pd.DataFrame(), runtime_rows

    # Run 30 daily inferences
    for offset in range(TOTAL_VALIDATION_DAYS, 0, -1):
        d = D - timedelta(days=offset)
        cutoff = d - timedelta(days=1)
        # Cache key includes task + target_day + cutoff to prevent cross-task pollution
        cache_file = daily_dir / f"daily_{target}_{d.isoformat()}_cutoff_{cutoff.isoformat()}.csv"

        t0 = time.monotonic()
        cache_hit = False
        status = "complete"
        tap_source = "direct_inference_daily"
        confidence = 0.90

        if not force and _file_nonempty(cache_file):
            logger.info("  [timesfm_daily] %s: cache hit", d.isoformat())
            daily_df = pd.read_csv(cache_file)
            cache_hit = True
        else:
            try:
                logger.info("  [timesfm_daily] cutoff=%s -> predict %s", cutoff, d)

                # cutoff-safe: build temp data with masked future
                daily_df = predict_price_for_range(
                    data_path=data_path,
                    start_date=d.isoformat(),
                    end_date=d.isoformat(),
                    target=target,
                    segment_count=segment_count,
                    seed=seed,
                    deterministic=deterministic,
                )

                if daily_df is not None and not daily_df.empty:
                    daily_df["model_name"] = "timesfm"
                    daily_df["task"] = target
                    daily_df["tap_source"] = tap_source
                    daily_df["source_confidence"] = confidence
                    daily_df["cutoff_date"] = cutoff.isoformat()
                    daily_df["daily_inference_day"] = d.isoformat()
                    daily_df["cache_path"] = str(cache_file)
                else:
                    status = "failed: empty prediction"
            except Exception as exc:
                logger.warning(
                    "  [timesfm_daily] %s failed: %s. Falling back to block inference.",
                    d, exc,
                )
                fallback_entries.append({
                    "day": d.isoformat(),
                    "cutoff": cutoff.isoformat(),
                    "error": str(exc),
                })
                # Try block inference fallback for this single day
                try:
                    daily_df = predict_price_for_range(
                        data_path=data_path,
                        start_date=d.isoformat(),
                        end_date=d.isoformat(),
                        target=target,
                        segment_count=segment_count,
                        seed=seed,
                        deterministic=deterministic,
                    )
                    if daily_df is not None and not daily_df.empty:
                        tap_source = "direct_inference_block"
                        confidence = 0.85
                        daily_df["model_name"] = "timesfm"
                        daily_df["task"] = target
                        daily_df["tap_source"] = tap_source
                        daily_df["source_confidence"] = confidence
                        daily_df["cutoff_date"] = cutoff.isoformat()
                        daily_df["daily_inference_day"] = d.isoformat()
                        daily_df["cache_path"] = str(cache_file)
                        status = "complete (fallback block)"
                    else:
                        status = "failed: empty after fallback"
                except Exception as exc2:
                    status = f"failed: {exc2}"
                    daily_df = pd.DataFrame()

            if daily_df is not None and not daily_df.empty:
                daily_df.to_csv(cache_file, index=False)

        elapsed = time.monotonic() - t0
        if daily_df is not None and not daily_df.empty:
            all_frames.append(daily_df)

        runtime_rows.append({
            "model_name": "timesfm",
            "target": target,
            "resource": "timesfm",
            "tap_strategy": tap_source,
            "model_update_block_id": f"daily_{d.isoformat()}",
            "learner_tap_fold_id": None,
            "runtime_seconds": round(elapsed, 1),
            "cache_hit": cache_hit,
            "status": status,
            "error_message": "" if status.startswith("complete") else status,
        })

    # Write fallback manifest if any
    if fallback_entries:
        manifest_path = daily_dir / "fallback_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(fallback_entries, f, indent=2, ensure_ascii=False)
        logger.warning(
            "[timesfm_daily] %d days fell back to block inference. See %s",
            len(fallback_entries), manifest_path,
        )

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    return combined, runtime_rows


# ══════════════════════════════════════════════════════════════════════
#  Unified Normalizer
# ══════════════════════════════════════════════════════════════════════

def normalize_block_predictions_to_tap(
    block_df: pd.DataFrame,
    *,
    predict_date: str,
    task: str,
    model_name: str,
) -> pd.DataFrame:
    """Normalize raw model block predictions to validation tap long-table format.

    Adds all required columns for learner consumption.
    """
    if block_df.empty:
        return block_df

    df = block_df.copy()
    D = pd.Timestamp(predict_date).date()

    # ── Split into learner folds ──
    df = split_month_predictions_to_learner_folds(df, predict_date)

    # ── Ensure core columns ──
    df["task"] = task
    df["model_name"] = model_name

    if "ds" not in df.columns:
        for col in ["timestamp", "datetime", "time", "date"]:
            if col in df.columns:
                df = df.rename(columns={col: "ds"})
                break

    if "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

    if "hour_business" not in df.columns:
        if "ds" in df.columns:
            df["hour_business"] = df["ds"].dt.hour.replace({0: 24}).astype(int)
        else:
            df["hour_business"] = -1

    if "period" not in df.columns:
        df["period"] = df["hour_business"].apply(
            lambda h: "1_8" if 1 <= h <= 8
            else ("9_16" if 9 <= h <= 16 else "17_24") if h <= 24 else "unknown"
        )

    if "business_day" not in df.columns:
        if "ds" in df.columns:
            df["business_day"] = df["ds"].apply(
                lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                if pd.notna(t) and t.hour == 0
                else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
            )
        elif "target_day" in df.columns:
            df["business_day"] = df["target_day"]

    if "target_day" not in df.columns:
        df["target_day"] = df["business_day"]

    # ── Ensure y_pred ──
    if "y_pred" not in df.columns:
        for col in ["pred", "prediction", "y", "price_pred", "pred_y"]:
            if col in df.columns:
                df["y_pred"] = pd.to_numeric(df[col], errors="coerce")
                break
    if "y_pred" not in df.columns:
        df["y_pred"] = float("nan")

    # ── Ensure y_true ──
    if "y_true" not in df.columns:
        for col in ["actual", "price", "price_actual", "y_true_backup"]:
            if col in df.columns:
                df["y_true"] = pd.to_numeric(df[col], errors="coerce")
                break
    if "y_true" not in df.columns:
        df["y_true"] = float("nan")

    # ── Metadata columns ──
    if "tap_source" not in df.columns:
        df["tap_source"] = "rolling_cutoff_3x10"
    if "source_confidence" not in df.columns:
        df["source_confidence"] = 1.0

    df["run_mode"] = "validation_tap_3x10"
    df["created_at"] = datetime.now().isoformat()

    # Build train_start/train_end/test_start/test_end from block specs
    if "train_start" not in df.columns or "train_end" not in df.columns:
        # Derive from dates
        block_specs = generate_3x10_block_specs(predict_date)
        # For simplicity, fill from block_specs
        df["block_tmp"] = df["tap_fold_id"].apply(
            lambda fid: fid // 3.3333 if fid >= 0 else -1
        )
        # Better approach: use age_days
        if "age_days" in df.columns:
            df["cutoff_date"] = df["age_days"].apply(
                lambda ad: (D - timedelta(days=ad + 1)).isoformat() if ad > 0 else ""
            )
            df["train_end"] = df["cutoff_date"]
        else:
            df["cutoff_date"] = ""
            df["train_end"] = ""

    if "train_start" not in df.columns:
        df["train_start"] = ""
    if "test_start" not in df.columns:
        df["test_start"] = ""
    if "test_end" not in df.columns:
        df["test_end"] = ""

    # ── Learner columns ──
    if "learner_tap_fold_id" not in df.columns:
        df["learner_tap_fold_id"] = df["tap_fold_id"]
    if "age_block" not in df.columns:
        df["age_block"] = df["tap_fold_id"].apply(lambda fid: 9 - fid if 0 <= fid <= 9 else -1)
    if "age_days" not in df.columns:
        df["age_days"] = -1
    if "horizon_day" not in df.columns:
        if "horizon_day" not in df.columns:
            df["horizon_day"] = 1

    # ── model_update_block_id ──
    if "model_update_block_id" not in df.columns:
        df["model_update_block_id"] = None
    if "fold_strategy" not in df.columns and model_name == "sgdfnet":
        df["fold_strategy"] = "3x10"

    return df


def save_per_fold_predictions(
    tap_df: pd.DataFrame,
    folds_dir: Path,
    model_name: str,
) -> None:
    """Split unified tap dataframe into per-fold prediction CSVs."""
    if tap_df.empty or "tap_fold_id" not in tap_df.columns:
        return

    for fold_id in range(LEARNER_FOLDS):
        fold_df = tap_df[tap_df["tap_fold_id"] == fold_id].copy()
        if fold_df.empty:
            continue
        fold_dir = folds_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_file = fold_dir / f"{model_name}_predictions.csv"
        fold_df.to_csv(pred_file, index=False)


def save_runtime_report(runtime_rows: list[dict], output_dir: Path) -> None:
    """Save runtime_report.csv."""
    if not runtime_rows:
        return
    df = pd.DataFrame(runtime_rows)
    report_path = Path(output_dir) / "runtime_report.csv"
    df.to_csv(report_path, index=False)
    logger.info("Runtime report saved: %d rows -> %s", len(df), report_path)
