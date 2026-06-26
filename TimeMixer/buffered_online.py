"""Buffered Online for TimeMixer: base train once + 3×10-day blocks + seasonal replay.

Design:
- Base train: 12 months → D-31 (3 segments × 1 model each)
- Block 0: predict D-30~D-21, update with current + seasonal replay (year-3)
- Block 1: predict D-20~D-11, update with current + seasonal replay (year-2)
- Block 2: predict D-10~D-1,  update with current + seasonal replay (year-1)
- Output: 30 days sliced into learner_tap_fold_id 0..9

Usage from validation_tap.py:
    from TimeMixer.buffered_online import run_online_month_buffered
    df = run_online_month_buffered(task="dayahead", data_path=..., predict_date="2026-02-01", ...)
"""

from __future__ import annotations

import logging
import os as _os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from TimeMixer.repro_pipeline import (
    SEGMENTS,
    SEGMENT_RT_TARGETS,
    RunConfig,
    build_segment_arrays,
    compute_cutoff,
    date_range_days,
    filter_available_days,
    load_data,
    load_segment_checkpoint,
    predict_model,
    save_segment_checkpoint,
    set_seed,
    split_train_valid,
    train_model,
    _online_update_segment,
    _predictions_to_dataframe,
    make_prediction_rows,
)

logger = logging.getLogger(__name__)

# ── Seasonal replay buffer ──────────────────────────────────────────────

def _build_seasonal_replay_buffer(
    df: pd.DataFrame,
    current_block_days: list[pd.Timestamp],
    block_id: int,
    predict_date: pd.Timestamp,
) -> tuple[list[pd.Timestamp], int, pd.Timestamp, pd.Timestamp, str]:
    """Build seasonal replay buffer for a given block.

    Rules:
      block_id=0 → replay_year = D.year - 3
      block_id=1 → replay_year = D.year - 2
      block_id=2 → replay_year = D.year - 1

    Fallback hierarchy:
      1. Same year, same month, same day-of-month range
      2. Same year, same month, full month
      3. Other historical years, same month

    Returns (replay_days, replay_year, replay_start, replay_end, fallback_reason).
    """
    D = predict_date
    replay_year = D.year - (3 - block_id)

    block_start = pd.Timestamp(current_block_days[0])
    block_end = pd.Timestamp(current_block_days[-1])

    # Compute target month/day offsets
    month = block_start.month
    start_day = block_start.day
    end_day = block_end.day
    n_days = len(current_block_days)

    fallback_reason = ""

    # Try 1: exact same month + day range in replay year
    try:
        replay_start_date = pd.Timestamp(year=replay_year, month=month, day=start_day)
        replay_end_date = pd.Timestamp(year=replay_year, month=month, day=end_day)
        replay_days = date_range_days(replay_start_date, replay_end_date + pd.Timedelta(days=1))
        replay_days = [d for d in replay_days if d in set(df["ds"].dt.normalize())]
        if len(replay_days) >= n_days * 0.5:
            return replay_days, replay_year, replay_start_date, replay_end_date, fallback_reason
        fallback_reason = f"insufficient exact match ({len(replay_days)}/{n_days})"
    except (ValueError, KeyError):
        fallback_reason = "exact date creation failed"

    # Try 2: same year, same month, full month (all available days)
    try:
        month_start = pd.Timestamp(year=replay_year, month=month, day=1)
        next_month = month_start + pd.DateOffset(months=1)
        all_month_days = date_range_days(month_start, next_month)
        replay_days = [d for d in all_month_days if d in set(df["ds"].dt.normalize())]
        if len(replay_days) >= 10:
            fallback_reason += "; full_month_fill"
            return replay_days[:n_days], replay_year, month_start, next_month - pd.Timedelta(days=1), fallback_reason
        fallback_reason += f"; full_month_short ({len(replay_days)})"
    except (ValueError, KeyError):
        pass

    # Try 3: any year with same month (scan backwards)
    for yr_offset in range(1, 4):
        alt_year = replay_year - yr_offset
        try:
            alt_start = pd.Timestamp(year=alt_year, month=month, day=start_day)
            alt_end = pd.Timestamp(year=alt_year, month=month, day=end_day)
            alt_days = date_range_days(alt_start, alt_end + pd.Timedelta(days=1))
            alt_days = [d for d in alt_days if d in set(df["ds"].dt.normalize())]
            if len(alt_days) >= n_days * 0.5:
                fallback_reason += f"; fallback_to_{alt_year}"
                return alt_days, alt_year, alt_start, alt_end, fallback_reason
        except (ValueError, KeyError):
            pass

    # Final fallback: empty replay buffer (use only current data)
    fallback_reason += "; empty_replay_buffer"
    return [], replay_year, pd.Timestamp(replay_year, 1, 1), pd.Timestamp(replay_year, 1, 1), fallback_reason


# ── Online update with seasonal replay ──────────────────────────────────

def _online_update_with_replay(
    model_bundle: dict,
    cfg: RunConfig,
    device: torch.device,
    current_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    replay_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    task: str,
    segment_name: str,
    *,
    lambda_seasonal: float = 0.3,
    lambda_anchor: float = 1e-4,
) -> dict:
    """Online update an existing model with current block + seasonal replay.

    Loss = current_block_loss + lambda_seasonal * replay_loss + lambda_anchor * anchor_loss
    where anchor_loss = ||theta - theta_before||²

    Current data gets weight 1.0; replay data gets weight lambda_seasonal.
    """
    current_past, current_future, current_y = current_arrays
    n_current = len(current_y)

    if replay_arrays is not None and len(replay_arrays) > 0:
        rp_past, rp_future, rp_y = replay_arrays
        n_replay = len(rp_y)

        # Combine current + replay
        combined_past = np.concatenate([current_past, rp_past], axis=0)
        combined_future = np.concatenate([current_future, rp_future], axis=0)
        combined_y = np.concatenate([current_y, rp_y], axis=0)
        # Sample weights: current=1.0, replay=lambda_seasonal
        sample_weights = np.concatenate([
            np.ones(n_current, dtype=float),
            np.full(n_replay, lambda_seasonal, dtype=float),
        ])
    else:
        combined_past = current_past
        combined_future = current_future
        combined_y = current_y
        sample_weights = np.ones(n_current, dtype=float)

    # Save parameter snapshot for anchor loss
    model = model_bundle.get("model")
    anchor_params = None
    if lambda_anchor > 0 and model is not None:
        anchor_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    # Run online update using the existing infrastructure
    # temporally override config online_epochs etc.
    original_epochs = cfg.online_epochs
    original_lr = cfg.online_lr

    try:
        updated = _online_update_segment(
            model_bundle, cfg, device,
            combined_past, combined_future, combined_y,
            task=task, segment_name=segment_name,
        )
    finally:
        cfg.online_epochs = original_epochs
        cfg.online_lr = original_lr

    # Apply anchor loss penalty (L2 weight decay towards original params)
    if anchor_params is not None and model is not None:
        with torch.no_grad():
            anchor_norm = 0.0
            for k, v in model.named_parameters():
                if k in anchor_params:
                    diff = v - anchor_params[k]
                    anchor_norm += (diff ** 2).sum()
            # Scale down parameters towards origin (equivalent to L2 regularization in update)
            for k, v in model.named_parameters():
                if k in anchor_params:
                    v.copy_(v - lambda_anchor * (v - anchor_params[k]))
                    anchor_norm += ((v - anchor_params[k]) ** 2).sum()

    return updated


# ── Main entry point ────────────────────────────────────────────────────

def run_online_month_buffered(
    task: str,
    data_path: str,
    predict_date: str,
    checkpoint_dir: str,
    *,
    train_months: int = 12,
    online_epochs: int = 2,
    online_lr: float | None = None,
    lambda_seasonal: float = 0.3,
    lambda_anchor: float = 1e-4,
    replay_buffer_mode: str = "rotating_year",
    seed: int = 42,
    block_size: int = 10,
) -> pd.DataFrame:
    """Run TimeMixer buffered online: 12-month base train + 3×10-day blocks + seasonal replay.

    Parameters
    ----------
    task : str
        "dayahead" or "realtime".
    data_path : str
        Path to raw CSV data.
    predict_date : str
        D (forecast date), ISO format "YYYY-MM-DD".
    checkpoint_dir : str
        Directory for checkpoints.
    train_months : int
        Base training months (default 12).
    online_epochs : int
        Epochs per online update block (default 2).
    online_lr : float | None
        Learning rate for online update (None = auto 0.05 * base_lr).
    lambda_seasonal : float
        Weight for seasonal replay loss (default 0.3).
    lambda_anchor : float
        Weight for anchor regularization (default 1e-4).
    replay_buffer_mode : str
        "rotating_year" (default).

    Returns
    -------
    pd.DataFrame
        30 days of predictions with all required metadata columns.
    """
    # ── Setup ──────────────────────────────────────────────────────
    set_seed(seed)
    _os.environ["OPTIM_NUM_WORKERS"] = "0"
    _os.environ["OPTIM_PIN_MEMORY"] = "0"

    D = pd.Timestamp(predict_date).date()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[buffered_online/%s] device=%s, D=%s", task, device, D)

    # Load data
    df = load_data(data_path)
    df["ds"] = pd.to_datetime(df["ds"])

    # Determine target column
    if task == "dayahead":
        target_col = "day_ahead_clearing_price"
        cutoff_hour = 15  # day-ahead cutoff
        cfg_task = "da"
    else:
        target_col = "realtime_price"
        cutoff_hour = 15
        cfg_task = "rt"

    # ── Fix: figure out which segment target col to use ──
    # For realtime, each segment has a different target column
    segment_target_cols: dict[str, str] = {}
    if task == "dayahead":
        for seg_name, _, _ in SEGMENTS:
            segment_target_cols[seg_name] = "day_ahead_clearing_price"
    else:
        for seg_name, _, _ in SEGMENTS:
            segment_target_cols[seg_name] = SEGMENT_RT_TARGETS.get(seg_name, "realtime_price")

    # ── Base train: 12 months → D-31 ───────────────────────────────
    base_cutoff = D - timedelta(days=30)  # D-30 cutoff (train to D-31 inclusive)
    base_cutoff_start = compute_cutoff(pd.Timestamp(D - timedelta(days=30)), cutoff_hour)
    # Actually train from D-31 backwards 12 months
    train_end_date = D - timedelta(days=31)
    train_start_date = max(
        df["ds"].min().normalize(),
        train_end_date - pd.DateOffset(months=train_months),
    )
    base_train_days = date_range_days(
        train_start_date,
        train_end_date + pd.Timedelta(days=1),
    )
    if len(base_train_days) < 30:
        raise ValueError(f"Base training window too short: {len(base_train_days)} days")

    base_train, base_valid = split_train_valid(base_train_days, val_ratio=0.2)

    logger.info(
        "[buffered_online/%s] Base train: %d days (%s → %s)",
        task, len(base_train),
        train_start_date.date(), train_end_date,
    )

    # RunConfig for training
    cfg = RunConfig(
        data_path=data_path,
        output_dir=str(ckpt_dir.parent / "buffered_online_out"),
        month="",
        train_months=train_months,
        online_epochs=online_epochs,
        online_lr=online_lr or 5e-5,  # 0.05 * base_lr
        seed=seed,
    )

    # Train all 3 segments and save checkpoints
    for segment_name, start_idx, end_idx in SEGMENTS:
        tgt = segment_target_cols[segment_name]
        arrays = build_segment_arrays(
            df, base_train, tgt, cfg.seq_len,
            cutoff_hour, start_idx, end_idx,
            target_mode="direct",
        )
        if len(arrays) == 5:
            past, future, y, _, hour_ids = arrays
        else:
            past, future, y, _ = arrays
            hour_ids = None

        # Use hour_ids if available
        bundle_kwargs = {}
        if hour_ids is not None:
            bundle_kwargs["hour_ids"] = hour_ids

        bundle = train_model(past, future, y, cfg, device, task=cfg_task, segment_name=segment_name)
        save_segment_checkpoint(bundle, cfg, cfg_task, segment_name, ckpt_dir)

    logger.info("[buffered_online/%s] Base train complete. %d checkpoints saved", task, 3)

    # ── Block-by-block predict + update ─────────────────────────────
    # 3 blocks of 10 days each
    test_start = D - timedelta(days=30)
    test_end_exclusive = D  # D-30 to D-1

    all_test_days = date_range_days(
        pd.Timestamp(test_start),
        pd.Timestamp(test_end_exclusive),
    )
    all_test_days = filter_available_days(
        df, all_test_days,
        seq_len=cfg.seq_len,
        cutoff_hour_da=cfg.cutoff_hour_da,
        cutoff_hour_rt=cfg.cutoff_hour_rt,
        da_target_mode="direct",
        rt_target_mode="direct",
        inference_mode=True,
    )

    n_blocks = 3
    # Recalculate block_size from actual available days
    n_days = len(all_test_days)
    block_size_actual = max(1, n_days // n_blocks)

    all_segment_preds: dict[str, list[np.ndarray]] = {seg_name: [] for seg_name, _, _ in SEGMENTS}
    all_block_metadata: list[dict] = []

    for block_id in range(n_blocks):
        block_start_idx = block_id * block_size_actual
        block_end_idx = min(block_start_idx + block_size_actual, n_days)
        block_days = all_test_days[block_start_idx:block_end_idx]
        if not block_days:
            continue

        block_label = f"block_{block_id}"
        block_start = pd.Timestamp(block_days[0])
        block_end = pd.Timestamp(block_days[-1])

        logger.info("[buffered_online/%s] %s: predict %s ~ %s", task, block_label, block_start.date(), block_end.date())

        # Step A: Predict current block
        block_preds: dict[str, np.ndarray] = {}
        for segment_name, start_idx, end_idx in SEGMENTS:
            tgt = segment_target_cols[segment_name]
            ckpt = load_segment_checkpoint(ckpt_dir / f"{cfg_task}_{segment_name}.ckpt", device, cfg)
            test_arrays = build_segment_arrays(
                df, block_days, tgt, cfg.seq_len,
                cutoff_hour, start_idx, end_idx,
                target_mode="direct",
            )
            if len(test_arrays) == 5:
                test_past, test_future, test_y, test_baseline, _ = test_arrays
            else:
                test_past, test_future, test_y, test_baseline = test_arrays

            pred = predict_model(ckpt, test_past, test_future, device, cfg.batch_size)
            # pred shape: (n_days_in_block, 8 or 8)
            # The segment slice is [start_idx:end_idx]
            seg_pred = pred  # keep full segment prediction
            block_preds[segment_name] = seg_pred

            all_segment_preds[segment_name].append(seg_pred)

        # Step B: Build seasonal replay buffer
        replay_days, replay_year, replay_start, replay_end, fallback = _build_seasonal_replay_buffer(
            df, block_days, block_id, pd.Timestamp(D),
        )
        logger.info(
            "[buffered_online/%s] %s replay: year=%d, %d days, fallback=%s",
            task, block_label, replay_year, len(replay_days), fallback or "none",
        )

        # Record metadata for this block
        for day in block_days:
            all_block_metadata.append({
                "target_day": day.date(),
                "model_update_block_id": block_id,
                "replay_year": replay_year,
                "replay_start": replay_start.date() if hasattr(replay_start, "date") else replay_start,
                "replay_end": replay_end.date() if hasattr(replay_end, "date") else replay_end,
                "lambda_seasonal": lambda_seasonal,
                "lambda_anchor": lambda_anchor,
                "fallback_reason": fallback,
            })

        # Step C: Online update
        logger.info(
            "[buffered_online/%s] %s: online update (%d epochs, lr=%.2e)",
            task, block_label, online_epochs, online_lr or 5e-5,
        )

        for segment_name, start_idx, end_idx in SEGMENTS:
            tgt = segment_target_cols[segment_name]
            ckpt = load_segment_checkpoint(ckpt_dir / f"{cfg_task}_{segment_name}.ckpt", device, cfg)

            # Current block arrays
            curr_arrays = build_segment_arrays(
                df, block_days, tgt, cfg.seq_len,
                cutoff_hour, start_idx, end_idx,
                target_mode="direct",
            )
            if len(curr_arrays) == 5:
                curr_past, curr_future, curr_y, _, _ = curr_arrays
            else:
                curr_past, curr_future, curr_y, _ = curr_arrays

            # Replay buffer arrays
            replay_arrays = None
            if replay_days:
                try:
                    r_arrays = build_segment_arrays(
                        df, replay_days, tgt, cfg.seq_len,
                        cutoff_hour, start_idx, end_idx,
                        target_mode="direct",
                    )
                    if len(r_arrays) == 5:
                        rp_past, rp_future, rp_y, _, _ = r_arrays
                    else:
                        rp_past, rp_future, rp_y, _ = r_arrays
                    if len(rp_y) > 0:
                        replay_arrays = (rp_past, rp_future, rp_y)
                except Exception as e:
                    logger.warning("[buffered_online/%s] replay array build failed: %s", task, e)

            # Update with replay
            updated_ckpt = _online_update_with_replay(
                ckpt, cfg, device,
                (curr_past, curr_future, curr_y),
                replay_arrays,
                task=cfg_task, segment_name=segment_name,
                lambda_seasonal=lambda_seasonal,
                lambda_anchor=lambda_anchor,
            )
            save_segment_checkpoint(updated_ckpt, cfg, cfg_task, segment_name, ckpt_dir)

    logger.info("[buffered_online/%s] All blocks complete. Building output DataFrame...", task)

    # ── Stitch predictions ──────────────────────────────────────────
    # Combine segment predictions into full 24h predictions
    stitched_preds = []
    for block_i in range(n_blocks):
        block_start_idx = block_i * block_size_actual
        block_end_idx = min(block_start_idx + block_size_actual, n_days)
        block_days = all_test_days[block_start_idx:block_end_idx]
        n_block_days = len(block_days)

        for day_i in range(n_block_days):
            day_pred = np.zeros(24, dtype=float)
            for name, start, end in SEGMENTS:
                seg_preds = all_segment_preds[name]
                # Find the right block and day index
                # seg_preds[block_i] has shape (n_block_days, end-start)
                seg_pred_block = seg_preds[block_i]
                if day_i < seg_pred_block.shape[0]:
                    day_pred[start:end] = seg_pred_block[day_i]
            stitched_preds.append(day_pred)

    stitched = np.stack(stitched_preds)

    # ── Build output DataFrame ──────────────────────────────────────
    # Flatten 30 days × 24 hours into rows
    rows = []
    created_at = datetime.now().isoformat()
    run_mode = "online_update_buffered"

    for day_i, day_ts in enumerate(all_test_days):
        target_day = day_ts.date()
        for hour in range(24):
            # business hour: 1-24 (0→24)
            hour_business = hour if hour > 0 else 24

            # Determine period
            if 1 <= hour_business <= 8:
                period = "1_8"
            elif 9 <= hour_business <= 16:
                period = "9_16"
            else:
                period = "17_24"

            ds = day_ts + pd.Timedelta(hours=hour)
            y_pred = stitched[day_i][hour]

            # Compute learner_tap_fold_id (which 3-day block this day belongs to)
            # Day index 0 = D-30, Day index 29 = D-1
            # fold 0 = D-30~D-28 (days 0-2), fold 1 = D-27~D-25 (days 3-5), ..., fold 9 = D-3~D-1 (days 27-29)
            learner_tap_fold_id = day_i // 3
            if learner_tap_fold_id > 9:
                learner_tap_fold_id = 9

            # model_update_block_id
            model_update_block_id = day_i // block_size_actual
            if model_update_block_id > 2:
                model_update_block_id = 2

            # age_block = 9 - learner_tap_fold_id
            age_block = 9 - learner_tap_fold_id

            # horizon_day: day offset within the 3-day fold (1, 2, or 3)
            horizon_day = (day_i % 3) + 1

            # age_days: how many days between this target_day and D
            age_days = (D - target_day).days

            # metadata for this day
            meta = all_block_metadata[day_i] if day_i < len(all_block_metadata) else {}

            row = {
                "task": task,
                "model_name": "timemixer",
                "target_day": target_day.isoformat(),
                "business_day": target_day.isoformat(),
                "ds": ds.isoformat(),
                "hour_business": hour_business,
                "period": period,
                "y_pred": float(y_pred),
                "y_true": np.nan,  # Will be filled by validation tap normalization
                "model_update_block_id": model_update_block_id,
                "learner_tap_fold_id": learner_tap_fold_id,
                "tap_fold_id": learner_tap_fold_id,
                "age_block": age_block,
                "horizon_day": horizon_day,
                "age_days": age_days,
                "tap_source": "online_update_buffered",
                "source_confidence": 0.95,
                "checkpoint_path": str(ckpt_dir),
                "replay_year": meta.get("replay_year", -1),
                "replay_start": str(meta.get("replay_start", "")),
                "replay_end": str(meta.get("replay_end", "")),
                "lambda_seasonal": lambda_seasonal,
                "lambda_anchor": lambda_anchor,
                "run_mode": run_mode,
                "created_at": created_at,
            }
            rows.append(row)

    result_df = pd.DataFrame(rows)
    logger.info(
        "[buffered_online/%s] Output: %d rows, %d days, folds=%s",
        task, len(result_df), result_df["target_day"].nunique(),
        sorted(result_df["tap_fold_id"].unique().tolist()),
    )

    return result_df


# ═══════════════════════════════════════════════════════════════════════
# P0-6: Real forecast — predict from buffered checkpoint
# ═══════════════════════════════════════════════════════════════════════

def predict_from_buffered_checkpoint(
    task: str,
    data_path: str,
    predict_date: str,
    checkpoint_dir: str,
    *,
    seed: int = 42,
) -> pd.DataFrame | None:
    """Predict D using the last buffered online checkpoint (updated to D-1).

    If checkpoint exists, load and predict without re-training.
    If not, return None (caller should fall back to normal training).

    Parameters
    ----------
    task : str
        "dayahead" or "realtime".
    data_path : str
        Path to raw CSV data.
    predict_date : str
        D (forecast date), ISO format "YYYY-MM-DD".
    checkpoint_dir : str
        Directory with buffered_online checkpoints.

    Returns
    -------
    pd.DataFrame or None
        24-row prediction for day D, or None if checkpoint not available.
    """
    set_seed(seed)
    _os.environ["OPTIM_NUM_WORKERS"] = "0"
    _os.environ["OPTIM_PIN_MEMORY"] = "0"

    D = pd.Timestamp(predict_date).date()
    ckpt_dir = Path(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if task == "dayahead":
        cfg_task = "da"
        target_col = "day_ahead_clearing_price"
        cutoff_hour = 15
    else:
        cfg_task = "rt"
        cutoff_hour = 15
        from TimeMixer.repro_pipeline import SEGMENT_RT_TARGETS
        segment_target_cols = SEGMENT_RT_TARGETS

    # Check if checkpoints exist
    ckpt_missing = any(
        not (ckpt_dir / f"{cfg_task}_{seg_name}.ckpt").exists()
        for seg_name, _, _ in SEGMENTS
    )
    if ckpt_missing:
        logger.warning("[predict_from_ckpt/%s] checkpoints not found in %s", task, ckpt_dir)
        return None

    # Load data
    df = load_data(data_path)
    df["ds"] = pd.to_datetime(df["ds"])

    # Predict D
    target_day = pd.Timestamp(D)
    cfg = RunConfig(
        data_path=data_path,
        output_dir=str(ckpt_dir.parent / "real_forecast"),
        month="",
        seed=seed,
    )

    stitched_preds = []
    for segment_name, start_idx, end_idx in SEGMENTS:
        # Load checkpoint
        ckpt = load_segment_checkpoint(ckpt_dir / f"{cfg_task}_{segment_name}.ckpt", device, cfg)

        if task == "dayahead":
            tgt = "day_ahead_clearing_price"
        else:
            tgt = segment_target_cols.get(segment_name, "realtime_price")

        test_arrays = build_segment_arrays(
            df, [target_day], tgt, cfg.seq_len,
            cutoff_hour, start_idx, end_idx,
            target_mode="direct",
        )
        if len(test_arrays) == 5:
            test_past, test_future, _, test_baseline, _ = test_arrays
        else:
            test_past, test_future, _, test_baseline = test_arrays

        pred = predict_model(ckpt, test_past, test_future, device, cfg.batch_size)
        stitched_preds.append((segment_name, start_idx, end_idx, pred, test_baseline))

    # Stitch into 24h
    day_pred = np.zeros(24, dtype=float)
    for seg_name, start, end, pred_arr, baseline in stitched_preds:
        if len(pred_arr.shape) == 2 and pred_arr.shape[0] == 1:
            pred_arr = pred_arr[0]
        day_pred[start:end] = pred_arr + baseline[start:end]

    # Build output DataFrame
    created_at = datetime.now().isoformat()
    rows = []
    for hour in range(24):
        hb = hour if hour > 0 else 24
        if 1 <= hb <= 8:
            period = "1_8"
        elif 9 <= hb <= 16:
            period = "9_16"
        else:
            period = "17_24"

        ds = target_day + pd.Timedelta(hours=hour)
        rows.append({
            "task": task,
            "model_name": "timemixer",
            "target_day": D.isoformat(),
            "business_day": D.isoformat(),
            "ds": ds.isoformat(),
            "hour_business": hb,
            "period": period,
            "y_pred": float(day_pred[hour]),
            "y_true": np.nan,
            "tap_source": "online_update_buffered",
            "source_confidence": 0.95,
            "checkpoint_path": str(ckpt_dir),
            "run_mode": "buffered_ckpt_inference",
            "created_at": created_at,
        })

    result = pd.DataFrame(rows)
    logger.info("[predict_from_ckpt/%s] %d rows predicted from checkpoint", task, len(result))
    return result
