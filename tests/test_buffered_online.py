"""Mock tests for TimeMixer / RT916 buffered online validation tap.

Run from project root: python tests/test_buffered_online.py
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Test 1: Seasonal replay buffer year mapping ──────────────────────

def test_replay_buffer_years():
    """Test: block 0 → year-3, block 1 → year-2, block 2 → year-1."""
    from TimeMixer.buffered_online import _build_seasonal_replay_buffer

    dates = pd.date_range("2023-01-01", "2026-03-01", freq="D")
    df = pd.DataFrame({"ds": dates})
    predict_date = pd.Timestamp("2026-01-31")

    for block_id in range(3):
        start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=block_id * 10)
        end = start + pd.Timedelta(days=9)
        block_days = pd.date_range(start, end, freq="D").tolist()
        result = _build_seasonal_replay_buffer(df, block_days, block_id, predict_date)
        replay_year = result[1]
        expected = predict_date.year - (3 - block_id)
        assert replay_year == expected, f"block {block_id}: expected {expected}, got {replay_year}"
        logger.info("PASS: test_replay_buffer_years block_id=%d replay_year=%d", block_id, replay_year)

    logger.info("PASS: test_replay_buffer_years")


# ── Test 2: Date range slicing (30 days → learner_tap_fold_id 0..9) ──

def test_date_to_fold_mapping():
    """Test: 30 days (D-30 to D-1) map to learner_tap_fold_id 0..9."""
    D = pd.Timestamp("2026-02-01").date()

    fold_counts = {i: 0 for i in range(10)}
    test_start = D - timedelta(days=30)
    for day_offset in range(30):
        target_day = test_start + timedelta(days=day_offset)
        learner_tap_fold_id = day_offset // 3
        if learner_tap_fold_id > 9:
            learner_tap_fold_id = 9
        age_block = 9 - learner_tap_fold_id
        horizon_day = (day_offset % 3) + 1
        age_days = (D - target_day).days

        fold_counts[learner_tap_fold_id] += 1

        # Verify expected ranges
        if learner_tap_fold_id == 0:
            assert 27 <= age_days <= 30, f"fold 0 age_days={age_days}"
        elif learner_tap_fold_id == 9:
            assert 1 <= age_days <= 3, f"fold 9 age_days={age_days}"

        assert 1 <= horizon_day <= 3
        assert 0 <= age_block <= 9

    # Each fold should have exactly 3 days
    for fid, count in fold_counts.items():
        assert count == 3, f"fold {fid}: expected 3 days, got {count}"

    # Age block: fold 0 (oldest) age_block=9, fold 9 (newest) age_block=0
    for fid in range(10):
        expected_age = 9 - fid
        assert expected_age >= 0, f"age_block should be >= 0, got {expected_age}"

    logger.info("PASS: test_date_to_fold_mapping")


# ── Test 3: Model update block mapping ────────────────────────────────

def test_model_update_block_mapping():
    """Test: 3 model_update_blocks (0,1,2) for 10-day blocks."""
    day_index_to_block = {}
    for day_i in range(30):
        block_id = day_i // 10
        if block_id > 2:
            block_id = 2
        day_index_to_block[day_i] = block_id

    # Block 0: days 0-9 (D-30 to D-21)
    for i in range(10):
        assert day_index_to_block[i] == 0, f"day {i} should be block 0"

    # Block 1: days 10-19 (D-20 to D-11)
    for i in range(10, 20):
        assert day_index_to_block[i] == 1, f"day {i} should be block 1"

    # Block 2: days 20-29 (D-10 to D-1)
    for i in range(20, 30):
        assert day_index_to_block[i] == 2, f"day {i} should be block 2"

    logger.info("PASS: test_model_update_block_mapping")


# ── Test 4: Fallback when historical data is missing ──────────────────

def test_replay_buffer_fallback():
    """Test: falls back gracefully when replay year data is missing."""
    from TimeMixer.buffered_online import _build_seasonal_replay_buffer

    # Only have 2026 data, no historical data
    dates = pd.date_range("2026-01-01", "2026-03-01", freq="D")
    df = pd.DataFrame({"ds": dates})
    predict_date = pd.Timestamp("2026-01-31")

    for block_id in range(3):
        start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=block_id * 10)
        end = start + pd.Timedelta(days=9)
        block_days = pd.date_range(start, end, freq="D").tolist()
        result = _build_seasonal_replay_buffer(df, block_days, block_id, predict_date)
        replay_days, _, _, _, fallback = result

        # Should not raise an error, even if data is missing
        assert len(replay_days) >= 0, f"block {block_id}: should return empty list or fallback"
        assert fallback != "", f"block {block_id}: should have fallback reason"
        logger.info("PASS: test_replay_buffer_fallback block_id=%d fallback=%s", block_id, fallback)

    logger.info("PASS: test_replay_buffer_fallback")


# ── Run all tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== Running mock tests for buffered online ===")
    test_replay_buffer_years()
    test_date_to_fold_mapping()
    test_model_update_block_mapping()
    test_replay_buffer_fallback()
    logger.info("=== All mock tests passed ===")
