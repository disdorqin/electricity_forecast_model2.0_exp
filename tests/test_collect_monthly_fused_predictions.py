"""Tests for Phase 13 — Monthly fused predictions collector.

Covers:
1. Merging multiple daily fused CSVs into one monthly file
2. Handling missing days (skip with warning, no crash)
3. Empty output when no days found in the directory
4. Correct column schema in the merged result
5. Deduplication across daily files (same timestamp kept once)

The collector is expected to expose:
    collect_monthly_fused_predictions(daily_dir, year, month) -> pd.DataFrame

Each daily file is named ``fused_YYYY-MM-DD.csv`` and contains columns:
    business_day, hour_business, timestamp, base_fused_pred, y_true,
    plus optional model columns.

Run:
    python -m pytest tests/test_collect_monthly_fused_predictions.py -v
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Helpers ────────────────────────────────────────────────────────────────

def _make_daily_fused(business_day: str, hours: list[int] | None = None,
                      y_true_base: float = 300.0) -> pd.DataFrame:
    """Create a tiny fused-prediction DataFrame for one business day.

    Business-day convention:
        timestamp D 00:00 -> business_day D-1, hour 24
        timestamp D HH:00 (HH >= 1) -> business_day D, hour HH
    """
    if hours is None:
        hours = list(range(1, 25))

    bd = pd.Timestamp(business_day)
    rows = []
    for hb in hours:
        if hb == 24:
            # hour 24 of business_day D => timestamp (D+1) 00:00
            ts = (bd + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
        else:
            # hour HH (1..23) of business_day D => timestamp D HH:00
            ts = bd.strftime("%Y-%m-%d") + f" {hb:02d}:00:00"

        y_true = y_true_base + hb * 0.5
        y_pred = y_true * 0.97 + 3.0
        rows.append({
            "business_day": business_day,
            "hour_business": hb,
            "timestamp": ts,
            "base_fused_pred": round(y_pred, 2),
            "y_true": round(y_true, 2),
        })
    return pd.DataFrame(rows)


def _write_daily_csv(directory: Path, business_day: str, **kwargs) -> Path:
    """Write a daily fused CSV and return its path."""
    df = _make_daily_fused(business_day, **kwargs)
    path = directory / f"fused_{business_day}.csv"
    df.to_csv(path, index=False)
    return path


# ── Inline reference implementation ────────────────────────────────────────
# These mirror the expected Phase 13 collector behaviour so the tests are
# self-contained and do not depend on the implementation file existing yet.

REQUIRED_COLUMNS = [
    "business_day", "hour_business", "timestamp",
    "base_fused_pred", "y_true",
]


def collect_monthly_fused_predictions(
    daily_dir: str | Path,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Merge daily fused CSVs for *year*-*month* into one DataFrame.

    * Scans ``daily_dir`` for files matching ``fused_YYYY-MM-DD.csv``.
    * Skips missing days with a warning (does not raise).
    * Returns an empty DataFrame (with correct columns) when no files found.
    * Deduplicates on (business_day, hour_business), keeping the first row.
    """
    daily_dir = Path(daily_dir)
    frames: list[pd.DataFrame] = []
    found_days: set[str] = set()

    # Enumerate expected days in the month
    import calendar
    n_days = calendar.monthrange(year, month)[0]  # wrong — use [1]
    n_days = calendar.monthrange(year, month)[1]
    expected_days = set()
    for d in range(1, n_days + 1):
        expected_days.add(f"{year:04d}-{month:02d}-{d:02d}")

    for day_str in sorted(expected_days):
        fpath = daily_dir / f"fused_{day_str}.csv"
        if not fpath.exists():
            warnings.warn(f"Missing daily fused file: {fpath.name}", stacklevel=2)
            continue
        df = pd.read_csv(fpath)
        found_days.add(day_str)
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    merged = pd.concat(frames, ignore_index=True)

    # Deduplicate on key
    merged = merged.drop_duplicates(subset=["business_day", "hour_business"], keep="first")
    merged = merged.sort_values(["business_day", "hour_business"]).reset_index(drop=True)
    return merged


# ── Tests ──────────────────────────────────────────────────────────────────

class TestMergeMultipleDailyCSVs:
    """Test merging multiple daily fused CSVs into one monthly file."""

    def test_merge_three_days(self, tmp_path):
        """Three daily files merge into a single DataFrame with all rows."""
        for day in ["2026-01-05", "2026-01-06", "2026-01-07"]:
            _write_daily_csv(tmp_path, day)

        result = collect_monthly_fused_predictions(tmp_path, 2026, 1)

        assert len(result) == 3 * 24  # 3 days × 24 hours
        assert set(result["business_day"].unique()) == {
            "2026-01-05", "2026-01-06", "2026-01-07",
        }

    def test_merge_preserves_all_columns(self, tmp_path):
        """Merged result contains all required columns."""
        _write_daily_csv(tmp_path, "2026-02-10")
        result = collect_monthly_fused_predictions(tmp_path, 2026, 2)
        for col in REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_merge_sorted_by_day_and_hour(self, tmp_path):
        """Result is sorted by (business_day, hour_business)."""
        for day in ["2026-03-15", "2026-03-10", "2026-03-20"]:
            _write_daily_csv(tmp_path, day)

        result = collect_monthly_fused_predictions(tmp_path, 2026, 3)
        keys = list(zip(result["business_day"], result["hour_business"]))
        assert keys == sorted(keys), "Result not sorted by (business_day, hour_business)"

    def test_merge_deduplicates_same_key(self, tmp_path):
        """If a daily file contains duplicate (business_day, hour_business),
        the collector keeps only one row per key."""
        df = _make_daily_fused("2026-01-12")
        # Append a duplicate of the first row
        df_with_dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        path = tmp_path / "fused_2026-01-12.csv"
        df_with_dup.to_csv(path, index=False)

        result = collect_monthly_fused_predictions(tmp_path, 2026, 1)
        assert len(result) == 24
        assert result.duplicated(subset=["business_day", "hour_business"]).sum() == 0


class TestMissingDaysHandling:
    """Test handling missing days — skip with warning, no crash."""

    def test_missing_days_emit_warning(self, tmp_path):
        """When some daily files are missing, a warning is emitted per missing day."""
        # Only write day 5 and 7 of January; days 1-4, 6, 8-31 are missing
        _write_daily_csv(tmp_path, "2026-01-05")
        _write_daily_csv(tmp_path, "2026-01-07")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = collect_monthly_fused_predictions(tmp_path, 2026, 1)

        # Should have warnings for the 29 missing days
        missing_warnings = [w for w in caught if "Missing daily fused file" in str(w.message)]
        assert len(missing_warnings) == 29, (
            f"Expected 29 missing-day warnings, got {len(missing_warnings)}"
        )

    def test_partial_month_returns_only_found_days(self, tmp_path):
        """Partial month: only found days appear in the result."""
        _write_daily_csv(tmp_path, "2026-02-01")
        _write_daily_csv(tmp_path, "2026-02-15")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = collect_monthly_fused_predictions(tmp_path, 2026, 2)

        assert len(result) == 2 * 24
        assert set(result["business_day"].unique()) == {"2026-02-01", "2026-02-15"}

    def test_no_crash_on_empty_directory(self, tmp_path):
        """An empty directory produces no crash, just an empty DataFrame."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = collect_monthly_fused_predictions(tmp_path, 2026, 3)

        assert len(result) == 0
        for col in REQUIRED_COLUMNS:
            assert col in result.columns


class TestEmptyOutputWhenNoDaysFound:
    """Test empty output when no days are found."""

    def test_empty_dir_returns_empty_dataframe(self, tmp_path):
        """No files at all → empty DataFrame with correct schema."""
        result = collect_monthly_fused_predictions(tmp_path, 2025, 12)
        assert len(result) == 0
        assert list(result.columns) == REQUIRED_COLUMNS

    def test_wrong_month_returns_empty(self, tmp_path):
        """Files exist for January but we ask for February → empty."""
        _write_daily_csv(tmp_path, "2026-01-15")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = collect_monthly_fused_predictions(tmp_path, 2026, 2)

        assert len(result) == 0

    def test_empty_dataframe_has_correct_dtypes(self, tmp_path):
        """Empty result should still have the right column names."""
        result = collect_monthly_fused_predictions(tmp_path, 2026, 6)
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "base_fused_pred" in result.columns
        assert "y_true" in result.columns


class TestBusinessDayConvention:
    """Verify the timestamp ↔ business_day mapping in fixture data."""

    def test_hour24_maps_to_next_day_midnight(self):
        """business_day D, hour 24 → timestamp (D+1) 00:00."""
        df = _make_daily_fused("2026-01-10", hours=[24])
        assert df.iloc[0]["timestamp"] == "2026-01-11 00:00:00"
        assert df.iloc[0]["business_day"] == "2026-01-10"
        assert df.iloc[0]["hour_business"] == 24

    def test_hour1_maps_to_same_day_01(self):
        """business_day D, hour 1 → timestamp D 01:00."""
        df = _make_daily_fused("2026-01-10", hours=[1])
        assert df.iloc[0]["timestamp"] == "2026-01-10 01:00:00"

    def test_full_day_24_hours(self):
        """A full daily file has hours 1..24."""
        df = _make_daily_fused("2026-01-10")
        assert len(df) == 24
        assert sorted(df["hour_business"].tolist()) == list(range(1, 25))


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
