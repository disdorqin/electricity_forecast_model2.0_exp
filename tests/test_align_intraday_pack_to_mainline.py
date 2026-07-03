"""Tests for Phase 13 — Align intraday pack to mainline.

Covers:
1. Matching on (business_day, hour_business) keys
2. Deduplication by highest cutoff_hour (keep latest correction)
3. Reporting matched / missing / duplicate counts
4. Empty pack produces NO_ALIGNED_REAL_REPLAY sentinel

The aligner is expected to expose:
    align_intraday_pack_to_mainline(
        intraday_pack: pd.DataFrame,
        mainline: pd.DataFrame,
    ) -> AlignResult

where AlignResult is a dataclass / NamedDict with:
    aligned: pd.DataFrame       — the matched rows
    n_matched: int
    n_missing: int              — mainline keys not in pack
    n_duplicate: int            — pack keys dropped by dedup
    status: str                 — "OK" | "NO_ALIGNED_REAL_REPLAY"

Business-day convention:
    timestamp D 00:00 → business_day D-1, hour 24
    timestamp D HH:00 (HH >= 1) → business_day D, hour HH

Run:
    python -m pytest tests/test_align_intraday_pack_to_mainline.py -v
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Inline reference implementation ────────────────────────────────────────

@dataclass
class AlignResult:
    """Result of aligning an intraday pack to the mainline."""
    aligned: pd.DataFrame
    n_matched: int = 0
    n_missing: int = 0
    n_duplicate: int = 0
    status: str = "OK"


def align_intraday_pack_to_mainline(
    intraday_pack: pd.DataFrame,
    mainline: pd.DataFrame,
) -> AlignResult:
    """Align intraday correction pack to mainline forecast table.

    Steps:
        1. If *intraday_pack* is empty → return NO_ALIGNED_REAL_REPLAY.
        2. Deduplicate pack on (business_day, hour_business) keeping the
           row with the **highest** cutoff_hour (latest correction wins).
        3. Inner-join pack to mainline on (business_day, hour_business).
        4. Report matched / missing / duplicate counts.
    """
    key_cols = ["business_day", "hour_business"]

    # ── 1. Empty pack → sentinel ──────────────────────────────────────
    if intraday_pack is None or len(intraday_pack) == 0:
        return AlignResult(
            aligned=pd.DataFrame(),
            n_matched=0,
            n_missing=len(mainline),
            n_duplicate=0,
            status="NO_ALIGNED_REAL_REPLAY",
        )

    # ── 2. Dedup by highest cutoff_hour ───────────────────────────────
    n_before_dedup = len(intraday_pack)
    if "cutoff_hour" in intraday_pack.columns:
        pack_deduped = (
            intraday_pack
            .sort_values("cutoff_hour", ascending=False)
            .drop_duplicates(subset=key_cols, keep="first")
        )
    else:
        pack_deduped = intraday_pack.drop_duplicates(subset=key_cols, keep="first")
    n_duplicate = n_before_dedup - len(pack_deduped)

    # ── 3. Inner join on key ──────────────────────────────────────────
    mainline_keys = set(
        zip(mainline["business_day"], mainline["hour_business"])
    )
    pack_keys = set(
        zip(pack_deduped["business_day"], pack_deduped["hour_business"])
    )

    matched_keys = mainline_keys & pack_keys
    missing_keys = mainline_keys - pack_keys
    n_matched = len(matched_keys)
    n_missing = len(missing_keys)

    if n_matched == 0:
        return AlignResult(
            aligned=pd.DataFrame(),
            n_matched=0,
            n_missing=n_missing,
            n_duplicate=n_duplicate,
            status="NO_ALIGNED_REAL_REPLAY",
        )

    # Build aligned frame: merge pack corrections onto mainline
    aligned = mainline.merge(
        pack_deduped,
        on=key_cols,
        how="inner",
        suffixes=("", "_pack"),
    )

    return AlignResult(
        aligned=aligned,
        n_matched=n_matched,
        n_missing=n_missing,
        n_duplicate=n_duplicate,
        status="OK",
    )


# ── Fixture helpers ────────────────────────────────────────────────────────

def _make_mainline(business_days: list[str],
                   hours: list[int] | None = None) -> pd.DataFrame:
    """Create a minimal mainline forecast table."""
    if hours is None:
        hours = list(range(1, 25))
    rows = []
    for bd in business_days:
        for hb in hours:
            ts = pd.Timestamp(bd)
            if hb == 24:
                ts_str = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
            else:
                ts_str = ts.strftime("%Y-%m-%d") + f" {hb:02d}:00:00"
            rows.append({
                "business_day": bd,
                "hour_business": hb,
                "timestamp": ts_str,
                "base_fused_pred": 300.0 + hb * 0.5,
                "y_true": 310.0 + hb * 0.3,
            })
    return pd.DataFrame(rows)


def _make_intraday_pack(business_day: str,
                        target_hours: list[int],
                        cutoff_hour: int = 12,
                        corrected_offset: float = -5.0) -> pd.DataFrame:
    """Create a minimal intraday correction pack for one day."""
    rows = []
    for hb in target_hours:
        ts = pd.Timestamp(business_day)
        if hb == 24:
            ts_str = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
        else:
            ts_str = ts.strftime("%Y-%m-%d") + f" {hb:02d}:00:00"
        rows.append({
            "business_day": business_day,
            "hour_business": hb,
            "cutoff_hour": cutoff_hour,
            "timestamp": ts_str,
            "intraday_corrected_pred": 300.0 + hb * 0.5 + corrected_offset,
            "policy_decision": "LOW_WEIGHT",
            "fusion_weight": 0.12,
        })
    return pd.DataFrame(rows)


# ── Tests ──────────────────────────────────────────────────────────────────

class TestMatchOnBusinessDayHourBusiness:
    """Test matching on (business_day, hour_business)."""

    def test_exact_match_single_day(self):
        """Pack and mainline with same (day, hours) → full match."""
        mainline = _make_mainline(["2026-01-10"], hours=[9, 10, 11, 12])
        pack = _make_intraday_pack("2026-01-10", target_hours=[9, 10, 11, 12])

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.status == "OK"
        assert result.n_matched == 4
        assert result.n_missing == 0
        assert len(result.aligned) == 4

    def test_partial_match(self):
        """Pack covers only some mainline hours → partial match + missing."""
        mainline = _make_mainline(["2026-01-10"], hours=list(range(1, 25)))
        pack = _make_intraday_pack("2026-01-10", target_hours=[13, 14, 15])

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.status == "OK"
        assert result.n_matched == 3
        assert result.n_missing == 21  # 24 - 3

    def test_multi_day_match(self):
        """Pack covers two days; mainline covers three → correct counts."""
        mainline = _make_mainline(
            ["2026-02-01", "2026-02-02", "2026-02-03"],
            hours=[9, 10, 11],
        )
        pack_d1 = _make_intraday_pack("2026-02-01", target_hours=[9, 10, 11])
        pack_d2 = _make_intraday_pack("2026-02-02", target_hours=[9, 10])
        pack = pd.concat([pack_d1, pack_d2], ignore_index=True)

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.n_matched == 5   # 3 + 2
        assert result.n_missing == 4   # 0 + 1 + 3

    def test_no_overlap_produces_sentinel(self):
        """Pack and mainline have different days → NO_ALIGNED_REAL_REPLAY."""
        mainline = _make_mainline(["2026-01-10"], hours=[9, 10])
        pack = _make_intraday_pack("2026-01-11", target_hours=[9, 10])

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.status == "NO_ALIGNED_REAL_REPLAY"
        assert result.n_matched == 0
        assert result.n_missing == 2


class TestDeduplicationByHighestCutoffHour:
    """Test deduplication by highest cutoff_hour."""

    def test_keep_highest_cutoff_hour(self):
        """When same (day, hour) appears with cutoff 10 and 14, keep cutoff 14."""
        mainline = _make_mainline(["2026-01-10"], hours=[13])
        pack_early = _make_intraday_pack("2026-01-10", [13], cutoff_hour=10,
                                         corrected_offset=-2.0)
        pack_late = _make_intraday_pack("2026-01-10", [13], cutoff_hour=14,
                                         corrected_offset=-8.0)
        pack = pd.concat([pack_early, pack_late], ignore_index=True)

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.n_matched == 1
        assert result.n_duplicate == 1
        # The kept row should be the one with cutoff_hour=14
        aligned_row = result.aligned.iloc[0]
        if "cutoff_hour" in result.aligned.columns:
            assert aligned_row["cutoff_hour"] == 14
        # Corrected pred should come from the cutoff=14 row
        if "intraday_corrected_pred" in result.aligned.columns:
            expected_pred = 300.0 + 13 * 0.5 - 8.0
            assert abs(aligned_row["intraday_corrected_pred"] - expected_pred) < 0.01

    def test_three_cutoffs_keep_latest(self):
        """Three cutoffs for same key → keep highest, report 2 duplicates."""
        mainline = _make_mainline(["2026-03-05"], hours=[15])
        packs = []
        for cutoff in [8, 12, 16]:
            packs.append(
                _make_intraday_pack("2026-03-05", [15], cutoff_hour=cutoff,
                                    corrected_offset=-1.0 * cutoff)
            )
        pack = pd.concat(packs, ignore_index=True)

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.n_matched == 1
        assert result.n_duplicate == 2
        if "cutoff_hour" in result.aligned.columns:
            assert result.aligned.iloc[0]["cutoff_hour"] == 16

    def test_no_duplicates_when_unique(self):
        """Each (day, hour) appears once → n_duplicate = 0."""
        mainline = _make_mainline(["2026-01-10"], hours=[9, 10, 11])
        pack = _make_intraday_pack("2026-01-10", target_hours=[9, 10, 11])

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.n_duplicate == 0


class TestReportingCounts:
    """Test reporting matched / missing / duplicate counts."""

    def test_counts_are_consistent(self):
        """n_matched + n_missing == total mainline unique keys."""
        mainline = _make_mainline(["2026-01-10"], hours=list(range(1, 25)))
        pack = _make_intraday_pack("2026-01-10", target_hours=[1, 2, 3, 4, 5])

        result = align_intraday_pack_to_mainline(pack, mainline)

        total_mainline_keys = len(
            mainline.drop_duplicates(subset=["business_day", "hour_business"])
        )
        assert result.n_matched + result.n_missing == total_mainline_keys

    def test_aligned_row_count_equals_matched(self):
        """len(aligned) == n_matched."""
        mainline = _make_mainline(["2026-01-10", "2026-01-11"], hours=[9, 10, 11])
        pack_d1 = _make_intraday_pack("2026-01-10", target_hours=[9, 10])
        pack_d2 = _make_intraday_pack("2026-01-11", target_hours=[11])
        pack = pd.concat([pack_d1, pack_d2], ignore_index=True)

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert len(result.aligned) == result.n_matched

    def test_counts_with_dedup(self):
        """Duplicate pack rows are counted but do not inflate matched."""
        mainline = _make_mainline(["2026-01-10"], hours=[9, 10])
        # Two rows for hour 9 (cutoff 10 and 14), one for hour 10
        pack_a = _make_intraday_pack("2026-01-10", [9], cutoff_hour=10)
        pack_b = _make_intraday_pack("2026-01-10", [9], cutoff_hour=14)
        pack_c = _make_intraday_pack("2026-01-10", [10], cutoff_hour=12)
        pack = pd.concat([pack_a, pack_b, pack_c], ignore_index=True)

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.n_matched == 2
        assert result.n_missing == 0
        assert result.n_duplicate == 1


class TestEmptyPackSentinel:
    """Test empty pack produces NO_ALIGNED_REAL_REPLAY."""

    def test_empty_dataframe(self):
        """Empty pack → status = NO_ALIGNED_REAL_REPLAY."""
        mainline = _make_mainline(["2026-01-10"], hours=[9, 10])
        result = align_intraday_pack_to_mainline(pd.DataFrame(), mainline)

        assert result.status == "NO_ALIGNED_REAL_REPLAY"
        assert result.n_matched == 0
        assert len(result.aligned) == 0

    def test_none_pack(self):
        """None pack → NO_ALIGNED_REAL_REPLAY."""
        mainline = _make_mainline(["2026-01-10"], hours=[9])
        result = align_intraday_pack_to_mainline(None, mainline)

        assert result.status == "NO_ALIGNED_REAL_REPLAY"

    def test_n_missing_equals_mainline_size(self):
        """When pack is empty, n_missing == number of unique mainline keys."""
        mainline = _make_mainline(["2026-01-10"], hours=list(range(1, 25)))
        result = align_intraday_pack_to_mainline(pd.DataFrame(), mainline)

        assert result.n_missing == 24

    def test_empty_mainline_with_nonempty_pack(self):
        """Non-empty pack but empty mainline → NO_ALIGNED_REAL_REPLAY."""
        mainline = pd.DataFrame(columns=["business_day", "hour_business",
                                          "timestamp", "base_fused_pred"])
        pack = _make_intraday_pack("2026-01-10", target_hours=[9, 10])

        result = align_intraday_pack_to_mainline(pack, mainline)

        assert result.status == "NO_ALIGNED_REAL_REPLAY"
        assert result.n_matched == 0


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
