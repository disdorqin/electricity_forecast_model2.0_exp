"""Tests for corrections.intraday_tracker.adapter — Phase 11."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.adapter import (
    load_intraday_pack,
    normalize_intraday_pack,
    validate_intraday_pack,
)
from corrections.intraday_tracker.schema import MAINLINE_OUTPUT_FIELDS


def _make_pack(n=5, mode="INTRADAY", cutoff=12, confidence=0.6, fusion_weight=0.12):
    """Create a synthetic intraday pack."""
    rows = []
    for i in range(n):
        rows.append({
            "business_day": "2026-02-15",
            "cutoff_hour": cutoff,
            "target_hour": cutoff + 1 + i,
            "ds": "2026-02-15",
            "mode": mode,
            "base_model_name": "sgdfnet",
            "base_pred": 100.0 + i,
            "intraday_base_correction": 10.0,
            "intraday_model_weight": 0.5,
            "intraday_pre_guardrail_correction": 5.0,
            "intraday_guardrail_weight": 1.0,
            "intraday_final_correction": 5.0,
            "intraday_corrected_pred": 105.0 + i,
            "intraday_confidence": confidence,
            "policy_decision": "LOW_WEIGHT",
            "fusion_weight": fusion_weight,
            "shadow_only_flag": False,
            "guardrail_reason": "",
            "observed_hours": "[9,10,11,12]",
            "n_observed": 4,
            "residual_std_today": 50.0,
            "bias_direction": "positive",
        })
    return pd.DataFrame(rows)


class TestLoadIntradayPack:
    def test_load_missing_file_returns_empty(self, tmp_path):
        df = load_intraday_pack(str(tmp_path / "nonexistent.csv"))
        assert len(df) == 0

    def test_load_existing_file(self, tmp_path):
        pack = _make_pack(3)
        path = tmp_path / "pack.csv"
        pack.to_csv(path, index=False)
        df = load_intraday_pack(str(path))
        assert len(df) == 3


class TestNormalizeIntradayPack:
    def test_normalize_empty_pack(self):
        df = normalize_intraday_pack(pd.DataFrame())
        assert len(df) == 0
        for col in MAINLINE_OUTPUT_FIELDS:
            assert col in df.columns

    def test_normalize_fills_defaults(self):
        # Pack with only some columns
        partial = pd.DataFrame({
            "business_day": ["2026-02-15"],
            "target_hour": [13],
            "intraday_corrected_pred": [110.0],
        })
        df = normalize_intraday_pack(partial)
        assert len(df) == 1
        assert "policy_decision" in df.columns
        assert df["policy_decision"].iloc[0] == "DISABLED"
        assert df["fusion_weight"].iloc[0] == 0.0


class TestValidateIntradayPack:
    def test_validate_valid_pack(self):
        pack = _make_pack()
        norm = normalize_intraday_pack(pack)
        result = validate_intraday_pack(norm, mode="online")
        assert result.valid

    def test_validate_mode_not_intraday(self):
        pack = _make_pack(mode="FULL_DAY")
        norm = normalize_intraday_pack(pack)
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid
        assert any("INTRADAY" in e for e in result.errors)

    def test_validate_target_le_cutoff(self):
        pack = _make_pack()
        pack.loc[0, "target_hour"] = 5  # less than cutoff 12
        norm = normalize_intraday_pack(pack)
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid

    def test_validate_fusion_weight_range(self):
        pack = _make_pack(fusion_weight=0.5)  # > 0.3
        norm = normalize_intraday_pack(pack)
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid

    def test_validate_confidence_range(self):
        pack = _make_pack(confidence=1.5)  # > 1.0
        norm = normalize_intraday_pack(pack)
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid

    def test_validate_online_no_ytrue(self):
        pack = _make_pack()
        pack["y_true"] = 100.0
        norm = normalize_intraday_pack(pack)
        norm["y_true"] = 100.0
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid
        assert any("y_true" in e for e in result.errors)

    def test_validate_duplicate_rows(self):
        pack = _make_pack(3)
        # Duplicate first row
        dup = pd.concat([pack, pack.iloc[[0]]], ignore_index=True)
        norm = normalize_intraday_pack(dup)
        result = validate_intraday_pack(norm, mode="online")
        assert not result.valid
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_validate_empty_pack_warning(self):
        result = validate_intraday_pack(pd.DataFrame(), mode="online")
        assert len(result.warnings) > 0
