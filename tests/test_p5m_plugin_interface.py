# -*- coding: utf-8 -*-
"""Tests for the P5M Plugin Interface (pipeline_ext).

Coverage
--------
1. Schema validation pass
2. Schema validation fail on missing required field
3. Duplicate (business_day, hour_business) detection
4. Registry register / list / get
5. Correction module apply order
6. leakage_safe == false should fail
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pipeline_ext.io import load_prediction_csv, validate_prediction_dataframe
from pipeline_ext.modules import CorrectionModule, MonitorModule, PredictionProvider
from pipeline_ext.registry import (
    _reset,
    get_module,
    list_corrections,
    list_modules,
    list_monitors,
    list_providers,
    register_correction_module,
    register_monitor_module,
    register_prediction_provider,
)
from pipeline_ext.schema import (
    check_leakage_safe,
    check_uniqueness,
    validate_schema,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_valid_df(**overrides) -> pd.DataFrame:
    """Create a minimal valid prediction DataFrame."""
    data = {
        "model_name": ["test_model"],
        "business_day": ["2026-06-01"],
        "hour_business": [1],
        "timestamp": ["2026-06-01 01:00:00"],
        "y_pred": [100.0],
        "source_file": ["test.csv"],
        "prediction_mode": ["dayahead"],
        "leakage_safe": [True],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _write_csv(df: pd.DataFrame) -> str:
    """Write a DataFrame to a temp CSV and return the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    return tmp.name


# ── 1. Schema validation pass ─────────────────────────────────────────


class TestSchemaValidation:
    def test_valid_dataframe_passes(self):
        df = _make_valid_df()
        # Should not raise
        result = validate_schema(df)
        assert result == []

    def test_valid_csv_loads(self):
        df = _make_valid_df()
        path = _write_csv(df)
        loaded = load_prediction_csv(path)
        assert len(loaded) == 1
        assert loaded["model_name"].iloc[0] == "test_model"
        Path(path).unlink()

    # ── 2. Schema validation fail on missing required field ──────────

    def test_missing_required_field_raises(self):
        df = _make_valid_df().drop(columns=["y_pred"])
        with pytest.raises(ValueError, match="Missing required field"):
            validate_schema(df)

    def test_missing_required_field_csv_fails(self):
        df = _make_valid_df().drop(columns=["model_name"])
        path = _write_csv(df)
        with pytest.raises(ValueError, match="Missing required field"):
            load_prediction_csv(path)
        Path(path).unlink()

    # ── 3. Duplicate (business_day, hour_business) detection ─────────

    def test_duplicate_business_day_hour_raises(self):
        rows = [
            ["model_a", "2026-06-01", 1, "2026-06-01 01:00", 100.0, "a.csv", "dayahead", True],
            ["model_b", "2026-06-01", 1, "2026-06-01 01:00", 101.0, "b.csv", "dayahead", True],
        ]
        df = pd.DataFrame(
            rows,
            columns=[
                "model_name", "business_day", "hour_business", "timestamp",
                "y_pred", "source_file", "prediction_mode", "leakage_safe",
            ],
        )
        with pytest.raises(ValueError, match="Duplicate"):
            check_uniqueness(df)

        # CSV load should also fail
        path = _write_csv(df)
        with pytest.raises(ValueError, match="Duplicate"):
            load_prediction_csv(path)
        Path(path).unlink()

    # ── 6. leakage_safe == false should fail ─────────────────────────

    def test_leakage_safe_false_raises(self):
        df = _make_valid_df(leakage_safe=[False])
        with pytest.raises(ValueError, match="leakage_safe"):
            check_leakage_safe(df)

    def test_leakage_safe_false_csv_fails(self):
        df = _make_valid_df(leakage_safe=[False])
        path = _write_csv(df)
        with pytest.raises(ValueError, match="leakage_safe"):
            load_prediction_csv(path)
        Path(path).unlink()

    def test_leakage_safe_string_false_also_fails(self):
        df = _make_valid_df(leakage_safe=["false"])
        with pytest.raises(ValueError, match="leakage_safe"):
            check_leakage_safe(df)

    def test_leakage_safe_missing_col_raises(self):
        df = _make_valid_df().drop(columns=["leakage_safe"])
        with pytest.raises(ValueError, match="leakage_safe"):
            check_leakage_safe(df)


# ── 4. Registry register / list / get ─────────────────────────────────


class TestRegistry:
    def setup_method(self):
        _reset()

    def test_register_and_list_providers(self):
        class DummyProvider(PredictionProvider):
            def load_predictions(self, path: str) -> pd.DataFrame:
                return _make_valid_df()

        register_prediction_provider("dummy", DummyProvider())
        assert "dummy" in list_providers()
        all_mods = list_modules()
        assert "dummy" in all_mods["providers"]

    def test_register_and_list_corrections(self):
        class DoubleCorrection(CorrectionModule):
            name = "double"
            def apply(self, df: pd.DataFrame) -> pd.DataFrame:
                return df

        register_correction_module("double", DoubleCorrection())
        assert "double" in list_corrections()

    def test_register_and_list_monitors(self):
        class SimpleMonitor(MonitorModule):
            name = "simple"
            def run(self, df: pd.DataFrame) -> dict:
                return {"ok": True}

        register_monitor_module("simple", SimpleMonitor())
        assert "simple" in list_monitors()

    def test_get_module_returns_correct_type(self):
        class MockProvider(PredictionProvider):
            def load_predictions(self, path: str) -> pd.DataFrame:
                return _make_valid_df()

        register_prediction_provider("mock", MockProvider())
        mod = get_module("mock")
        assert isinstance(mod, PredictionProvider)

    def test_get_module_not_found_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get_module("nonexistent")

    def test_register_wrong_type_raises(self):
        with pytest.raises(TypeError):
            register_prediction_provider("bad", "not_a_provider")  # type: ignore[arg-type]

    def test_list_modules_structure(self):
        result = list_modules()
        assert set(result) == {"providers", "corrections", "monitors"}


# ── 5. Correction module apply order ──────────────────────────────────


class TestCorrectionOrder:
    def setup_method(self):
        _reset()

    def test_corrections_apply_in_registration_order(self):
        """Verify corrections are applied in the expected sequence."""
        order: list[str] = []

        class FirstCorrection(CorrectionModule):
            name = "first"
            def apply(self, df: pd.DataFrame) -> pd.DataFrame:
                order.append("first")
                return df

        class SecondCorrection(CorrectionModule):
            name = "second"
            def apply(self, df: pd.DataFrame) -> pd.DataFrame:
                order.append("second")
                return df

        register_correction_module("first", FirstCorrection())
        register_correction_module("second", SecondCorrection())

        df = _make_valid_df()
        from pipeline_ext.pipeline import DryRunPipeline

        pipeline = DryRunPipeline(out_dir=tempfile.mkdtemp())
        pipeline._run(
            df, source_label="test",
            correction_names=["first", "second"],
            monitor_names=[],
        )
        assert order == ["first", "second"], f"Expected order first->second, got {order}"

    def test_corrections_modify_dataframe(self):
        """Verify a correction actually changes the data."""

        class AddOneCorrection(CorrectionModule):
            name = "add_one"
            def apply(self, df: pd.DataFrame) -> pd.DataFrame:
                out = df.copy()
                out["y_pred"] = out["y_pred"] + 1
                return out

        df = _make_valid_df(y_pred=[100.0])
        mod = AddOneCorrection()
        result = mod.apply(df)
        assert result["y_pred"].iloc[0] == 101.0


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_dataframe_raises_on_duplicate_check(self):
        df = pd.DataFrame(columns=["business_day", "hour_business"])
        result = check_uniqueness(df, raise_on_error=False)
        assert isinstance(result, list)

    def test_validate_prediction_dataframe_in_memory(self):
        df = _make_valid_df()
        result = validate_prediction_dataframe(df)
        assert len(result) == 1

    def test_missing_y_pred_raises(self):
        df = _make_valid_df().drop(columns=["y_pred"])
        path = _write_csv(df)
        with pytest.raises(ValueError, match="Missing required field"):
            load_prediction_csv(path)
        Path(path).unlink()

    def test_csv_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prediction_csv("/nonexistent/path.csv")
