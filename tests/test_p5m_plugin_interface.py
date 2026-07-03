# -*- coding: utf-8 -*-
"""
test_p5m_plugin_interface.py — Smoke tests for the P5M Plugin / Interface module.

Verifies that:
  1. PredictionTableSpec validates / rejects DataFrames correctly.
  2. standardize_predictions normalises a raw DataFrame.
  3. ExternalPredictionSource + load_external_predictions works end-to-end.
  4. CorrectionModule ABC enforces the interface.
  5. CorrectionRegistry registers, deduplicates, and runs modules.
  6. MonitorModule ABC enforces the interface.
  7. MonitorRegistry registers, deduplicates, and runs modules.
  8. run_corrections / run_monitors standalone functions work.
  9. PipelineAdapter orchestrates everything without model-name references.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from plugin.schema import (
    DEFAULT_COLUMN_MAP,
    PredictionTableSpec,
    standardize_predictions,
)
from plugin.external_loader import (
    ExternalPredictionSource,
    load_external_predictions,
)
from plugin.correction_base import CorrectionModule
from plugin.correction_registry import CorrectionRegistry, run_corrections
from plugin.monitor_base import MonitorModule
from plugin.monitor_registry import MonitorRegistry, run_monitors
from plugin.pipeline_adapter import PipelineAdapter


# ====================================================================
# 1. PredictionTableSpec
# ====================================================================


class TestPredictionTableSpec:
    def test_default_spec_has_all_required_columns(self):
        spec = PredictionTableSpec()
        assert "task" in spec.required_columns
        assert "model_name" in spec.required_columns
        assert "target_day" in spec.required_columns
        assert "ds" in spec.required_columns
        assert "period" in spec.required_columns
        assert "hour_business" in spec.required_columns
        assert "y_true" in spec.required_columns
        assert "y_pred" in spec.required_columns
        assert len(spec.required_columns) == 8

    def test_validate_passes_on_good_df(self):
        df = _make_good_prediction_df()
        spec = PredictionTableSpec()
        # Should not raise
        spec.validate(df)

    def test_validate_raises_on_missing_columns(self):
        df = pd.DataFrame({"task": ["dayahead"], "y_pred": [100.0]})
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="missing required columns"):
            spec.validate(df)

    def test_validate_raises_on_bad_task(self):
        df = _make_good_prediction_df()
        df["task"] = "unknown_task"
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="Unsupported task"):
            spec.validate(df)

    def test_validate_raises_on_bad_period(self):
        df = _make_good_prediction_df()
        df["period"] = "25_32"
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="Unsupported period"):
            spec.validate(df)


# ====================================================================
# 2. standardize_predictions
# ====================================================================


class TestStandardizePredictions:
    def test_standardize_good_dataframe(self):
        df = _make_good_prediction_df()
        result = standardize_predictions(df)
        assert isinstance(result, pd.DataFrame)
        assert all(c in result.columns for c in PredictionTableSpec().required_columns)
        assert result["task"].iloc[0] == "dayahead"
        assert result["period"].iloc[0] == "9_16"
        assert result["hour_business"].iloc[0] == 12

    def test_standardize_normalises_task_values(self):
        df = _make_good_prediction_df(task="DA")
        result = standardize_predictions(df)
        assert result["task"].iloc[0] == "dayahead"

        df2 = _make_good_prediction_df(task="rt")
        result2 = standardize_predictions(df2)
        assert result2["task"].iloc[0] == "realtime"

    def test_standardize_infers_period(self):
        df = _make_good_prediction_df(period="")
        df["hour_business"] = 3
        result = standardize_predictions(df)
        assert result["period"].iloc[0] == "1_8"

    def test_standardize_raises_on_missing_columns(self):
        df = pd.DataFrame({"task": ["dayahead"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            standardize_predictions(df)

    def test_standardize_raises_on_nan_values(self):
        df = _make_good_prediction_df()
        df["y_pred"] = float("nan")
        with pytest.raises(ValueError, match="NaN"):
            standardize_predictions(df)

    def test_different_model_names_preserved(self):
        df = _make_good_prediction_df(model_name="my_custom_model_v42")
        result = standardize_predictions(df)
        assert result["model_name"].iloc[0] == "my_custom_model_v42"


# ====================================================================
# 3. ExternalPredictionSource + load_external_predictions
# ====================================================================


class TestExternalLoader:
    def test_load_with_identity_mapping(self):
        rows = _good_csv_rows()
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)

        try:
            source = ExternalPredictionSource(
                path=tmp_path,
                column_mapping=None,  # identity — CSV already has canonical names
            )
            df = load_external_predictions(source)
            assert len(df) == 2
            assert set(df["model_name"].unique()) == {"model_a"}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_with_custom_column_mapping(self):
        """CSV uses 'date' and 'hour' instead of canonical names."""
        rows = [
            ["date", "hour", "actual", "prediction"],
            ["2026-01-15", "12", "350.0", "340.0"],
            ["2026-01-15", "13", "360.0", "345.0"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)

        try:
            source = ExternalPredictionSource(
                path=tmp_path,
                column_mapping={
                    "date": "target_day",
                    "hour": "hour_business",
                    "actual": "y_true",
                    "prediction": "y_pred",
                },
                model_name_override="custom_model",
                task_override="realtime",
            )
            df = load_external_predictions(source)
            assert len(df) == 2
            assert all(df["model_name"] == "custom_model")
            assert all(df["task"] == "realtime")
            assert df["hour_business"].iloc[0] == 12
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_raises_on_missing_file(self):
        source = ExternalPredictionSource(path="/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            load_external_predictions(source)

    def test_load_with_model_name_override_only(self):
        """CSV missing model_name column gets it from override."""
        rows = [
            ["target_day", "hour_business", "y_true", "y_pred", "task"],
            ["2026-01-15", "12", "350.0", "340.0", "dayahead"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)

        try:
            source = ExternalPredictionSource(
                path=tmp_path,
                column_mapping={
                    "target_day": "target_day",
                    "hour_business": "hour_business",
                    "y_true": "y_true",
                    "y_pred": "y_pred",
                    "task": "task",
                    "model_name": "model_name",
                },
                model_name_override="no_name_model",
            )
            df = load_external_predictions(source)
            assert all(df["model_name"] == "no_name_model")
        finally:
            Path(tmp_path).unlink(missing_ok=True)


# ====================================================================
# 4. CorrectionModule ABC
# ====================================================================


class TestCorrectionModule:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            CorrectionModule()  # type: ignore[abstract]

    def test_concrete_module_works(self):
        module = _DummyCorrection(add_value=10.0)
        assert module.name == "dummy_correction"

        df = _make_good_prediction_df()
        assert module.validate(df) is True

        result = module.correct(df)
        assert "dummy_correction_adjusted" in result.columns

    def test_validate_returns_false_for_bad_df(self):
        module = _DummyCorrection(add_value=10.0)
        bad_df = pd.DataFrame({"foo": [1, 2, 3]})
        assert module.validate(bad_df) is False


# ====================================================================
# 5. CorrectionRegistry
# ====================================================================


class TestCorrectionRegistry:
    def test_register_and_run(self):
        registry = CorrectionRegistry()
        registry.register(_DummyCorrection(add_value=5.0))
        registry.register(_DummyCorrection(add_value=3.0, name="second"))

        df = _make_good_prediction_df()
        result = registry.run_all(df)
        assert len(result) == len(df)

    def test_duplicate_name_raises(self):
        registry = CorrectionRegistry()
        registry.register(_DummyCorrection(add_value=1.0, name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_DummyCorrection(add_value=2.0, name="dup"))

    def test_unregister(self):
        registry = CorrectionRegistry()
        mod = _DummyCorrection(add_value=1.0, name="tmp")
        registry.register(mod)
        assert "tmp" in registry.names
        registry.unregister("tmp")
        assert "tmp" not in registry.names

    def test_get_returns_none_for_missing(self):
        registry = CorrectionRegistry()
        assert registry.get("nonexistent") is None

    def test_non_correction_module_raises(self):
        registry = CorrectionRegistry()
        with pytest.raises(TypeError, match="CorrectionModule"):
            registry.register("not_a_module")  # type: ignore[arg-type]

    def test_validate_all(self):
        registry = CorrectionRegistry()
        registry.register(_DummyCorrection(add_value=1.0, name="m1"))
        registry.register(_DummyCorrection(add_value=2.0, name="m2"))
        df = _make_good_prediction_df()
        results = registry.validate_all(df)
        assert results == {"m1": True, "m2": True}

    def test_len(self):
        registry = CorrectionRegistry()
        assert len(registry) == 0
        registry.register(_DummyCorrection(add_value=1.0, name="a"))
        assert len(registry) == 1
        registry.register(_DummyCorrection(add_value=2.0, name="b"))
        assert len(registry) == 2


# ====================================================================
# 6. MonitorModule ABC
# ====================================================================


class TestMonitorModule:
    def test_abc_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            MonitorModule()  # type: ignore[abstract]

    def test_concrete_monitor_works(self):
        module = _DummyMonitor()
        assert module.name == "dummy_monitor"

        df = _make_good_prediction_df()
        result = module.monitor(df)
        assert isinstance(result, dict)
        assert "row_count" in result
        assert result["row_count"] == len(df)


# ====================================================================
# 7. MonitorRegistry
# ====================================================================


class TestMonitorRegistry:
    def test_register_and_run(self):
        registry = MonitorRegistry()
        registry.register(_DummyMonitor(name="m1"))
        registry.register(_DummyMonitor(name="m2"))

        df = _make_good_prediction_df()
        report = registry.run_all(df)
        assert "m1.row_count" in report
        assert "m2.row_count" in report

    def test_duplicate_name_raises(self):
        registry = MonitorRegistry()
        registry.register(_DummyMonitor(name="dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_DummyMonitor(name="dup"))

    def test_unregister(self):
        registry = MonitorRegistry()
        mod = _DummyMonitor(name="tmp")
        registry.register(mod)
        assert "tmp" in registry.names
        registry.unregister("tmp")
        assert "tmp" not in registry.names

    def test_get_returns_none_for_missing(self):
        registry = MonitorRegistry()
        assert registry.get("nonexistent") is None

    def test_non_monitor_module_raises(self):
        registry = MonitorRegistry()
        with pytest.raises(TypeError, match="MonitorModule"):
            registry.register(42)  # type: ignore[arg-type]

    def test_len(self):
        registry = MonitorRegistry()
        assert len(registry) == 0
        registry.register(_DummyMonitor(name="x"))
        assert len(registry) == 1


# ====================================================================
# 8. run_corrections / run_monitors standalone
# ====================================================================


class TestStandaloneFunctions:
    def test_run_corrections(self):
        registry = CorrectionRegistry()
        registry.register(_DummyCorrection(add_value=7.0, name="standalone"))
        df = _make_good_prediction_df()
        result = run_corrections(df, registry)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_run_monitors(self):
        registry = MonitorRegistry()
        registry.register(_DummyMonitor(name="standalone"))
        df = _make_good_prediction_df()
        report = run_monitors(df, registry)
        assert isinstance(report, dict)
        assert "standalone.row_count" in report

    def test_kwargs_forwarded(self):
        """Verify kwargs pass through to module methods."""
        registry = CorrectionRegistry()
        mod = _DummyCorrection(add_value=99.0, name="kwargs_test")
        registry.register(mod)
        df = _make_good_prediction_df()
        result = registry.run_all(df, extra_param="hello")
        assert "extra_param_received" in result.attrs


# ====================================================================
# 9. PipelineAdapter (end-to-end, no model names)
# ====================================================================


class TestPipelineAdapter:
    def test_adapter_run_with_explicit_df(self):
        adapter = PipelineAdapter()
        adapter.register_correction(_DummyCorrection(add_value=5.0, name="corr1"))
        adapter.register_monitor(_DummyMonitor(name="mon1"))

        df = _make_good_prediction_df()
        result, report = adapter.run(df=df)
        assert isinstance(result, pd.DataFrame)
        assert isinstance(report, dict)
        assert len(result) == len(df)
        assert "mon1.row_count" in report

    def test_adapter_run_with_external_sources(self):
        rows = _good_csv_rows()
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)

        try:
            adapter = PipelineAdapter()
            source = ExternalPredictionSource(
                path=tmp_path,
                model_name_override="ext_model",
            )
            adapter.register_external_source(source)
            adapter.register_monitor(_DummyMonitor(name="ext_mon"))

            result, report = adapter.run()
            assert len(result) == 2
            assert all(result["model_name"] == "ext_model")
            assert "ext_mon.row_count" in report
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_adapter_no_sources_returns_empty(self):
        adapter = PipelineAdapter()
        result, report = adapter.run()
        assert result.empty
        assert report == {}

    def test_adapter_to_csv(self):
        adapter = PipelineAdapter()
        df = _make_good_prediction_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = adapter.to_csv(df, Path(tmpdir) / "out.csv")
            assert out_path.exists()
            reloaded = pd.read_csv(out_path)
            assert len(reloaded) == len(df)

    def test_adapter_external_sources_property(self):
        adapter = PipelineAdapter()
        assert adapter.external_sources == []
        src = ExternalPredictionSource(path="dummy.csv")
        adapter.register_external_source(src)
        assert len(adapter.external_sources) == 1

    def test_registries_accessible_via_adapter(self):
        adapter = PipelineAdapter()
        assert isinstance(adapter.correction_registry, CorrectionRegistry)
        assert isinstance(adapter.monitor_registry, MonitorRegistry)

    def test_standardize_predictions_with_spec(self):
        spec = PredictionTableSpec()
        df = _make_good_prediction_df()
        result = standardize_predictions(df, spec=spec)
        assert not result.empty

    def test_standardize_preserves_realtime_task(self):
        df = _make_good_prediction_df(task="realtime")
        result = standardize_predictions(df)
        assert result["task"].iloc[0] == "realtime"

    def test_frozenset_in_spec(self):
        spec = PredictionTableSpec()
        assert isinstance(spec.valid_tasks, frozenset)
        assert isinstance(spec.valid_periods, frozenset)


# ====================================================================
# Helpers
# ====================================================================


def _make_good_prediction_df(
    task: str = "dayahead",
    period: str = "9_16",
    model_name: str = "test_model",
) -> pd.DataFrame:
    """Build a minimal valid prediction DataFrame."""
    return pd.DataFrame({
        "task": [task, task],
        "model_name": [model_name, model_name],
        "target_day": ["2026-01-15", "2026-01-15"],
        "ds": ["2026-01-15 12:00:00", "2026-01-15 13:00:00"],
        "period": [period, period],
        "hour_business": [12, 13],
        "y_true": [350.0, 360.0],
        "y_pred": [340.0, 345.0],
    })


def _good_csv_rows() -> list[list[str]]:
    """Return a list of rows (header + data) in canonical column order."""
    return [
        ["task", "model_name", "target_day", "ds", "period",
         "hour_business", "y_true", "y_pred"],
        ["dayahead", "model_a", "2026-01-15", "2026-01-15 12:00",
         "9_16", "12", "350.0", "340.0"],
        ["dayahead", "model_a", "2026-01-15", "2026-01-15 13:00",
         "9_16", "13", "360.0", "345.0"],
    ]


def _rows_to_csv(rows: list[list[str]]) -> str:
    """Convert a list of rows (header + data) to a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue()


def _write_temp_csv(csv_text: str) -> str:
    """Write a CSV string to a temp file and return the path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write(csv_text)
    tmp.close()
    return tmp.name


# ------------------------------------------------------------------
# Dummy implementations for testing
# ------------------------------------------------------------------


class _DummyCorrection(CorrectionModule):
    """A dummy correction that adds a fixed value to y_pred."""

    def __init__(self, add_value: float = 0.0, name: str = "dummy_correction"):
        self._name = name
        self._add_value = add_value

    @property
    def name(self) -> str:
        return self._name

    def correct(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        result = df.copy()
        result[f"{self._name}_adjusted"] = result["y_pred"] + self._add_value
        if kwargs:
            result.attrs["extra_param_received"] = True
        return result

    def validate(self, df: pd.DataFrame) -> bool:
        return "y_pred" in df.columns


class _DummyMonitor(MonitorModule):
    """A dummy monitor that reports row count and null counts."""

    def __init__(self, name: str = "dummy_monitor"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def monitor(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "row_count": len(df),
            "null_y_pred": int(df["y_pred"].isna().sum()) if "y_pred" in df.columns else 0,
        }
