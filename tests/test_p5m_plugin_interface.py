# -*- coding: utf-8 -*-
"""
test_p5m_plugin_interface.py — Smoke tests for the P5M Plugin / Interface module.

Verifies:
  1. PredictionTableSpec validates / rejects DataFrames correctly (new P5 contract).
  2. standardize_predictions normalises a raw DataFrame.
  3. ExternalPredictionSource + load_external_predictions works end-to-end.
  4. CorrectionModule ABC enforces the interface.
  5. CorrectionRegistry registers, deduplicates, and runs modules.
  6. MonitorModule ABC enforces the interface.
  7. MonitorRegistry registers, deduplicates, and runs modules.
  8. run_corrections / run_monitors standalone functions work.
  9. PipelineAdapter orchestrates everything without model-name references.
 10. New schema requirements: y_true optional, leakage_safe strict,
     business_day/hour_business primary key, hour 24 mapping, long-format,
     backward-compatible aliases.
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
    COLUMN_ALIASES,
    PredictionTableSpec,
    standardize_predictions,
    apply_column_aliases,
    _construct_timestamp,
    _parse_timestamp,
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
# Helpers: build minimal valid DataFrames / CSVs
# ====================================================================

REQUIRED = [
    "model_name", "business_day", "hour_business", "timestamp",
    "y_pred", "source_file", "prediction_mode", "leakage_safe",
]


def _make_good_df(**overrides) -> pd.DataFrame:
    """Build a minimal valid prediction DataFrame (new P5 contract).

    All required columns present, leakage_safe = true.
    """
    data = {
        "model_name": ["test_model", "test_model"],
        "business_day": ["2026-01-15", "2026-01-15"],
        "hour_business": [12, 13],
        "timestamp": pd.to_datetime(["2026-01-15 12:00:00", "2026-01-15 13:00:00"]),
        "y_pred": [340.0, 345.0],
        "source_file": ["test.csv", "test.csv"],
        "prediction_mode": ["dayahead", "dayahead"],
        "leakage_safe": ["true", "true"],
    }
    for k, v in overrides.items():
        data[k] = v
    return pd.DataFrame(data)


def _good_csv_rows() -> list[list[str]]:
    return [
        REQUIRED,
        ["model_a", "2026-01-15", "12", "2026-01-15 12:00:00",
         "340.0", "test.csv", "dayahead", "true"],
        ["model_a", "2026-01-15", "13", "2026-01-15 13:00:00",
         "345.0", "test.csv", "dayahead", "true"],
    ]


def _rows_to_csv(rows: list[list[str]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerows(rows)
    return buf.getvalue()


def _write_temp_csv(csv_text: str) -> str:
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    tmp.write(csv_text)
    tmp.close()
    return tmp.name


# ====================================================================
# 0. Column alias / mapping helpers
# ====================================================================


class TestColumnAliases:
    def test_target_day_mapped_to_business_day(self):
        df = pd.DataFrame({"target_day": ["2026-01-15"], "hour_business": [12],
                           "model_name": ["m"], "y_pred": [100.0],
                           "timestamp": ["2026-01-15 12:00"], "source_file": ["s"],
                           "prediction_mode": ["dayahead"], "leakage_safe": ["true"]})
        result = apply_column_aliases(df)
        assert "business_day" in result.columns
        assert "target_day" not in result.columns

    def test_ds_mapped_to_timestamp(self):
        df = pd.DataFrame({"ds": ["2026-01-15 12:00"], "model_name": ["m"],
                           "business_day": ["2026-01-15"], "hour_business": [12],
                           "y_pred": [100.0], "source_file": ["s"],
                           "prediction_mode": ["dayahead"], "leakage_safe": ["true"]})
        result = apply_column_aliases(df)
        assert "timestamp" in result.columns
        assert "ds" not in result.columns

    def test_alias_not_applied_when_canonical_exists(self):
        df = pd.DataFrame({"target_day": ["old"], "business_day": ["2026-01-15"],
                           "model_name": ["m"], "hour_business": [12],
                           "timestamp": ["2026-01-15 12:00"], "y_pred": [100.0],
                           "source_file": ["s"], "prediction_mode": ["dayahead"],
                           "leakage_safe": ["true"]})
        result = apply_column_aliases(df)
        assert result["business_day"].iloc[0] == "2026-01-15"  # canonical kept


# ====================================================================
# 0b. Timestamp ↔ (business_day, hour_business)
# ====================================================================


class TestTimestampMapping:
    def test_construct_timestamp_normal(self):
        ts = _construct_timestamp("2026-01-15", 12)
        assert ts.hour == 12
        assert ts.strftime("%Y-%m-%d") == "2026-01-15"

    def test_construct_timestamp_hour_24(self):
        """Hour 24 of Jan 15 = 00:00 Jan 16."""
        ts = _construct_timestamp("2026-01-15", 24)
        assert ts.hour == 0
        assert ts.minute == 0
        assert ts.strftime("%Y-%m-%d") == "2026-01-16"

    def test_parse_timestamp_midnight_to_hour_24(self):
        """00:00 → business_day = prev day, hour = 24."""
        ts = pd.Series(pd.to_datetime(["2026-01-16 00:00:00"]))
        bd, hb = _parse_timestamp(ts)
        assert bd.iloc[0] == "2026-01-15"
        assert hb.iloc[0] == 24

    def test_parse_timestamp_normal_hour(self):
        ts = pd.Series(pd.to_datetime(["2026-01-15 12:00:00"]))
        bd, hb = _parse_timestamp(ts)
        assert bd.iloc[0] == "2026-01-15"
        assert hb.iloc[0] == 12


# ====================================================================
# 1. PredictionTableSpec (new contract)
# ====================================================================


class TestPredictionTableSpec:
    def test_default_spec_has_correct_required_columns(self):
        spec = PredictionTableSpec()
        assert "model_name" in spec.required_columns
        assert "business_day" in spec.required_columns
        assert "hour_business" in spec.required_columns
        assert "timestamp" in spec.required_columns
        assert "y_pred" in spec.required_columns
        assert "source_file" in spec.required_columns
        assert "prediction_mode" in spec.required_columns
        assert "leakage_safe" in spec.required_columns
        assert "y_true" not in spec.required_columns
        assert len(spec.required_columns) == 8

    def test_validate_passes_on_good_df(self):
        spec = PredictionTableSpec()
        df = _make_good_df()
        spec.validate(df)  # should not raise

    def test_validate_raises_on_missing_columns(self):
        df = pd.DataFrame({"model_name": ["m"], "y_pred": [100.0]})
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="missing required columns"):
            spec.validate(df)

    def test_validate_fails_on_leakage_safe_not_true(self):
        df = _make_good_df(leakage_safe=["false", "false"])
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="leakage_safe"):
            spec.validate(df)

    def test_validate_fails_on_leakage_safe_empty(self):
        df = _make_good_df(leakage_safe=["", ""])
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="leakage_safe"):
            spec.validate(df)

    def test_validate_fails_on_bad_prediction_mode(self):
        df = _make_good_df(prediction_mode=["unknown", "unknown"])
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="prediction_mode"):
            spec.validate(df)

    def test_validate_fails_on_bad_hour(self):
        df = _make_good_df(hour_business=[0, 25])
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="hour_business"):
            spec.validate(df)

    def test_validate_fails_on_duplicate_key(self):
        df = _make_good_df()  # both rows: model_a, 2026-01-15, 12 & 13 — unique
        # Make them duplicate
        df.loc[1, "hour_business"] = 12
        spec = PredictionTableSpec()
        with pytest.raises(ValueError, match="Duplicate"):
            spec.validate(df)

    def test_validate_passes_long_format(self):
        df = _make_good_df()
        df.loc[1, "hour_business"] = 12  # duplicate key
        spec = PredictionTableSpec(allow_long_format=True)
        spec.validate(df)  # should not raise


# ====================================================================
# 2. standardize_predictions
# ====================================================================


class TestStandardizePredictions:
    def test_standardize_good_dataframe(self):
        df = _make_good_df()
        result = standardize_predictions(df)
        assert isinstance(result, pd.DataFrame)
        assert all(c in result.columns for c in PredictionTableSpec().required_columns)

    def test_standardize_raises_on_missing_columns(self):
        df = pd.DataFrame({"model_name": ["m"]})
        with pytest.raises(ValueError, match="missing required columns"):
            standardize_predictions(df)

    def test_standardize_infers_timestamp(self):
        """When timestamp is missing, build from business_day + hour_business."""
        df = _make_good_df()
        df = df.drop(columns=["timestamp"])
        result = standardize_predictions(df, infer_timestamp=True)
        assert "timestamp" in result.columns
        assert result["timestamp"].iloc[0] is not pd.NaT

    def test_standardize_infers_hour_business_from_timestamp(self):
        """When hour_business missing, infer from timestamp."""
        df = _make_good_df()
        df = df.drop(columns=["hour_business"])
        result = standardize_predictions(df)
        assert "hour_business" in result.columns
        assert result["hour_business"].iloc[0] == 12

    def test_standardize_maps_hour_24_correctly(self):
        """timestamp=2026-01-02 00:00 → business_day=2026-01-01, hour=24."""
        df = _make_good_df()
        df["timestamp"] = pd.to_datetime(["2026-01-02 00:00:00", "2026-01-02 01:00:00"])
        df["hour_business"] = [0, 1]  # row 1 not midnight → keeps hour 1
        result = standardize_predictions(df)
        assert result["business_day"].iloc[0] == "2026-01-01"
        assert result["hour_business"].iloc[0] == 24
        assert result["hour_business"].iloc[1] == 1  # non-midnight unaffected

    def test_y_true_not_required(self):
        """DataFrame without y_true must still pass."""
        df = _make_good_df()
        df = df.drop(columns=["y_true"], errors="ignore")
        result = standardize_predictions(df)
        assert "y_true" not in result.columns

    def test_y_true_optional_when_present(self):
        """y_true is preserved when present."""
        df = _make_good_df()
        df["y_true"] = [350.0, 360.0]
        result = standardize_predictions(df)
        assert "y_true" in result.columns

    def test_target_day_alias_accepted(self):
        """Old target_day column maps to business_day."""
        df = _make_good_df()
        df = df.drop(columns=["business_day"])
        df["target_day"] = ["2026-01-15", "2026-01-15"]
        result = standardize_predictions(df)
        assert "business_day" in result.columns
        assert result["business_day"].iloc[0] == "2026-01-15"

    def test_ds_alias_accepted(self):
        """Old ds column maps to timestamp."""
        df = _make_good_df()
        df = df.drop(columns=["timestamp"])
        df["ds"] = ["2026-01-15 12:00:00", "2026-01-15 13:00:00"]
        result = standardize_predictions(df)
        assert "timestamp" in result.columns

    def test_different_model_names_preserved(self):
        df = _make_good_df(model_name=["my_custom_model_v42", "my_custom_model_v42"])
        result = standardize_predictions(df)
        assert result["model_name"].iloc[0] == "my_custom_model_v42"

    def test_leakage_safe_is_set_when_missing(self):
        df = _make_good_df()
        df = df.drop(columns=["leakage_safe"])
        result = standardize_predictions(df)
        assert "leakage_safe" in result.columns
        assert result["leakage_safe"].iloc[0] == "true"

    def test_prediction_mode_normalised(self):
        df = _make_good_df(prediction_mode=["DA", "rt"])
        result = standardize_predictions(df)
        assert result["prediction_mode"].iloc[0] == "dayahead"
        assert result["prediction_mode"].iloc[1] == "realtime"


# ====================================================================
# 3. ExternalPredictionSource + load_external_predictions (new contract)
# ====================================================================


class TestExternalLoader:
    def test_load_with_identity_mapping(self):
        rows = _good_csv_rows()
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)
        try:
            source = ExternalPredictionSource(path=tmp_path)
            df = load_external_predictions(source)
            assert len(df) == 2
            assert set(df["model_name"].unique()) == {"model_a"}
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_with_custom_column_mapping(self):
        rows = [
            ["date", "hour", "prediction", "mdl"],
            ["2026-01-15", "12", "340.0", "cust_v1"],
            ["2026-01-15", "13", "345.0", "cust_v1"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)
        try:
            source = ExternalPredictionSource(
                path=tmp_path,
                column_mapping={
                    "date": "business_day",
                    "hour": "hour_business",
                    "prediction": "y_pred",
                    "mdl": "model_name",
                },
                source_file_tag="custom_source",
                prediction_mode_override="realtime",
            )
            df = load_external_predictions(source)
            assert len(df) == 2
            assert all(df["source_file"] == "custom_source")
            assert all(df["prediction_mode"] == "realtime")
            assert df["hour_business"].iloc[0] == 12
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_raises_on_missing_file(self):
        source = ExternalPredictionSource(path="/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            load_external_predictions(source)

    def test_load_missing_model_name_uses_override(self):
        rows = [
            ["business_day", "hour_business", "y_pred", "timestamp"],
            ["2026-01-15", "12", "340.0", "2026-01-15 12:00"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)
        try:
            source = ExternalPredictionSource(
                path=tmp_path,
                column_mapping={
                    "business_day": "business_day",
                    "hour_business": "hour_business",
                    "y_pred": "y_pred",
                    "timestamp": "timestamp",
                },
                model_name_override="no_name_model",
                source_file_tag="test_source",
                prediction_mode_override="external",
            )
            df = load_external_predictions(source)
            assert all(df["model_name"] == "no_name_model")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_no_y_true_still_passes(self):
        """External CSV without y_true must still load successfully."""
        rows = [
            REQUIRED,
            ["model_a", "2026-01-15", "12", "2026-01-15 12:00",
             "340.0", "test.csv", "dayahead", "true"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)
        try:
            source = ExternalPredictionSource(path=tmp_path)
            df = load_external_predictions(source)
            assert len(df) == 1
            assert "y_true" not in df.columns
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_leakage_safe_false_fails(self):
        rows = [
            REQUIRED,
            ["model_a", "2026-01-15", "12", "2026-01-15 12:00",
             "340.0", "test.csv", "dayahead", "false"],
        ]
        csv_text = _rows_to_csv(rows)
        tmp_path = _write_temp_csv(csv_text)
        try:
            source = ExternalPredictionSource(path=tmp_path)
            with pytest.raises(ValueError, match="leakage_safe"):
                load_external_predictions(source)
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
        df = _make_good_df()
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
        df = _make_good_df()
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
        df = _make_good_df()
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
        df = _make_good_df()
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
        df = _make_good_df()
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
        df = _make_good_df()
        result = run_corrections(df, registry)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

    def test_run_monitors(self):
        registry = MonitorRegistry()
        registry.register(_DummyMonitor(name="standalone"))
        df = _make_good_df()
        report = run_monitors(df, registry)
        assert isinstance(report, dict)
        assert "standalone.row_count" in report

    def test_kwargs_forwarded(self):
        registry = CorrectionRegistry()
        mod = _DummyCorrection(add_value=99.0, name="kwargs_test")
        registry.register(mod)
        df = _make_good_df()
        result = registry.run_all(df, extra_param="hello")
        assert "extra_param_received" in result.attrs


# ====================================================================
# 9. PipelineAdapter
# ====================================================================

class TestPipelineAdapter:
    def test_adapter_run_with_explicit_df(self):
        adapter = PipelineAdapter()
        adapter.register_correction(_DummyCorrection(add_value=5.0, name="corr1"))
        adapter.register_monitor(_DummyMonitor(name="mon1"))
        df = _make_good_df()
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
        df = _make_good_df()
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

    def test_frozenset_in_spec(self):
        spec = PredictionTableSpec()
        assert isinstance(spec.valid_prediction_modes, frozenset)


# ====================================================================
# 10. New contract: y_true not required, leakage_safe strict, hour 24
# ====================================================================

class TestP5Contract:
    def test_prediction_csv_without_y_true_passes(self):
        """No y_true column → schema validation passes."""
        df = _make_good_df()
        df = df.drop(columns=["y_true"], errors="ignore")
        spec = PredictionTableSpec()
        result = standardize_predictions(df, spec=spec)
        assert "y_true" not in result.columns

    def test_with_y_true_preserved_as_optional(self):
        """y_true present → preserved, not required."""
        df = _make_good_df()
        df["y_true"] = [350.0, 360.0]
        result = standardize_predictions(df)
        assert "y_true" in result.columns
        assert result["y_true"].iloc[0] == 350.0

    def test_leakage_safe_false_raises(self):
        """leakage_safe=literal false → fail."""
        df = _make_good_df(leakage_safe=["false", "false"])
        with pytest.raises(ValueError, match="leakage_safe"):
            PredictionTableSpec().validate(df)

    def test_leakage_safe_missing_defaults_true(self):
        """leakage_safe column missing → defaulted to true."""
        df = _make_good_df()
        df = df.drop(columns=["leakage_safe"])
        result = standardize_predictions(df)
        assert result["leakage_safe"].iloc[0] == "true"

    def test_business_day_hour_business_primary_key(self):
        """Duplicate (model_name, business_day, hour_business) is rejected."""
        df = _make_good_df()
        df.loc[1, "hour_business"] = 12  # same key as row 0
        with pytest.raises(ValueError, match="Duplicate"):
            PredictionTableSpec().validate(df)

    def test_allow_long_format_permits_same_hour_multiple_models(self):
        """allow_long_format=True → multiple models per hour OK."""
        df = _make_good_df(model_name=["model_a", "model_b"])
        spec = PredictionTableSpec(allow_long_format=True)
        spec.validate(df)  # different model, same hour — fine

    def test_hour_24_mapping(self):
        """Hour 24 → business_day - 1."""
        df = _make_good_df()
        df["timestamp"] = pd.to_datetime(["2026-01-02 00:00:00", "2026-01-02 02:00:00"])
        df["hour_business"] = [0, 2]  # placeholder, row 0 will become 24
        result = standardize_predictions(df)
        assert result["business_day"].iloc[0] == "2026-01-01"
        assert result["hour_business"].iloc[0] == 24

    def test_target_day_alias_in_full_pipeline(self):
        """Old target_day alias works end-to-end."""
        df = _make_good_df()
        df = df.drop(columns=["business_day"])
        df["target_day"] = ["2026-01-15", "2026-01-15"]
        # Also drop timestamp to test inference from aliased columns
        df = df.drop(columns=["timestamp"])
        df["ds"] = ["2026-01-15 12:00:00", "2026-01-15 13:00:00"]
        result = standardize_predictions(df)
        assert "business_day" in result.columns
        assert "timestamp" in result.columns


# ====================================================================
# Dummy implementations for testing
# ====================================================================


class _DummyCorrection(CorrectionModule):
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
