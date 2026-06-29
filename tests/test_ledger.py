# -*- coding: utf-8 -*-
"""Tests for the Prediction Ledger system."""
from __future__ import annotations
import sys, os, tempfile, shutil
from pathlib import Path
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestLedgerSchema:
    def test_column_count(self):
        from ledger.schema import LEDGER_COLUMNS
        assert len(LEDGER_COLUMNS) == 16

    def test_required_columns_present(self):
        from ledger.schema import LEDGER_COLUMNS
        required = ["run_date", "forecast_date", "hour_business", "timestamp",
                     "target", "model_name", "y_pred", "final_pred",
                     "period", "pipeline_version", "created_at"]
        for col in required:
            assert col in LEDGER_COLUMNS

    def test_timestamp_mapping_hour24(self):
        from ledger.schema import timestamp_from_business_hour, business_hour_from_timestamp
        ts = timestamp_from_business_hour("2026-06-29", 24)
        assert ts.hour == 0
        assert ts.strftime("%Y-%m-%d") == "2026-06-30"
        bd, hb = business_hour_from_timestamp(ts)
        assert bd == "2026-06-29"
        assert hb == 24

    def test_make_empty_ledger(self):
        from ledger.schema import make_empty_ledger, LEDGER_COLUMNS
        df = make_empty_ledger()
        assert df.shape == (0, 16)
        assert list(df.columns) == LEDGER_COLUMNS

    def test_validate_good(self):
        from ledger.schema import validate_ledger_schema, LEDGER_COLUMNS
        df = pd.DataFrame({col: [0] for col in LEDGER_COLUMNS})
        df["hour_business"] = 12
        df["target"] = "dayahead"
        df["period"] = "9_16"
        errors = validate_ledger_schema(df, strict=False)
        assert len(errors) == 0

    def test_validate_bad_hour(self):
        from ledger.schema import validate_ledger_schema, LEDGER_COLUMNS
        df = pd.DataFrame({col: [0] for col in LEDGER_COLUMNS})
        df["hour_business"] = 99
        df["target"] = "dayahead"
        df["period"] = "9_16"
        errors = validate_ledger_schema(df, strict=False)
        assert len(errors) > 0

class TestLedgerQuality:
    def _make_good_ledger(self):
        rows = []
        targets = {
            "dayahead": ["lightgbm", "timesfm", "timemixer"],
            "realtime": ["sgdfnet", "timemixer", "rt916", "timesfm"],
        }
        for target, models in targets.items():
            for fdate in ["2026-06-28", "2026-06-29"]:
                for hb in range(1, 25):
                    for model in models:
                        if hb < 24:
                            ts = pd.Timestamp(fdate) + pd.Timedelta(hours=hb)
                        else:
                            ts = pd.Timestamp(fdate) + pd.Timedelta(days=1)
                        rows.append({
                            "run_date": "2026-06-29",
                            "forecast_date": fdate,
                            "hour_business": hb,
                            "timestamp": str(ts),
                            "target": target,
                            "model_name": model,
                            "y_pred": 100.0 + hb,
                            "base_fused_pred": 110.0,
                            "spike_corrected_pred": float("nan"),
                            "final_pred": 112.0,
                            "y_true": float("nan"),
                            "period": "1_8" if hb <= 8 else ("9_16" if hb <= 16 else "17_24"),
                            "available_data_cutoff": f"{target}_2026-06-29_14:00",
                            "pipeline_version": "r3d_tap_gef_v1",
                            "source_file": "outputs/2026-06-29",
                            "created_at": "2026-06-29T12:00:00",
                        })
        return pd.DataFrame(rows)

    def test_quality_pass(self):
        from ledger.quality import run_ledger_quality_check
        df = self._make_good_ledger()
        report = run_ledger_quality_check(df)
        assert report.passed, f"Expected pass, got: {report.all_errors}"
        assert report.total_rows == 336
        assert report.n_targets == 2

    def test_quality_hour_out_of_range(self):
        from ledger.quality import run_ledger_quality_check
        df = self._make_good_ledger()
        df.loc[0, "hour_business"] = 99
        report = run_ledger_quality_check(df)
        assert not report.passed
        assert len(report.hour_business_out_of_range) > 0

    def test_quality_missing_final_pred(self):
        from ledger.quality import run_ledger_quality_check
        df = self._make_good_ledger()
        df.loc[0, "final_pred"] = float("nan")
        report = run_ledger_quality_check(df, strict=True)
        assert not report.passed
        assert report.missing_final_pred > 0

    def test_quality_duplicates(self):
        from ledger.quality import run_ledger_quality_check
        df = self._make_good_ledger()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        report = run_ledger_quality_check(df)
        assert not report.passed
        assert len(report.duplicate_rows) > 0

class TestLedgerAppend:
    def test_find_pipeline_outputs_missing(self):
        from ledger.append import find_pipeline_outputs
        result = find_pipeline_outputs("2099-01-01", output_root="nonexistent")
        assert all(v is None for v in result.values())

    def test_ledger_append_missing_date(self):
        from ledger.append import ledger_append_from_pipeline_run
        result = ledger_append_from_pipeline_run("2099-01-01", output_root="nonexistent", dry_run=True)
        assert result.get("status") == "error"

    def test_append_to_new_ledger(self):
        from ledger.append import _append_df
        tmpdir = Path(tempfile.mkdtemp())
        ledger_file = tmpdir / "test_ledger.csv"
        try:
            rows = []
            for target in ["dayahead", "realtime"]:
                for hb in range(1, 25):
                    rows.append({
                        "run_date": "2026-06-29",
                        "forecast_date": "2026-06-29",
                        "hour_business": hb,
                        "timestamp": f"2026-06-29 {hb:02d}:00:00" if hb < 24 else "2026-06-30 00:00:00",
                        "target": target,
                        "model_name": "test_model",
                        "y_pred": 100.0,
                        "base_fused_pred": 110.0,
                        "spike_corrected_pred": float("nan"),
                        "final_pred": 112.0,
                        "y_true": float("nan"),
                        "period": "1_8" if hb <= 8 else ("9_16" if hb <= 16 else "17_24"),
                        "available_data_cutoff": f"{target}_2026-06-29_14:00",
                        "pipeline_version": "r3d_tap_gef_v1",
                        "source_file": "outputs/2026-06-29",
                        "created_at": "2026-06-29T12:00:00",
                    })
            df = pd.DataFrame(rows)
            n = _append_df(df, ledger_file, warnings=[])
            assert n == len(rows), f"Expected {len(rows)}, got {n}"
            assert ledger_file.exists()
            warnings = []
            n2 = _append_df(df, ledger_file, warnings=warnings)
            assert n2 == 0, f"Expected 0 (all duplicates), got {n2}"
            assert any("duplicate" in w for w in warnings)
            warnings2 = []
            n3 = _append_df(df, ledger_file, force=True, warnings=warnings2)
            assert n3 == len(rows), f"Expected {len(rows)} (force), got {n3}"
        finally:
            shutil.rmtree(tmpdir)
