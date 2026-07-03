"""Tests for corrections.intraday_tracker.manifest — Phase 11."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.manifest import (
    IntradayManifest,
    build_manifest,
    write_manifest_and_report,
)


def _make_stats(**overrides):
    stats = {
        "intraday_enabled": True,
        "intraday_mode": "shadow",
        "prediction_mode": "INTRADAY",
        "pack_rows": 10,
        "matched_rows": 8,
        "applied_rows": 0,
        "shadow_rows": 8,
        "disabled_rows": 0,
        "avg_fusion_weight": 0.12,
        "avg_confidence": 0.6,
        "policy_counts": {"LOW_WEIGHT": 8},
        "guardrail_counts": {},
        "fallback_reason": None,
        "safe_fallback": True,
    }
    stats.update(overrides)
    return stats


class TestBuildManifest:
    def test_build_manifest_from_stats(self):
        stats = _make_stats()
        m = build_manifest(stats, pack_path="/path/to/pack.csv")
        assert m.intraday_enabled is True
        assert m.intraday_mode == "shadow"
        assert m.pack_rows == 10
        assert m.matched_rows == 8
        assert m.pack_path == "/path/to/pack.csv"

    def test_manifest_to_dict(self):
        stats = _make_stats()
        m = build_manifest(stats)
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "intraday_enabled" in d
        assert "pack_rows" in d
        assert "policy_counts" in d


class TestWriteManifestAndReport:
    def test_write_manifest_creates_json(self, tmp_path):
        stats = _make_stats()
        m = build_manifest(stats, pack_path="test.csv")
        write_manifest_and_report(m, str(tmp_path))
        json_path = tmp_path / "intraday_mainline_manifest.json"
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["intraday_mode"] == "shadow"

    def test_write_report_creates_md(self, tmp_path):
        stats = _make_stats()
        m = build_manifest(stats)
        write_manifest_and_report(m, str(tmp_path))
        md_path = tmp_path / "intraday_application_report.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "Intraday Tracker" in content

    def test_shadow_report_text(self, tmp_path):
        stats = _make_stats(intraday_mode="shadow")
        m = build_manifest(stats)
        write_manifest_and_report(m, str(tmp_path))
        content = (tmp_path / "intraday_application_report.md").read_text(encoding="utf-8")
        assert "NOT changed" in content or "shadow" in content.lower()

    def test_off_report_text(self, tmp_path):
        stats = _make_stats(intraday_mode="off", intraday_enabled=False)
        m = build_manifest(stats)
        write_manifest_and_report(m, str(tmp_path))
        content = (tmp_path / "intraday_application_report.md").read_text(encoding="utf-8")
        assert "disabled" in content.lower() or "off" in content.lower()

    def test_intraday_rows_csv(self, tmp_path):
        stats = _make_stats()
        m = build_manifest(stats)
        rows = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        write_manifest_and_report(m, str(tmp_path), intraday_rows_df=rows)
        csv_path = tmp_path / "intraday_rows.csv"
        assert csv_path.exists()
        loaded = pd.read_csv(csv_path)
        assert len(loaded) == 2

    def test_full_day_check_in_report(self, tmp_path):
        stats = _make_stats(prediction_mode="FULL_DAY")
        m = build_manifest(stats)
        write_manifest_and_report(m, str(tmp_path))
        content = (tmp_path / "intraday_application_report.md").read_text(encoding="utf-8")
        assert "FULL_DAY" in content
