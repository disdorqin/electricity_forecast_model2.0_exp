# -*- coding: utf-8 -*-
"""Dry-run pipeline orchestration for the plugin interface.

This module is **not** the production pipeline.  It provides a
lightweight ``DryRunPipeline`` that:

  1. Loads predictions via a registered PredictionProvider (or direct path).
  2. Validates against the unified schema.
  3. Applies registered CorrectionModule instances in order.
  4. Runs registered MonitorModule instances and collects results.
  5. Writes outputs (corrected CSV + monitor report) to a user-specified
     output directory.

The intent is to *smoke-test* the integration of external models before
they are wired into the actual production pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline_ext.io import load_prediction_csv, validate_prediction_dataframe
from pipeline_ext.registry import (
    get_module,
    list_corrections,
    list_monitors,
    list_providers,
)
from pipeline_ext.schema import check_uniqueness

logger = logging.getLogger(__name__)


@dataclass
class DryRunResult:
    """Result of a dry-run pipeline execution."""

    prediction_path: str
    row_count: int
    correction_order: list[str]
    monitor_results: dict[str, dict[str, Any]]
    output_path: Path | None = None


class DryRunPipeline:
    """A lightweight, non-production pipeline for smoke-testing integrations."""

    def __init__(self, out_dir: str | Path = "outputs/plugin_smoke"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def run_from_path(
        self,
        prediction_path: str | Path,
        correction_modules: list[str] | None = None,
        monitor_modules: list[str] | None = None,
        allow_long_format: bool = False,
    ) -> DryRunResult:
        """Run the dry-run pipeline loading predictions from *prediction_path*.

        Parameters
        ----------
        prediction_path : str | Path
            Path to a prediction CSV.
        correction_modules : list[str] | None
            Ordered names of correction modules to apply.
            If None, applies all registered corrections in registration order.
        monitor_modules : list[str] | None
            Names of monitor modules to run.
            If None, runs all registered monitors.
        allow_long_format : bool
            Passed through to :func:`~pipeline_ext.io.load_prediction_csv`.

        Returns
        -------
        DryRunResult
        """
        logger.info("Dry-run pipeline starting — loading %s", prediction_path)
        df = load_prediction_csv(prediction_path, allow_long_format=allow_long_format)
        return self._run(df, prediction_path, correction_modules, monitor_modules)

    def run_from_provider(
        self,
        provider_name: str,
        path: str,
        correction_modules: list[str] | None = None,
        monitor_modules: list[str] | None = None,
    ) -> DryRunResult:
        """Run the dry-run pipeline using a registered PredictionProvider.

        Parameters
        ----------
        provider_name : str
            Registered provider name.
        path : str
            Path passed to the provider's ``load_predictions`` method.
        correction_modules, monitor_modules : as in :meth:`run_from_path`.

        Returns
        -------
        DryRunResult
        """
        provider = get_module(provider_name)
        logger.info("Dry-run pipeline starting — provider '%s' loading %s", provider_name, path)
        df = provider.load_predictions(path)
        df = validate_prediction_dataframe(df)
        return self._run(df, f"{provider_name}:{path}", correction_modules, monitor_modules)

    # ── Internal ───────────────────────────────────────────────────────

    def _run(
        self,
        df: pd.DataFrame,
        source_label: str,
        correction_names: list[str] | None,
        monitor_names: list[str] | None,
    ) -> DryRunResult:
        if correction_names is None:
            correction_names = sorted(list_corrections())
        if monitor_names is None:
            monitor_names = sorted(list_monitors())

        # Apply corrections in order
        for name in correction_names:
            module = get_module(name)
            logger.info("Applying correction: %s", name)
            df = module.apply(df)

        # Run monitors
        monitor_results: dict[str, dict[str, Any]] = {}
        for name in monitor_names:
            module = get_module(name)
            logger.info("Running monitor: %s", name)
            monitor_results[name] = module.run(df)

        # Write outputs
        out_csv = self.out_dir / f"corrected_{Path(source_label).stem}.csv"
        df.to_csv(out_csv, index=False)
        logger.info("Corrected output written to %s", out_csv)

        out_report = self.out_dir / "monitor_report.json"
        report = {
            "source": str(source_label),
            "corrections_applied": correction_names,
            "monitors_run": monitor_names,
            "monitor_results": {
                k: _serialisable(v) for k, v in monitor_results.items()
            },
        }
        with open(out_report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("Monitor report written to %s", out_report)

        return DryRunResult(
            prediction_path=str(source_label),
            row_count=len(df),
            correction_order=correction_names,
            monitor_results=monitor_results,
            output_path=out_csv,
        )


# ── Convenience entry-point ────────────────────────────────────────────


def run_dry_run(
    prediction_path: str | Path,
    correction_modules: list[str] | None = None,
    monitor_modules: list[str] | None = None,
    out_dir: str | Path = "outputs/plugin_smoke",
    allow_long_format: bool = False,
) -> DryRunResult:
    """One-shot convenience wrapper around :class:`DryRunPipeline`."""
    pipeline = DryRunPipeline(out_dir=out_dir)
    return pipeline.run_from_path(
        prediction_path=prediction_path,
        correction_modules=correction_modules,
        monitor_modules=monitor_modules,
        allow_long_format=allow_long_format,
    )


def _serialisable(obj: Any) -> Any:
    """Convert a value to a JSON-serialisable type."""
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialisable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if obj is None:
        return None
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return _serialisable(obj.to_dict(orient="list"))
    return str(obj)
