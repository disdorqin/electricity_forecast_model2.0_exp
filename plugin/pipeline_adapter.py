# -*- coding: utf-8 -*-
"""
pipeline_adapter.py — Unified entry point for production pipeline scripts.

The adapter owns the registries and exposes a high-level API that does not
reference any concrete model name.  Pipeline scripts construct an adapter,
register their modules, and call ``run()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from plugin.correction_base import CorrectionModule
from plugin.correction_registry import CorrectionRegistry
from plugin.external_loader import (
    ExternalPredictionSource,
    load_external_predictions,
)
from plugin.monitor_base import MonitorModule
from plugin.monitor_registry import MonitorRegistry
from plugin.schema import PredictionTableSpec


class PipelineAdapter:
    """Facade that ties together external loading, correction, and monitoring.

    Usage
    -----
    >>> adapter = PipelineAdapter()
    >>> adapter.register_external_source(ExternalPredictionSource(...))
    >>> adapter.register_correction(my_correction_module)
    >>> adapter.register_monitor(my_monitor_module)
    >>> result, report = adapter.run()
    """

    def __init__(
        self,
        correction_registry: Optional[CorrectionRegistry] = None,
        monitor_registry: Optional[MonitorRegistry] = None,
        spec: Optional[PredictionTableSpec] = None,
    ) -> None:
        self._correction_registry = correction_registry or CorrectionRegistry()
        self._monitor_registry = monitor_registry or MonitorRegistry()
        self._spec = spec or PredictionTableSpec()
        self._external_sources: list[ExternalPredictionSource] = []

    # ── Registration ────────────────────────────────────────────────

    def register_external_source(self, source: ExternalPredictionSource) -> None:
        """Register an external-model prediction CSV to be loaded at ``run()`` time."""
        self._external_sources.append(source)

    def register_correction(self, module: CorrectionModule) -> None:
        """Register a correction module."""
        self._correction_registry.register(module)

    def register_monitor(self, module: MonitorModule) -> None:
        """Register a monitor module."""
        self._monitor_registry.register(module)

    # ── Properties ──────────────────────────────────────────────────

    @property
    def correction_registry(self) -> CorrectionRegistry:
        return self._correction_registry

    @property
    def monitor_registry(self) -> MonitorRegistry:
        return self._monitor_registry

    @property
    def external_sources(self) -> list[ExternalPredictionSource]:
        return list(self._external_sources)

    # ── Execution ───────────────────────────────────────────────────

    def load_all_external(self) -> pd.DataFrame:
        """Load and concatenate all registered external prediction sources.

        Returns
        -------
        pd.DataFrame
            Single DataFrame containing all loaded predictions.
        """
        frames: list[pd.DataFrame] = []
        for source in self._external_sources:
            df = load_external_predictions(source, spec=self._spec)
            frames.append(df)

        if not frames:
            return pd.DataFrame(columns=list(self._spec.required_columns))

        return pd.concat(frames, ignore_index=True)

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        correction_kwargs: Optional[dict[str, Any]] = None,
        monitor_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Execute the full pipeline: load, correct, monitor.

        Parameters
        ----------
        df : pd.DataFrame or None
            If provided, use this as the base DataFrame instead of loading
            from registered external sources.
        correction_kwargs : dict or None
            Extra keyword arguments forwarded to each correction module.
        monitor_kwargs : dict or None
            Extra keyword arguments forwarded to each monitor module.

        Returns
        -------
        (pd.DataFrame, dict[str, Any])
            The (possibly corrected) prediction table and the aggregated
            monitor metrics report.
        """
        # 1. Load
        if df is None:
            df = self.load_all_external()

        if df.empty:
            return df, {}

        # 2. Correct
        corrected = self._correction_registry.run_all(
            df, **(correction_kwargs or {})
        )

        # 3. Monitor
        report = self._monitor_registry.run_all(
            corrected, **(monitor_kwargs or {})
        )

        return corrected, report

    # ── Data export helper ───────────────────────────────────────────

    def to_csv(
        self,
        df: pd.DataFrame,
        path: str | Path,
        **kwargs,
    ) -> Path:
        """Write a DataFrame to CSV (convenience wrapper).

        Parameters
        ----------
        df : pd.DataFrame
            Data to write.
        path : str | Path
            Destination file path.
        **kwargs
            Forwarded to ``pd.DataFrame.to_csv()``.

        Returns
        -------
        Path
            The resolved output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, **kwargs)
        return path
