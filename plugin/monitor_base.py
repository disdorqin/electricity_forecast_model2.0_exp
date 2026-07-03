# -*- coding: utf-8 -*-
"""
monitor_base.py — Abstract base class for monitor modules.

Monitor modules run lightweight checks over the prediction DataFrame
(e.g. drift detection, outlier alerts, coverage gaps) and return a
dictionary of metrics.  They must not mutate the DataFrame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class MonitorModule(ABC):
    """Interface that every monitor module must implement.

    Subclasses are identified by their ``name`` property, which must be
    unique within a registry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable unique identifier for this monitor module."""
        ...

    @abstractmethod
    def monitor(self, df: pd.DataFrame, **kwargs) -> dict:
        """Run monitoring checks over the prediction DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Prediction table (may include correction columns).
        **kwargs
            Module-specific parameters (e.g. alert thresholds).

        Returns
        -------
        dict
            A flat dictionary of metric names → values.  The pipeline
            aggregates these into a single report.
        """
        ...
