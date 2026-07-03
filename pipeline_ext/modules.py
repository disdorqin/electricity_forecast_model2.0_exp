# -*- coding: utf-8 -*-
"""Base classes for the plugin interface.

Defines three abstract module types:

    PredictionProvider
        Loads predictions from an external source and returns a
        DataFrame conforming to the unified schema.

    CorrectionModule
        Applies a deterministic correction to a prediction DataFrame.
        Corrections are applied in registration order.

    MonitorModule
        Analyses a prediction DataFrame and returns a dict of metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class PredictionProvider(ABC):
    """Interface for loading predictions from an external source."""

    @abstractmethod
    def load_predictions(self, path: str) -> pd.DataFrame:
        """Load predictions from *path* and return a schema-conformant DataFrame.

        The returned DataFrame must contain all REQUIRED_FIELDS defined
        in ``pipeline_ext.schema``.
        """
        ...


class CorrectionModule(ABC):
    """Interface for a prediction correction step.

    Subclasses must set *name* to a unique identifier.
    """

    name: str

    @abstractmethod
    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the correction to *df* and return the modified DataFrame.

        The correction should be deterministic and idempotent where
        possible.
        """
        ...


class MonitorModule(ABC):
    """Interface for a monitoring / analysis step.

    Subclasses must set *name* to a unique identifier.
    """

    name: str

    @abstractmethod
    def run(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyse *df* and return a dictionary of metrics.

        The returned dict should be JSON-serialisable.
        """
        ...
