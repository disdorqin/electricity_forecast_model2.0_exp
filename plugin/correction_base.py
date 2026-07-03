# -*- coding: utf-8 -*-
"""
correction_base.py — Abstract base class for correction modules.

Any concrete correction module (e.g. high-spike lift, negative-price guard,
midnight dip adjuster) implements this interface so the pipeline can
discover and run them dynamically without knowing their internal details.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class CorrectionModule(ABC):
    """Interface that every correction module must implement.

    Subclasses are identified by their ``name`` property, which must be
    unique within a registry.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable unique identifier for this correction module."""
        ...

    @abstractmethod
    def correct(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply this correction to the prediction DataFrame.

        The method should add or modify columns in-place or return a new
        DataFrame with the correction applied.  Typical added columns
        include ``{name}_corrected_pred`` and ``{name}_reason_code``.

        Parameters
        ----------
        df : pd.DataFrame
            Prediction table conforming to ``PredictionTableSpec``
            (plus any previously-applied corrections).
        **kwargs
            Module-specific parameters (e.g. profile configs).

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame with correction columns.
        """
        ...

    @abstractmethod
    def validate(self, df: pd.DataFrame) -> bool:
        """Check that *df* has the columns this correction requires.

        Parameters
        ----------
        df : pd.DataFrame
            Input prediction table.

        Returns
        -------
        bool
            True if *df* is valid for this correction, False otherwise.
        """
        ...
