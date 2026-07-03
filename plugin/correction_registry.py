# -*- coding: utf-8 -*-
"""
correction_registry.py — Dynamic registry for correction modules.

Modules register themselves with a unique name.  The pipeline calls
``run_corrections(df, registry)`` which iterates over all registered
modules and applies them in registration order.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from plugin.correction_base import CorrectionModule


class CorrectionRegistry:
    """Holds a collection of named ``CorrectionModule`` instances.

    Usage
    -----
    >>> registry = CorrectionRegistry()
    >>> registry.register(my_correction_module)
    >>> result = registry.run_all(df)
    """

    def __init__(self) -> None:
        self._modules: dict[str, CorrectionModule] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(self, module: CorrectionModule) -> None:
        """Register a correction module.

        Raises
        ------
        ValueError
            If a module with the same ``name`` is already registered.
        """
        if not isinstance(module, CorrectionModule):
            raise TypeError(
                f"Expected CorrectionModule, got {type(module).__name__}"
            )
        if module.name in self._modules:
            raise ValueError(
                f"Correction module {module.name!r} is already registered"
            )
        self._modules[module.name] = module

    def unregister(self, name: str) -> None:
        """Remove a previously-registered module by name."""
        self._modules.pop(name, None)

    def get(self, name: str) -> Optional[CorrectionModule]:
        """Retrieve a registered module by name, or None."""
        return self._modules.get(name)

    @property
    def names(self) -> list[str]:
        """Return the list of registered module names (in registration order)."""
        return list(self._modules.keys())

    def __len__(self) -> int:
        return len(self._modules)

    # ── Execution ───────────────────────────────────────────────────

    def run_all(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply every registered correction module in-order.

        Each module receives the DataFrame produced by the previous module.

        Parameters
        ----------
        df : pd.DataFrame
            Input prediction table.
        **kwargs
            Keyword arguments forwarded to every module's ``correct()``
            method.

        Returns
        -------
        pd.DataFrame
            Augmented DataFrame after all corrections have been applied.
        """
        result = df.copy()
        for name, module in self._modules.items():
            result = module.correct(result, **kwargs)
        return result

    def validate_all(self, df: pd.DataFrame) -> dict[str, bool]:
        """Run ``validate()`` on every registered module.

        Returns
        -------
        dict[str, bool]
            Mapping of module name → validation result.
        """
        return {name: mod.validate(df) for name, mod in self._modules.items()}


# ── Standalone runner ───────────────────────────────────────────────────


def run_corrections(
    df: pd.DataFrame,
    registry: CorrectionRegistry,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function — equivalent to ``registry.run_all(df, **kwargs)``.

    This is the function the pipeline script calls so it never needs to
    import a concrete module.

    Parameters
    ----------
    df : pd.DataFrame
        Input prediction table.
    registry : CorrectionRegistry
        Registry of correction modules to apply.
    **kwargs
        Forwarded to each module's ``correct()`` method.

    Returns
    -------
    pd.DataFrame
        Corrected DataFrame.
    """
    return registry.run_all(df, **kwargs)
