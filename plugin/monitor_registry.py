# -*- coding: utf-8 -*-
"""
monitor_registry.py — Dynamic registry for monitor modules.

Modules register themselves with a unique name.  The pipeline calls
``run_monitors(df, registry)`` which iterates over all registered
modules and collects their metrics into a single report.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from plugin.monitor_base import MonitorModule


class MonitorRegistry:
    """Holds a collection of named ``MonitorModule`` instances.

    Usage
    -----
    >>> registry = MonitorRegistry()
    >>> registry.register(my_monitor_module)
    >>> report = registry.run_all(df)
    """

    def __init__(self) -> None:
        self._modules: dict[str, MonitorModule] = {}

    # ── Registration ────────────────────────────────────────────────

    def register(self, module: MonitorModule) -> None:
        """Register a monitor module.

        Raises
        ------
        ValueError
            If a module with the same ``name`` is already registered.
        TypeError
            If *module* is not a ``MonitorModule`` instance.
        """
        if not isinstance(module, MonitorModule):
            raise TypeError(
                f"Expected MonitorModule, got {type(module).__name__}"
            )
        if module.name in self._modules:
            raise ValueError(
                f"Monitor module {module.name!r} is already registered"
            )
        self._modules[module.name] = module

    def unregister(self, name: str) -> None:
        """Remove a previously-registered module by name."""
        self._modules.pop(name, None)

    def get(self, name: str) -> Optional[MonitorModule]:
        """Retrieve a registered module by name, or None."""
        return self._modules.get(name)

    @property
    def names(self) -> list[str]:
        """Return the list of registered module names (in registration order)."""
        return list(self._modules.keys())

    def __len__(self) -> int:
        return len(self._modules)

    # ── Execution ───────────────────────────────────────────────────

    def run_all(self, df: pd.DataFrame, **kwargs) -> dict[str, Any]:
        """Run every registered monitor module and aggregate results.

        Parameters
        ----------
        df : pd.DataFrame
            Prediction table.
        **kwargs
            Keyword arguments forwarded to every module's ``monitor()``
            method.

        Returns
        -------
        dict[str, Any]
            Flattened metrics dictionary keyed by ``{name}.{metric_key}``.
        """
        report: dict[str, Any] = {}
        for module_name, module in self._modules.items():
            metrics = module.monitor(df, **kwargs)
            for key, value in metrics.items():
                report[f"{module_name}.{key}"] = value
        return report


# ── Standalone runner ───────────────────────────────────────────────────


def run_monitors(
    df: pd.DataFrame,
    registry: MonitorRegistry,
    **kwargs,
) -> dict[str, Any]:
    """Convenience function — equivalent to ``registry.run_all(df, **kwargs)``.

    This is the function the pipeline script calls so it never needs to
    import a concrete module.

    Parameters
    ----------
    df : pd.DataFrame
        Input prediction table.
    registry : MonitorRegistry
        Registry of monitor modules to run.
    **kwargs
        Forwarded to each module's ``monitor()`` method.

    Returns
    -------
    dict[str, Any]
        Aggregated metrics dictionary.
    """
    return registry.run_all(df, **kwargs)
