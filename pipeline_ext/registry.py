# -*- coding: utf-8 -*-
"""Module registry for the plugin interface.

Provides a central registry to discover and retrieve:
  - PredictionProvider  (load predictions from external sources)
  - CorrectionModule     (post-process / correct predictions)
  - MonitorModule        (run quality or risk monitors)
"""

from __future__ import annotations

from typing import Any

from pipeline_ext.modules import (
    CorrectionModule,
    MonitorModule,
    PredictionProvider,
)

# ── Internal registries ────────────────────────────────────────────────
_PROVIDERS: dict[str, PredictionProvider] = {}
_CORRECTIONS: dict[str, CorrectionModule] = {}
_MONITORS: dict[str, MonitorModule] = {}

# ── Registration ───────────────────────────────────────────────────────


def register_prediction_provider(name: str, provider: PredictionProvider) -> None:
    """Register a prediction provider under *name*."""
    if not isinstance(provider, PredictionProvider):
        raise TypeError(f"Expected PredictionProvider, got {type(provider).__name__}")
    _PROVIDERS[name] = provider


def register_correction_module(name: str, module: CorrectionModule) -> None:
    """Register a correction module under *name*."""
    if not isinstance(module, CorrectionModule):
        raise TypeError(f"Expected CorrectionModule, got {type(module).__name__}")
    _CORRECTIONS[name] = module


def register_monitor_module(name: str, module: MonitorModule) -> None:
    """Register a monitor module under *name*."""
    if not isinstance(module, MonitorModule):
        raise TypeError(f"Expected MonitorModule, got {type(module).__name__}")
    _MONITORS[name] = module


# ── Retrieval ──────────────────────────────────────────────────────────


def get_module(name: str) -> PredictionProvider | CorrectionModule | MonitorModule:
    """Look up a module by name across all registry categories."""
    if name in _PROVIDERS:
        return _PROVIDERS[name]
    if name in _CORRECTIONS:
        return _CORRECTIONS[name]
    if name in _MONITORS:
        return _MONITORS[name]
    raise KeyError(
        f"Module '{name}' not found in any registry. "
        f"Providers: {sorted(_PROVIDERS)} | "
        f"Corrections: {sorted(_CORRECTIONS)} | "
        f"Monitors: {sorted(_MONITORS)}"
    )


# ── Listing ────────────────────────────────────────────────────────────


def list_modules() -> dict[str, list[str]]:
    """Return a dict mapping category → list of registered module names."""
    return {
        "providers": sorted(_PROVIDERS),
        "corrections": sorted(_CORRECTIONS),
        "monitors": sorted(_MONITORS),
    }


def list_providers() -> dict[str, PredictionProvider]:
    """Return a copy of the provider registry."""
    return dict(_PROVIDERS)


def list_corrections() -> dict[str, CorrectionModule]:
    """Return a copy of the correction registry."""
    return dict(_CORRECTIONS)


def list_monitors() -> dict[str, MonitorModule]:
    """Return a copy of the monitor registry."""
    return dict(_MONITORS)


# ── Reset (for testing) ────────────────────────────────────────────────


def _reset() -> None:
    """Clear all registries.  Intended for test teardown only."""
    _PROVIDERS.clear()
    _CORRECTIONS.clear()
    _MONITORS.clear()
