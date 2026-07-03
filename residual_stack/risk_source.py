# -*- coding: utf-8 -*-
"""Risk source classification for the residual stack.

Determines whether spike risk probability data is legitimate for
official evaluation or only for dry-run testing.

Risk source hierarchy::

    real_prob          — Real model output (probability from spike risk model)
                         → eligible for official GO/NO-GO
    calibrated_prob    — Calibrated probability (post-hoc scaling)
                         → eligible for official GO/NO-GO
    synthetic_flag     — Synthetic probability derived from a binary flag
                         (e.g. high_spike_flag → 0.9) → dry-run only
    missing            — No spike risk data available
                         → DATA-MISSING for configs B/D
"""

from __future__ import annotations

import enum
from typing import Optional

import pandas as pd


class RiskSource(str, enum.Enum):
    """Classification of the spike risk data source."""

    REAL_PROB = "real_prob"
    """Real spike probability from a trained risk model."""

    CALIBRATED_PROB = "calibrated_prob"
    """Post-hoc calibrated probability (e.g. isotonic regression)."""

    SYNTHETIC_FLAG = "synthetic_flag"
    """Synthetic probability derived from a binary flag — dry-run only."""

    MISSING = "missing"
    """No spike risk data available at all."""


# ── Detection ──────────────────────────────────────────────────────────


def detect_risk_source(
    spike_risk_path: Optional[str],
    df: Optional[pd.DataFrame] = None,
) -> RiskSource:
    """Determine the risk source from available data.

    Parameters
    ----------
    spike_risk_path : str | None
        Path to a spike risk CSV. If set, assumed to contain real
        high_spike_prob (since it comes from the spike risk model pipeline).
    df : pd.DataFrame | None
        Optional prediction DataFrame to inspect for ``high_spike_prob``
        or ``high_spike_flag`` columns (as fallback).

    Returns
    -------
    RiskSource
        Classification based on available data.
    """
    # Explicit risk CSV path == real probability (from spike risk pipeline)
    if spike_risk_path is not None:
        return RiskSource.REAL_PROB

    if df is not None:
        # Check for real probability column in the pack
        if "high_spike_prob" in df.columns:
            return RiskSource.CALIBRATED_PROB

        # Binary flag only — would need synthetic conversion
        if "high_spike_flag" in df.columns:
            return RiskSource.SYNTHETIC_FLAG

    return RiskSource.MISSING


# ── Policy ─────────────────────────────────────────────────────────────


def is_official_source(source: RiskSource) -> bool:
    """Return True if *source* is eligible for official GO/NO-GO verdict."""
    return source in (RiskSource.REAL_PROB, RiskSource.CALIBRATED_PROB)


def is_synthetic_source(source: RiskSource) -> bool:
    """Return True if *source* is synthetic (dry-run only)."""
    return source == RiskSource.SYNTHETIC_FLAG


def is_missing_source(source: RiskSource) -> bool:
    """Return True if *source* is missing (no risk data)."""
    return source == RiskSource.MISSING


def resolve_risk_policy(
    source: RiskSource,
    allow_synthetic: bool = False,
) -> tuple[bool, str]:
    """Resolve whether a config can produce an official verdict.

    Parameters
    ----------
    source : RiskSource
        Detected risk data source.
    allow_synthetic : bool
        If True, synthetic risk is permitted for evaluation
        (as dry-run, not official GO/NO-GO).

    Returns
    -------
    (can_run, status_label)
        can_run   — True if the config can execute.
        status_label — One of ``official``, ``dry_run``, ``data_missing``.
    """
    if is_official_source(source):
        return True, "official"

    if is_synthetic_source(source):
        if allow_synthetic:
            return True, "dry_run"
        return False, "data_missing"

    # MISSING
    return False, "data_missing"


def format_risk_verdict(run_status: str, verdict: str) -> str:
    """Format a verdict string with its run status prefix.

    Examples
    --------
        "[official] GO"
        "[dry-run] NO-GO: ..."
        "[data-missing] n/a"
    """
    if run_status == "data_missing":
        return "[data-missing] n/a"
    return f"[{run_status}] {verdict}"
