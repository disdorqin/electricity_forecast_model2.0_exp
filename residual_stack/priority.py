# -*- coding: utf-8 -*-
"""Priority rules for the residual stack.

Core rule — high_spike takes priority over negative:
    If high_spike correction was applied (lift > 0 or high_spike_prob high),
    the negative module MUST NOT apply downward correction.

This module provides stateless helper functions used by the orchestrator.
"""

from __future__ import annotations

import pandas as pd


def check_high_spike_priority(
    high_spike_prob: float,
    lift_applied: float = 0.0,
    spike_prob_threshold: float = 0.5,
) -> bool:
    """Return True if high_spike has priority and negative should be blocked.

    Parameters
    ----------
    high_spike_prob : float
        High-spike probability score (0-1).
    lift_applied : float
        Positive if high-spike lift was actually applied.
    spike_prob_threshold : float
        Threshold above which high-spike is considered active.

    Returns
    -------
    bool
        True → negative correction should NOT fire.
    """
    return (lift_applied > 0) or (high_spike_prob >= spike_prob_threshold)


def should_apply_negative(
    high_spike_prob: float,
    negative_risk: float,
    lift_applied: float = 0.0,
    spike_prob_threshold: float = 0.5,
    negative_risk_threshold: float = 0.4,
) -> tuple[bool, str]:
    """Decide whether negative correction should be applied.

    Returns
    -------
    (apply, reason) :
        apply  — True if negative correction should fire.
        reason — Explanation string.
    """
    if check_high_spike_priority(high_spike_prob, lift_applied, spike_prob_threshold):
        return False, "blocked_by_high_spike_priority"

    if negative_risk < negative_risk_threshold:
        return False, "negative_risk_below_threshold"

    return True, "negative_correction_allowed"


def format_module_sequence(
    high_spike_applied: bool,
    negative_applied: bool,
    guardrail_applied: bool = True,
) -> str:
    """Format the module sequence as a readable string.

    Examples::
        "high_spike→guardrail"
        "high_spike→negative→guardrail"
        "none"
    """
    modules: list[str] = []
    if high_spike_applied:
        modules.append("high_spike")
    if negative_applied:
        modules.append("negative")
    if guardrail_applied:
        modules.append("guardrail")

    if not modules:
        return "none"

    return "→".join(modules)
