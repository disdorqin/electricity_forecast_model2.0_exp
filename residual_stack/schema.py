# -*- coding: utf-8 -*-
"""Schema and reason codes for the residual stack.

Defines the output columns that every stack-run must produce, reason-code
constants, and helper enums.
"""

from __future__ import annotations

from typing import Final

# ── Stack output columns (appended by the orchestrator) ────────────────

STACK_OUTPUT_COLUMNS: Final[list[str]] = [
    "base_pred",
    "after_high_spike_pred",
    "after_negative_pred",
    "final_pred",
    "high_spike_applied",
    "negative_applied",
    "correction_reason",
    "module_sequence",
]
"""
Every run of :class:`~residual_stack.orchestrator.ResidualStackOrchestrator`
produces these columns.

- base_pred             — Input prediction (typically base_fused_pred).
- after_high_spike_pred — After high-spike correction (may equal base_pred).
- after_negative_pred   — After negative/low-valley correction (may equal above).
- final_pred            — After final guardrail (the deployable prediction).
- high_spike_applied    — bool: was high-spike lift applied?
- negative_applied      — bool: was downward correction applied?
- correction_reason     — Human-readable summary of which corrections fired and why.
- module_sequence       — Ordered list of modules that ran, e.g. "high_spike→negative".
"""

# ── Reason codes ──────────────────────────────────────────────────────


class REASON_CODES:
    """Human-readable correction reason strings."""

    # No correction applied
    NONE: str = "no_correction"
    """No correction was needed or threshold not met."""

    # High-spike only
    SPIKE_LIFTED: str = "high_spike_lifted"
    """High-spike correction applied lift."""

    SPIKE_GUARDRAIL_CLIPPED: str = "high_spike_guardrail_clipped"
    """High-spike correction applied but clipped by guardrail."""

    # Negative only
    NEGATIVE_DOWN: str = "negative_downward"
    """Negative/low-valley downward correction applied."""

    NEGATIVE_GUARDRAIL_SKIPPED: str = "negative_blocked_by_guardrail"
    """Negative correction was candidate but blocked by guardrail."""

    # Both
    SPIKE_THEN_NEGATIVE: str = "spike_lifted_then_negative"
    """Both corrections applied (rare — only when spike is mild and negative risk high)."""

    # Priority
    SPIKE_BLOCKS_NEGATIVE: str = "spike_blocks_negative"
    """High-spike correction active — negative correction suppressed by priority rule."""

    # Error / edge
    DATA_LIMITED: str = "data_limited"
    """Insufficient data to evaluate or apply corrections."""


# ── Module sequence constants ─────────────────────────────────────────

MODULE_HIGH_SPIKE: Final[str] = "high_spike"
MODULE_NEGATIVE: Final[str] = "negative"
MODULE_GUARDRAIL: Final[str] = "guardrail"
