# -*- coding: utf-8 -*-
"""
guardrail.py — Safety guardrails for realtime high-spike correction.

Provides:
  - GuardrailConfig: dataclass for tuning parameters
  - SpikeGuardrail: evaluates whether a corrected prediction should be kept, clipped, or rejected
  - Reason-code constants shared with residual_lift module
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from extreme.realtime_high_spike.residual_lift import PERIOD_DEFS, get_period

# ── Shared reason codes ──────────────────────────────────────────────
# These constants match those in ResidualLiftCorrector for consistency.

NO_CORRECTION_LOW_PROB: str = "NO_CORRECTION_LOW_PROB"
"""Spike probability too low → keep base_pred."""

NO_CORRECTION_NEGATIVE_BASE: str = "NO_CORRECTION_NEGATIVE_BASE"
"""Base prediction is negative → don't lift (negative-price module override)."""

NO_CORRECTION_NORMAL_HOUR: str = "NO_CORRECTION_NORMAL_HOUR"
"""Normal (non-spike-prone) hour with moderate probability → protect."""

LIFT_APPLIED: str = "LIFT_APPLIED"
"""Lift applied within bounds."""

GUARDRAIL_CLIPPED: str = "GUARDRAIL_CLIPPED"
"""Lift clipped by guardrail ratio/absolute cap."""


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class GuardrailConfig:
    """Safety bounds for spike correction guardrail.

    Attributes:
        min_prob_for_lift: Minimum spike probability required to allow lift.
        protect_normal_hours: If True, apply extra protection during normal hours.
        normal_hour_prob_cap: Spike probability cap for normal-hour protection.
        max_lift_ratio_9_16: Maximum lift as fraction of base_pred in 9_16 period.
        max_absolute_lift_9_16: Maximum absolute lift in 9_16 period.
        max_lift_ratio_1_8: Lift ratio cap for 1_8 period.
        max_absolute_lift_1_8: Absolute lift cap for 1_8 period.
        max_lift_ratio_17_24: Lift ratio cap for 17_24 period.
        max_absolute_lift_17_24: Absolute lift cap for 17_24 period.
        max_allowed_price: Maximum allowed final prediction price.
        min_allowed_price: Minimum allowed final prediction price.
        negative_base_guard: If True, reject corrections when base_pred is negative.
    """
    min_prob_for_lift: float = 0.5
    protect_normal_hours: bool = False
    normal_hour_prob_cap: float = 0.65

    # Per-period lift caps (9_16 is the spike-prone period)
    max_lift_ratio_9_16: float = 0.5
    max_absolute_lift_9_16: float = 500.0
    max_lift_ratio_1_8: float = 0.3
    max_absolute_lift_1_8: float = 300.0
    max_lift_ratio_17_24: float = 0.3
    max_absolute_lift_17_24: float = 300.0

    # Sanity bounds
    max_allowed_price: float = 1500.0
    min_allowed_price: float = -200.0

    # Negative base guard
    negative_base_guard: bool = True


# ── Guardrail result ─────────────────────────────────────────────────

class GuardrailResult:
    """Result of a guardrail evaluation."""

    __slots__ = ("final_pred", "reason_code", "clipped_from", "extra")

    def __init__(
        self,
        final_pred: float,
        reason_code: str,
        clipped_from: Optional[float] = None,
        extra: Optional[dict[str, Any]] = None,
    ):
        self.final_pred = final_pred
        self.reason_code = reason_code
        self.clipped_from = clipped_from
        self.extra = extra or {}


# ── Guardrail ─────────────────────────────────────────────────────────

class SpikeGuardrail:
    """Evaluates safety of spike-corrected predictions.

    Pipeline:
        1. Negative base guard → reject if base is negative
        2. Low probability check → reject if prob < min threshold
        3. Normal hour protection → reject for non-9_16 hours with moderate prob
        4. Lift capping → clip if corrected_pred exceeds ratio/absolute limits
        5. Sanity bounds → clip to [min_allowed, max_allowed]
    """

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()

    # ── Main evaluate ──────────────────────────────────────────────

    def evaluate(
        self,
        base_pred: float,
        spike_prob: float,
        corrected_pred: float,
        hour_business: int,
    ) -> GuardrailResult:
        """Run the full guardrail pipeline.

        Returns:
            GuardrailResult with final_pred and reason_code.
        """
        # 1. Negative base guard
        if self.config.negative_base_guard and base_pred < 0:
            return GuardrailResult(
                final_pred=base_pred,
                reason_code=NO_CORRECTION_NEGATIVE_BASE,
            )

        # 2. Low probability check
        if spike_prob < self.config.min_prob_for_lift:
            return GuardrailResult(
                final_pred=base_pred,
                reason_code=NO_CORRECTION_LOW_PROB,
            )

        period = get_period(hour_business)

        # 3. Normal hour protection
        if (
            self.config.protect_normal_hours
            and period != "9_16"
            and spike_prob < self.config.normal_hour_prob_cap
        ):
            return GuardrailResult(
                final_pred=base_pred,
                reason_code=NO_CORRECTION_NORMAL_HOUR,
            )

        # 4. Lift capping — per-period caps
        ratio_cap: float
        abs_cap: float
        if period == "9_16":
            ratio_cap = self.config.max_lift_ratio_9_16
            abs_cap = self.config.max_absolute_lift_9_16
        elif period == "1_8":
            ratio_cap = self.config.max_lift_ratio_1_8
            abs_cap = self.config.max_absolute_lift_1_8
        else:  # 17_24
            ratio_cap = self.config.max_lift_ratio_17_24
            abs_cap = self.config.max_absolute_lift_17_24

        lift_amount = corrected_pred - base_pred
        max_lift_by_ratio = base_pred * ratio_cap
        allowed_lift = min(lift_amount, max_lift_by_ratio, abs_cap)
        allowed_lift = max(0.0, allowed_lift)

        if lift_amount > allowed_lift:
            final_pred = base_pred + allowed_lift
            return GuardrailResult(
                final_pred=final_pred,
                reason_code=GUARDRAIL_CLIPPED,
                clipped_from=corrected_pred,
            )

        # 5. Sanity bounds
        final_pred = corrected_pred
        if final_pred > self.config.max_allowed_price:
            return GuardrailResult(
                final_pred=self.config.max_allowed_price,
                reason_code=GUARDRAIL_CLIPPED,
                clipped_from=corrected_pred,
            )
        if final_pred < self.config.min_allowed_price:
            return GuardrailResult(
                final_pred=self.config.min_allowed_price,
                reason_code=GUARDRAIL_CLIPPED,
                clipped_from=corrected_pred,
            )

        return GuardrailResult(
            final_pred=final_pred,
            reason_code=LIFT_APPLIED,
        )
