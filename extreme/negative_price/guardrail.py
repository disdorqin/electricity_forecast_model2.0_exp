# -*- coding: utf-8 -*-
"""
guardrail.py — Safety guardrails for negative price / low valley correction.

Provides:
    - NegativeGuardrailConfig: Configuration for safety bounds
    - NegativeGuardrail: Evaluates whether downward correction is safe

Ensures:
    1. Mutual exclusion with high_spike correction (never both active)
    2. Max downward amount per period
    3. Min price floor (don't push predictions unreasonably low)
    4. Normal-hour protection (don't correct during non-valley periods)
    5. 9_16 spike-period protection (don't interfere with high-price spike module)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from extreme.negative_price.residual_correction import (
    PERIOD_DEFS,
    get_period,
    DOWNWARD_CORRECTION_APPLIED,
    DOWNWARD_CORRECTION_CAPPED,
)


@dataclass
class NegativeGuardrailConfig:
    """Safety bounds for negative price correction guardrail.

    Attributes:
        max_downward_ratio_9_16: Max downward as fraction of base_pred in 9_16.
        max_absolute_downward_9_16: Max absolute downward in 9_16.
        max_downward_ratio_1_8: Max downward ratio in 1_8.
        max_absolute_downward_1_8: Max absolute downward in 1_8.
        max_downward_ratio_17_24: Max downward ratio in 17_24.
        max_absolute_downward_17_24: Max absolute downward in 17_24.
        min_allowed_price: Absolute minimum allowed prediction price.
        spike_gate_active: If True, reject correction when high_spike prob > threshold.
        spike_prob_threshold: High-spike probability threshold for mutual exclusion.
    """
    max_downward_ratio_9_16: float = 0.10
    max_absolute_downward_9_16: float = 15.0
    max_downward_ratio_1_8: float = 0.25
    max_absolute_downward_1_8: float = 40.0
    max_downward_ratio_17_24: float = 0.20
    max_absolute_downward_17_24: float = 30.0
    min_allowed_price: float = -200.0
    spike_gate_active: bool = True
    spike_prob_threshold: float = 0.5


class GuardrailResult:
    """Result of a guardrail evaluation."""
    __slots__ = ("final_pred", "reason_code", "clipped_from")

    def __init__(
        self,
        final_pred: float,
        reason_code: str,
        clipped_from: Optional[float] = None,
    ):
        self.final_pred = final_pred
        self.reason_code = reason_code
        self.clipped_from = clipped_from


class NegativeGuardrail:
    """Safety guardrail for negative price correction.

    Pipeline:
        1. High-spike gate → reject if high_spike probability > threshold
        2. Per-period downward capping
        3. Absolute price floor
    """

    def __init__(self, config: Optional[NegativeGuardrailConfig] = None):
        self.config = config or NegativeGuardrailConfig()

    def evaluate(
        self,
        base_pred: float,
        corrected_pred: float,
        hour_business: int,
        spike_prob: float = 0.0,
    ) -> GuardrailResult:
        """Run the guardrail pipeline on a downward-corrected prediction.

        Args:
            base_pred: Prediction before any correction.
            corrected_pred: After downward correction (from residual_correction).
            hour_business: Business hour (1-24).
            spike_prob: High-spike probability (for mutual exclusion).

        Returns:
            GuardrailResult with final_pred and reason_code.
        """
        # 1. High-spike gate: reject downward if high_spike is active
        if self.config.spike_gate_active and spike_prob > self.config.spike_prob_threshold:
            return GuardrailResult(
                final_pred=base_pred,
                reason_code="GUARDRAIL_HIGH_SPIKE_GATE",
                clipped_from=corrected_pred,
            )

        period = get_period(hour_business)
        downward_amount = corrected_pred - base_pred

        # If no downward correction, pass through
        if downward_amount >= 0:
            return GuardrailResult(
                final_pred=corrected_pred,
                reason_code=DOWNWARD_CORRECTION_APPLIED,
            )

        # 2. Per-period downward capping
        if period == "9_16":
            ratio_cap = self.config.max_downward_ratio_9_16
            abs_cap = self.config.max_absolute_downward_9_16
        elif period == "1_8":
            ratio_cap = self.config.max_downward_ratio_1_8
            abs_cap = self.config.max_absolute_downward_1_8
        else:  # 17_24
            ratio_cap = self.config.max_downward_ratio_17_24
            abs_cap = self.config.max_absolute_downward_17_24

        max_downward_by_ratio = base_pred * ratio_cap
        max_allowed_downward = min(abs(downward_amount), abs(max_downward_by_ratio), abs_cap)
        max_allowed_downward = -max_allowed_downward  # make negative

        if downward_amount < max_allowed_downward:
            # Capped
            final_pred = base_pred + max_allowed_downward
            return GuardrailResult(
                final_pred=final_pred,
                reason_code="GUARDRAIL_CAPPED",
                clipped_from=corrected_pred,
            )

        # 3. Absolute price floor
        if corrected_pred < self.config.min_allowed_price:
            return GuardrailResult(
                final_pred=self.config.min_allowed_price,
                reason_code="GUARDRAIL_CAPPED",
                clipped_from=corrected_pred,
            )

        return GuardrailResult(
            final_pred=corrected_pred,
            reason_code=DOWNWARD_CORRECTION_APPLIED,
        )
