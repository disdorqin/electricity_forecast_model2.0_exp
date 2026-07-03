# -*- coding: utf-8 -*-
"""
residual_correction.py — Downward residual correction for negative price / low valley.

Provides:
    - NegativeResidualConfig: Configuration for downward correction
    - NegativeResidualCorrector: Computes downward lift amounts

Downward correction reduces (pulls down) the prediction when negative/low
price risk is high. This is the INVERSE of the high_spike module's upward lift.

Correction is bounded to prevent over-correction and normal-hour degradation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme.negative_price.schema import (
    NEGATIVE_PRICE_THRESHOLD,
)


# ── Reason codes ──────────────────────────────────────────────────────

DOWNWARD_NO_CORRECTION_LOW_RISK: str = "DOWNWARD_NO_CORRECTION_LOW_RISK"
DOWNWARD_NO_CORRECTION_ALREADY_LOW: str = "DOWNWARD_NO_CORRECTION_ALREADY_LOW"
DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE: str = "DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE"
DOWNWARD_CORRECTION_APPLIED: str = "DOWNWARD_CORRECTION_APPLIED"
DOWNWARD_CORRECTION_CAPPED: str = "DOWNWARD_CORRECTION_CAPPED"

# ── Period definitions (mirror high_spike for consistency) ────────────

PERIOD_DEFS: dict[str, range] = {
    "1_8": range(1, 9),
    "9_16": range(9, 17),
    "17_24": range(17, 25),
}


def get_period(hour_business: int) -> str:
    """Map hour_business (1-24) to period label."""
    if hour_business not in range(1, 25):
        raise ValueError(f"hour_business must be 1-24, got {hour_business}")
    for name, hr_range in PERIOD_DEFS.items():
        if hour_business in hr_range:
            return name
    return "17_24"


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class NegativeResidualConfig:
    """Configuration for downward residual correction.

    Attributes:
        risk_threshold: Minimum negative risk score to apply correction.
        max_downward_ratio: Max downward adjustment as fraction of base_pred.
        max_absolute_downward: Max absolute downward adjustment in price units.
        downward_quantile: Quantile of historical negative residual to use.
        min_pred_floor: Minimum allowed prediction after downward correction.
        period_aware: Apply per-period quantiles vs global.
        9_16_protection: Reduce downward correction during 9_16 (spike-prone).
        protect_normal_hours: Extra caution during non-valley periods.
        normal_hour_low_valley_risk_cap: Risk cap for normal-hour correction.
        mode: 'normal' or 'relaxed' (relaxed = offline diagnostic only).
    """
    risk_threshold: float = 0.3
    max_downward_ratio: float = 0.20
    max_absolute_downward: float = 30.0
    downward_quantile: float = 0.10
    min_pred_floor: float = -100.0
    period_aware: bool = True
    period_9_16_protection: bool = True
    protect_normal_hours: bool = True
    normal_hour_risk_cap: float = 0.5
    mode: str = "normal"

    def is_relaxed(self) -> bool:
        return self.mode == "relaxed"


# ── Result ────────────────────────────────────────────────────────────

@dataclass
class DownwardCorrectionResult:
    """Result of a downward correction computation."""
    corrected_pred: float
    downward_amount: float
    reason_code: str


# ── Corrector ─────────────────────────────────────────────────────────

class NegativeResidualCorrector:
    """Computes downward residual correction for negative/low price regimes.

    Fits historical quantiles of (y_true - y_pred) to determine how much
    to pull down predictions when negative risk is high.
    """

    def __init__(self, config: Optional[NegativeResidualConfig] = None):
        self.config = config or NegativeResidualConfig()
        self._downward_candidates: dict[str, float] = {}
        self._fitted: bool = False

    def fit_from_history(
        self,
        history: pd.DataFrame,
        pred_col: str = "base_fused_pred",
        y_true_col: str = "y_true",
    ) -> "NegativeResidualCorrector":
        """Fit per-period downward quantiles from historical residuals.

        Computes residual = y_true - y_pred (negative means overprediction).
        Uses a LOW quantile (e.g. 0.10) to find the typical overprediction
        amount during negative/low price regimes.

        Args:
            history: Historical DataFrame with y_true and pred_col.
            pred_col: Column name for predictions.
            y_true_col: Column name for actual values.

        Returns:
            Self for chaining.
        """
        df = history.copy()
        if "period" not in df.columns:
            df["period"] = df["hour_business"].apply(get_period)

        residual = df[y_true_col] - df[pred_col]

        if self.config.period_aware:
            candidates: dict[str, float] = {}
            for period_name in PERIOD_DEFS:
                mask = df["period"] == period_name
                period_residuals = residual[mask].dropna().values
                if len(period_residuals) == 0:
                    candidates[period_name] = 0.0
                else:
                    q = np.quantile(period_residuals, self.config.downward_quantile)
                    candidates[period_name] = min(0.0, float(q))  # negative or zero
            self._downward_candidates = candidates
        else:
            global_q = float(np.quantile(residual.dropna().values, self.config.downward_quantile))
            global_q = min(0.0, global_q)
            self._downward_candidates = {p: global_q for p in PERIOD_DEFS}

        self._fitted = True
        return self

    def set_downward_candidates(
        self, candidates: dict[str, float],
    ) -> "NegativeResidualCorrector":
        """Manually set downward correction candidates (bypass fitting)."""
        self._downward_candidates = {
            p: min(0.0, c) for p, c in candidates.items()
        }
        self._fitted = True
        return self

    def get_quantiles(self) -> dict[str, float]:
        """Return the fitted downward candidates per period."""
        if not self._downward_candidates:
            return {p: 0.0 for p in PERIOD_DEFS}
        return dict(self._downward_candidates)

    def compute_downward_correction(
        self,
        base_pred: float,
        negative_risk: float,
        low_valley_risk: float,
        hour_business: int,
        high_spike_active: bool = False,
    ) -> DownwardCorrectionResult:
        """Compute downward correction for a single prediction.

        Args:
            base_pred: The base prediction (before any correction).
            negative_risk: Probability of negative price (0-1).
            low_valley_risk: Probability of low valley (0-1).
            hour_business: Business hour (1-24).
            high_spike_active: Whether high_spike correction is active for this row.

        Returns:
            DownwardCorrectionResult with corrected_pred, downward_amount, reason.
        """
        period = get_period(hour_business)
        max_risk = max(negative_risk, low_valley_risk)

        # 0. Mutual exclusion gate: if high_spike is active, skip downward
        if high_spike_active:
            return DownwardCorrectionResult(
                corrected_pred=base_pred,
                downward_amount=0.0,
                reason_code=DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE,
            )

        # 1. Low risk → no correction
        if max_risk < self.config.risk_threshold:
            return DownwardCorrectionResult(
                corrected_pred=base_pred,
                downward_amount=0.0,
                reason_code=DOWNWARD_NO_CORRECTION_LOW_RISK,
            )

        # 2. Already low enough → no correction needed
        if base_pred <= NEGATIVE_PRICE_THRESHOLD:
            return DownwardCorrectionResult(
                corrected_pred=base_pred,
                downward_amount=0.0,
                reason_code=DOWNWARD_NO_CORRECTION_ALREADY_LOW,
            )

        # 3. 9_16 protection: reduce downward in spike-prone hours
        if self.config.period_9_16_protection and period == "9_16":
            if low_valley_risk < self.config.risk_threshold * 1.5:
                return DownwardCorrectionResult(
                    corrected_pred=base_pred,
                    downward_amount=0.0,
                    reason_code=DOWNWARD_NO_CORRECTION_LOW_RISK,
                )

        # 4. Normal hour protection
        if (
            self.config.protect_normal_hours
            and period != "9_16"
            and max_risk < self.config.normal_hour_risk_cap
        ):
            return DownwardCorrectionResult(
                corrected_pred=base_pred,
                downward_amount=0.0,
                reason_code=DOWNWARD_NO_CORRECTION_LOW_RISK,
            )

        # 5. Compute downward amount
        raw_downward = self._downward_candidates.get(period, 0.0)

        # In relax mode, boost downward for diagnostic purposes
        if self.config.is_relaxed():
            raw_downward = min(raw_downward, -10.0)

        # 6. Cap downward by ratio and absolute limits
        ratio_cap = -base_pred * self.config.max_downward_ratio
        capped_downward = max(raw_downward, ratio_cap, -self.config.max_absolute_downward)
        capped_downward = min(0.0, capped_downward)  # ensure negative or zero

        if capped_downward >= 0:
            return DownwardCorrectionResult(
                corrected_pred=base_pred,
                downward_amount=0.0,
                reason_code=DOWNWARD_NO_CORRECTION_LOW_RISK,
            )

        corrected = base_pred + capped_downward

        # 7. Floor at min_pred_floor
        if corrected < self.config.min_pred_floor:
            corrected = self.config.min_pred_floor
            capped_downward = corrected - base_pred

        reason = (
            DOWNWARD_CORRECTION_CAPPED
            if abs(capped_downward) < abs(raw_downward)
            else DOWNWARD_CORRECTION_APPLIED
        )

        return DownwardCorrectionResult(
            corrected_pred=corrected,
            downward_amount=capped_downward,
            reason_code=reason,
        )
