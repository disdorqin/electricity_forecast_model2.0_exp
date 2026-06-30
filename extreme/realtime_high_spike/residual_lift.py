# -*- coding: utf-8 -*-
"""
residual_lift.py — Compute residual lift candidates for realtime high-spike correction.

Provides:
  - ResidualLiftConfig: dataclass for tuning parameters
  - ResidualLiftCorrector: fits per-period lift quantiles from history and computes lift
  - get_period: hour → period label (1_8 / 9_16 / 17_24)
  - PERIOD_DEFS: period-to-hour-range mapping
"""

from __future__ import annotations

import enum
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# ── Correction Mode ──────────────────────────────────────────────────

class CorrectionMode(str, enum.Enum):
    """Correction mode flag — controls threshold strictness.

    Attributes:
        NORMAL:  Default production-safe mode (strict thresholds, normal-hour protection).
        RELAXED: Offline-only diagnostic mode that ensures lift fires on high-prob hours.
                 DO NOT use in production_pipeline.
    """
    NORMAL = "normal"
    RELAXED = "relaxed"

    def is_relaxed(self) -> bool:
        return self == self.RELAXED


def apply_correction_mode_threshold(
    base_threshold: float,
    mode: CorrectionMode,
    *,
    relaxed_multiplier: float = 0.6,
) -> float:
    """Adjust a threshold based on correction mode.

    In RELAXED mode the threshold is multiplied by *relaxed_multiplier*,
    making it easier for correction to fire.  In NORMAL mode the threshold
    is returned unchanged.
    """
    if mode.is_relaxed():
        return base_threshold * relaxed_multiplier
    return base_threshold


# ── Period definitions ────────────────────────────────────────────────

PERIOD_DEFS: dict[str, range] = {
    "1_8": range(1, 9),
    "9_16": range(9, 17),
    "17_24": range(17, 25),
}


def get_period(hour_business: int) -> str:
    """Map an hour (1-24) to its period label.

    Args:
        hour_business: Business hour (1-24).

    Returns:
        Period label: '1_8', '9_16', or '17_24'.

    Raises:
        ValueError: If hour is not in 1..24.
    """
    if hour_business not in range(1, 25):
        raise ValueError(f"hour_business must be 1-24, got {hour_business}")
    for period_name, hour_range in PERIOD_DEFS.items():
        if hour_business in hour_range:
            return period_name
    raise ValueError(f"Cannot map hour {hour_business} to any period")  # pragma: no cover


# ── Config ────────────────────────────────────────────────────────────

@dataclass
class ResidualLiftConfig:
    """Configuration for residual lift computation.

    Attributes:
        spike_prob_threshold: Minimum spike probability to consider applying lift.
        lift_quantile: Quantile of historical residual to use as lift candidate.
        period_aware: If True, fit per-period lift quantiles instead of global.
        protect_normal_hours: If True, apply extra guard during non-spike-prone periods.
        normal_hour_prob_cap: Spike probability cap for normal-hour protection.
        max_lift_ratio: Maximum lift as fraction of base_pred (0-1).
        max_absolute_lift: Maximum absolute lift in price units.
        period_9_16_boost: Multiplier on the 9_16 lift candidate.
        mode: CorrectionMode — RELAXED loosens thresholds to ensure lift fires on
              high-prob hours (offline-only, do NOT use in production).
        min_lift_floor: Minimum lift amount in price units when mode=RELAXED
                        (ensures lift > 0 even if fitted candidate is very small).
    """
    spike_prob_threshold: float = 0.5
    lift_quantile: float = 0.90
    period_aware: bool = True
    protect_normal_hours: bool = False
    normal_hour_prob_cap: float = 0.65
    max_lift_ratio: float = 0.5
    max_absolute_lift: float = 500.0
    period_9_16_boost: float = 1.0
    mode: CorrectionMode = CorrectionMode.NORMAL
    min_lift_floor: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.mode, str):
            self.mode = CorrectionMode(self.mode)


# ── Lift result ───────────────────────────────────────────────────────

class LiftResult:
    """Result of a single lift computation."""

    __slots__ = ("corrected_pred", "lift_applied", "reason_code", "extra")

    def __init__(
        self,
        corrected_pred: float,
        lift_applied: float,
        reason_code: str,
        extra: Optional[dict[str, Any]] = None,
    ):
        self.corrected_pred = corrected_pred
        self.lift_applied = lift_applied
        self.reason_code = reason_code
        self.extra = extra or {}


# ── Corrector ─────────────────────────────────────────────────────────

class ResidualLiftCorrector:
    """Computes residual lift candidates from historical prediction residuals.

    Usage:
        corrector = ResidualLiftCorrector(config)
        corrector.fit_from_history(history_df)
        result = corrector.compute_lift(base_pred=300.0, spike_prob=0.8, hour_business=12)
    """

    REASON_LOW_PROB: str = "NO_CORRECTION_LOW_PROB"
    REASON_APPLIED: str = "LIFT_APPLIED"
    REASON_CAPPED: str = "LIFT_CAPPED"
    REASON_NORMAL_HOUR: str = "NO_CORRECTION_NORMAL_HOUR"
    REASON_NEGATIVE_BASE: str = "NO_CORRECTION_NEGATIVE_BASE"

    def __init__(self, config: Optional[ResidualLiftConfig] = None):
        self.config = config or ResidualLiftConfig()
        self._lift_candidates: dict[str, float] = {}
        self._fitted: bool = False

    # ── Fitting ────────────────────────────────────────────────────

    def fit_from_history(self, history: pd.DataFrame) -> "ResidualLiftCorrector":
        """Fit per-period or global lift quantiles from historical data.

        Expected history columns:
            - business_day
            - hour_business
            - y_true
            - base_fused_pred
            - period (optional, inferred from hour_business if missing)
        """
        df = history.copy()
        if "period" not in df.columns:
            df["period"] = df["hour_business"].apply(get_period)

        # Compute residual = y_true - base_fused_pred
        residual = df["y_true"] - df["base_fused_pred"]

        if self.config.period_aware:
            # Per-period quantile
            candidates: dict[str, float] = {}
            for period_name in PERIOD_DEFS:
                mask = df["period"] == period_name
                if mask.sum() < 5:
                    warnings.warn(f"Period {period_name} has only {mask.sum()} samples; quantile may be unstable")
                period_residuals = residual[mask].dropna().values
                if len(period_residuals) == 0:
                    candidates[period_name] = 0.0
                else:
                    q = np.quantile(period_residuals, self.config.lift_quantile)
                    candidates[period_name] = max(0.0, float(q))
            self._lift_candidates = candidates
        else:
            # Global quantile
            global_q = float(np.quantile(residual.dropna().values, self.config.lift_quantile))
            global_q = max(0.0, global_q)
            self._lift_candidates = {p: global_q for p in PERIOD_DEFS}

        # Apply 9_16 boost
        self._lift_candidates["9_16"] *= self.config.period_9_16_boost

        self._fitted = True
        return self

    # ── Manual candidates ──────────────────────────────────────────

    def set_lift_candidates(self, candidates: dict[str, float]) -> "ResidualLiftCorrector":
        """Manually set lift candidates (bypass fitting)."""
        lift = dict(candidates)
        # Apply min_lift_floor in RELAXED mode
        if self.config.mode.is_relaxed() and self.config.min_lift_floor > 0:
            for p in lift:
                if lift[p] < self.config.min_lift_floor:
                    lift[p] = self.config.min_lift_floor
        self._lift_candidates = dict(lift)
        # Apply 9_16 boost when setting manually too
        self._lift_candidates["9_16"] = (
            lift.get("9_16", 0.0) * self.config.period_9_16_boost
        )
        self._fitted = True
        return self

    def get_quantiles(self) -> dict[str, float]:
        """Return the fitted lift candidates per period."""
        if not self._lift_candidates:
            return {p: 0.0 for p in PERIOD_DEFS}
        return dict(self._lift_candidates)

    def get_lift_candidate(self, period: str) -> float:
        """Return the lift candidate for a given period."""
        return self._lift_candidates.get(period, 0.0)

    # ── Compute lift ───────────────────────────────────────────────

    # ── Effective thresholds (mode-aware) ─────────────────────────

    def _effective_spike_threshold(self) -> float:
        """Return the spike-probability threshold adjusted for correction mode."""
        return apply_correction_mode_threshold(
            self.config.spike_prob_threshold, self.config.mode,
        )

    def _effective_normal_hour_cap(self) -> float:
        """Return the normal-hour protection cap adjusted for correction mode.

        In RELAXED mode this is set to 0.0 so normal-hour protection
        (spike_prob < cap) is always False, effectively disabling it.
        """
        if self.config.mode.is_relaxed():
            return 0.0  # disables normal-hour protection (spike_prob < 0 is never true)
        return self.config.normal_hour_prob_cap

    def _effective_ratio_cap(self) -> float:
        """Return the max lift ratio cap adjusted for correction mode.

        RELAXED mode doubles the ratio to make sure lift can fire.
        """
        if self.config.mode.is_relaxed():
            return max(self.config.max_lift_ratio, 0.5)
        return self.config.max_lift_ratio

    def _effective_abs_cap(self) -> float:
        """Return the max absolute lift cap adjusted for correction mode."""
        if self.config.mode.is_relaxed():
            return max(self.config.max_absolute_lift, 500.0)
        return self.config.max_absolute_lift

    # ── Compute lift ───────────────────────────────────────────────

    def compute_lift(
        self,
        base_pred: float,
        spike_prob: float,
        hour_business: int,
    ) -> LiftResult:
        """Compute the lift amount for a single hour.

        Args:
            base_pred: The base fused prediction.
            spike_prob: Predicted spike probability (0-1).
            hour_business: Business hour (1-24).

        Returns:
            LiftResult with corrected_pred, lift_applied, reason_code.
        """
        period = get_period(hour_business)
        eff_threshold = self._effective_spike_threshold()

        # 1. Low probability → no correction
        if spike_prob < eff_threshold:
            return LiftResult(
                corrected_pred=base_pred,
                lift_applied=0.0,
                reason_code=self.REASON_LOW_PROB,
            )

        # 2. Normal hour protection (if enabled and period is not 9_16)
        eff_normal_cap = self._effective_normal_hour_cap()
        if (
            self.config.protect_normal_hours
            and period != "9_16"
            and spike_prob < eff_normal_cap
        ):
            return LiftResult(
                corrected_pred=base_pred,
                lift_applied=0.0,
                reason_code=self.REASON_NORMAL_HOUR,
            )

        # 3. Compute target lift from candidate
        raw_lift = self._lift_candidates.get(period, 0.0)

        # 3b. In RELAXED mode, enforce a minimum lift floor so correction
        #     always fires on high-prob hours even if the fitted candidate
        #     is near zero.
        if self.config.mode.is_relaxed() and self.config.min_lift_floor > 0:
            raw_lift = max(raw_lift, self.config.min_lift_floor)

        # 4. Cap lift by ratio and absolute limits
        eff_ratio_cap = self._effective_ratio_cap()
        eff_abs_cap = self._effective_abs_cap()
        ratio_cap = base_pred * eff_ratio_cap
        capped_lift = min(raw_lift, ratio_cap, eff_abs_cap)
        capped_lift = max(0.0, capped_lift)

        if capped_lift <= 0.0:
            return LiftResult(
                corrected_pred=base_pred,
                lift_applied=0.0,
                reason_code=self.REASON_LOW_PROB,
            )

        corrected_pred = base_pred + capped_lift

        if capped_lift < raw_lift:
            return LiftResult(
                corrected_pred=corrected_pred,
                lift_applied=capped_lift,
                reason_code=self.REASON_CAPPED,
            )

        return LiftResult(
            corrected_pred=corrected_pred,
            lift_applied=capped_lift,
            reason_code=self.REASON_APPLIED,
        )
