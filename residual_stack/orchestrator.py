# -*- coding: utf-8 -*-
"""ResidualStackOrchestrator — Unified high-spike + negative correction runner.

Wires together:

    extreme.realtime_high_spike
    extreme.negative_price

into a single pipeline with guaranteed priority ordering (high_spike > negative)
and full correction traceability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme.realtime_high_spike.apply_correction import (
    CorrectionMode,
    CorrectionProfile as SpikeProfile,
)
from extreme.negative_price.apply_negative_correction import (
    NegativeCorrectionProfile,
    apply_negative_correction,
)
from extreme.negative_price.residual_correction import NegativeResidualCorrector
from extreme.negative_price.risk_model import NegativeRiskModel

from residual_stack.risk_source import (
    RiskSource,
    detect_risk_source,
    resolve_risk_policy,
)
from residual_stack.priority import (
    check_high_spike_priority,
    format_module_sequence,
    should_apply_negative,
)
from residual_stack.schema import (
    MODULE_GUARDRAIL,
    MODULE_HIGH_SPIKE,
    MODULE_NEGATIVE,
    REASON_CODES,
    STACK_OUTPUT_COLUMNS,
)

logger = logging.getLogger(__name__)


# ── Profile bundle ─────────────────────────────────────────────────────


@dataclass
class StackProfile:
    """Combined profile for the entire residual stack.

    Attributes
    ----------
    name : str
        Human-readable name for this combination (e.g. "medium+conservative").
    spike_profile_name : str
        Profile name for high-spike correction ('conservative' / 'medium' / 'aggressive').
    negative_profile_name : str
        Profile name for negative correction ('conservative' / 'moderate' / 'aggressive').
    spike_mode : str
        CorrectionMode for high-spike ('normal' or 'relaxed').
    """
    name: str = "default"
    spike_profile_name: str = "medium"
    negative_profile_name: str = "conservative"
    spike_mode: str = "normal"


# ── Result container ───────────────────────────────────────────────────


@dataclass
class StackResult:
    """Result of a residual stack run."""

    df: pd.DataFrame
    """Corrected DataFrame with STACK_OUTPUT_COLUMNS appended."""

    profile_used: StackProfile
    """Profile combo that was used."""

    metrics: dict[str, Any] = field(default_factory=dict)
    """Computed metrics (populated by :func:`~residual_stack.metrics.compute_stack_metrics`)."""

    data_limited: bool = False
    """True if negative sample count was too low for reliable evaluation."""

    risk_source: RiskSource = RiskSource.MISSING
    """Detected risk source for spike data."""

    run_status: str = "data_missing"
    """Policy resolution status: official / dry_run / data_missing."""


# ── Orchestrator ───────────────────────────────────────────────────────


class ResidualStackOrchestrator:
    """Orchestrate the high-spike → negative → guardrail correction pipeline.

    Usage::

        orch = ResidualStackOrchestrator()
        result = orch.run(
            prediction_pack_path="outputs/pack.csv",
            spike_risk_path="outputs/risk.csv",
            profile=StackProfile(),
        )
        print(result.metrics)
    """

    def __init__(
        self,
        pred_col: str = "base_fused_pred",
        spike_prob_col: str = "high_spike_prob",
    ):
        self.pred_col = pred_col
        self.spike_prob_col = spike_prob_col

    def run(
        self,
        prediction_pack_path: str | Path,
        spike_risk_path: str | Path | None = None,
        profile: Optional[StackProfile] = None,
        history_df: Optional[pd.DataFrame] = None,
        risk_model: Optional[NegativeRiskModel] = None,
    ) -> StackResult:
        """Run the full residual stack.

        Parameters
        ----------
        prediction_pack_path : str | Path
            Path to prediction pack CSV (must contain base_fused_pred,
            business_day, hour_business, y_true at minimum).
        spike_risk_path : str | Path | None
            Path to spike risk predictions CSV (must contain high_spike_prob).
            If None, high_spike correction is skipped.
        profile : StackProfile | None
            Combined profile. Defaults to medium+conservative.
        history_df : pd.DataFrame | None
            Historical data for fitting lift/downward quantiles.
        risk_model : NegativeRiskModel | None
            Pre-fitted negative risk model.

        Returns
        -------
        StackResult
            Corrected DataFrame with structured metadata, including
            ``risk_source`` and ``run_status`` fields for policy evaluation.
        """
        profile = profile or StackProfile()
        logger.info(
            "Residual stack starting — spike=%s negative=%s",
            profile.spike_profile_name,
            profile.negative_profile_name,
        )

        # ── Load prediction pack ────────────────────────────────────
        df = pd.read_csv(prediction_pack_path)
        df = self._deduplicate_by_time(df)
        raw_len = len(df)

        if self.pred_col not in df.columns:
            raise ValueError(
                f"Prediction column '{self.pred_col}' not found in pack. "
                f"Available: {sorted(df.columns)}"
            )

        # ── Detect risk source from available data ──────────────────
        risk_source = detect_risk_source(
            spike_risk_path=str(spike_risk_path) if spike_risk_path else None,
            df=df,
        )
        _, run_status = resolve_risk_policy(risk_source)
        logger.info(
            "Risk source: %s → %s", risk_source.value, run_status,
        )

        # ── Step 1: High-spike correction ───────────────────────────
        if spike_risk_path is not None:
            spike_profile = self._build_spike_profile(profile)
            df = self._apply_high_spike_correction(
                df, spike_risk_path, spike_profile, history_df,
            )
        else:
            # No spike correction — initialise columns
            df["after_high_spike_pred"] = df[self.pred_col].values
            df["high_spike_applied"] = False
            df["spike_corrected_pred"] = df[self.pred_col].values
            df["reason_code"] = REASON_CODES.NONE

        # ── Step 2: Negative / low-valley correction ────────────────
        df = self._apply_negative_correction(
            df, profile, history_df, risk_model,
        )

        # ── Step 3: Final guardrail and output columns ──────────────
        df = self._apply_final_guardrail(df, profile)

        # ── Verify output columns ───────────────────────────────────
        missing = [c for c in STACK_OUTPUT_COLUMNS if c not in df.columns]
        if missing:
            raise RuntimeError(f"Stack output columns missing: {missing}")

        logger.info("Residual stack done — %d rows processed", raw_len)

        result = StackResult(
            df=df, profile_used=profile,
            risk_source=risk_source, run_status=run_status,
        )
        return result

    # ── Internal: high-spike step ──────────────────────────────────────

    def _build_spike_profile(self, stack_profile: StackProfile) -> SpikeProfile:
        return SpikeProfile(
            name=stack_profile.spike_profile_name,
            mode=CorrectionMode(stack_profile.spike_mode),
        )

    def _apply_high_spike_correction(
        self,
        df: pd.DataFrame,
        spike_risk_path: str | Path,
        spike_profile: SpikeProfile,
        history_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Apply high-spike correction and record intermediate state."""
        logger.info("High-spike correction applied via spike risk: %s", spike_risk_path)

        # Merge spike results onto base df
        spike_result = pd.read_csv(spike_risk_path)
        for col in ("business_day",):
            if col in spike_result.columns:
                spike_result[col] = spike_result[col].astype(str)
            if col in df.columns:
                df[col] = df[col].astype(str)

        merged = pd.merge(
            df, spike_result,
            on=["business_day", "hour_business"],
            how="left",
            suffixes=("", "_risk"),
        )

        # Use spike risk module directly for row-by-row correction
        from extreme.realtime_high_spike.residual_lift import (
            ResidualLiftConfig,
            ResidualLiftCorrector,
            get_period,
        )
        from extreme.realtime_high_spike.guardrail import (
            GuardrailConfig,
            SpikeGuardrail,
        )

        lcfg = spike_profile.to_lift_config()
        gcfg = spike_profile.to_guardrail_config()

        corrector = ResidualLiftCorrector(lcfg)
        if history_df is not None and not history_df.empty:
            corrector.fit_from_history(history_df)
        else:
            corrector.set_lift_candidates({p: 50.0 for p in ["1_8", "9_16", "17_24"]})
            corrector._lift_candidates["9_16"] *= lcfg.period_9_16_boost

        guardrail = SpikeGuardrail(gcfg)

        spike_list: list[float] = []
        final_list: list[float] = []
        reason_list: list[str] = []
        lift_list: list[float] = []

        for _, row in merged.iterrows():
            base_pred = row.get(self.pred_col, 0.0)
            spike_prob = row.get(self.spike_prob_col, 0.0)
            hour_business = row.get("hour_business", 12)

            if pd.isna(base_pred):
                base_pred = 0.0
            if pd.isna(spike_prob):
                spike_prob = 0.0

            lift_result = corrector.compute_lift(
                base_pred=float(base_pred),
                spike_prob=float(spike_prob),
                hour_business=int(hour_business),
            )
            corrected = lift_result.corrected_pred
            guard_result = guardrail.evaluate(
                base_pred=float(base_pred),
                spike_prob=float(spike_prob),
                corrected_pred=corrected,
                hour_business=int(hour_business),
            )

            spike_list.append(corrected)
            final_list.append(guard_result.final_pred)
            reason_list.append(guard_result.reason_code)
            lift_list.append(guard_result.final_pred - float(base_pred))

        merged["after_high_spike_pred"] = final_list
        merged["high_spike_applied"] = [l > 0 for l in lift_list]
        merged["spike_corrected_pred"] = spike_list
        merged["spike_reason_code"] = reason_list
        merged["lift_applied"] = lift_list
        merged[self.spike_prob_col] = merged.get(self.spike_prob_col, 0.0)

        return merged

    # ── Internal: negative correction step ─────────────────────────────

    def _apply_negative_correction(
        self,
        df: pd.DataFrame,
        profile: StackProfile,
        history_df: pd.DataFrame | None,
        risk_model: NegativeRiskModel | None,
    ) -> pd.DataFrame:
        """Apply negative correction (downward only, respecting high_spike priority)."""
        neg_profile = NegativeCorrectionProfile(name=profile.negative_profile_name)

        from extreme.negative_price.guardrail import (
            NegativeGuardrail,
            NegativeGuardrailConfig,
        )

        corrector = NegativeResidualCorrector(neg_profile.to_residual_config())
        if history_df is not None and not history_df.empty:
            corrector.fit_from_history(history_df, pred_col=self.pred_col)
        else:
            corrector.set_downward_candidates({
                "1_8": -15.0, "9_16": -5.0, "17_24": -10.0,
            })

        guardrail_cfg = NegativeGuardrailConfig(
            spike_gate_active=True,
            spike_prob_threshold=0.5,
        )
        guardrail = NegativeGuardrail(guardrail_cfg)

        after_neg_list: list[float] = []
        neg_applied_list: list[bool] = []
        neg_reason_list: list[str] = []
        module_seq_list: list[str] = []

        for _, row in df.iterrows():
            base_after_spike = row.get("after_high_spike_pred", 0.0)
            spike_prob = row.get(self.spike_prob_col, 0.0)
            hour_business = row.get("hour_business", 12)
            spike_applied = row.get("high_spike_applied", False)
            lift_applied = row.get("lift_applied", 0.0)

            if pd.isna(base_after_spike):
                base_after_spike = 0.0
            if pd.isna(spike_prob):
                spike_prob = 0.0

            # Priority check: high_spike blocks negative
            spike_active = bool(spike_applied) or check_high_spike_priority(
                high_spike_prob=float(spike_prob),
                lift_applied=float(lift_applied) if not pd.isna(lift_applied) else 0.0,
                spike_prob_threshold=0.5,
            )

            if spike_active:
                after_neg_list.append(float(base_after_spike))
                neg_applied_list.append(False)
                neg_reason_list.append(REASON_CODES.SPIKE_BLOCKS_NEGATIVE)
                module_seq_list.append(
                    format_module_sequence(
                        high_spike_applied=bool(spike_applied),
                        negative_applied=False,
                    )
                )
                continue

            # Try downward correction
            correction_result = corrector.compute_downward_correction(
                base_pred=float(base_after_spike),
                negative_risk=0.0,  # will use heuristic if no risk_model
                low_valley_risk=0.0,
                hour_business=int(hour_business),
                high_spike_active=False,
            )

            guard_result = guardrail.evaluate(
                base_pred=float(base_after_spike),
                corrected_pred=correction_result.corrected_pred,
                hour_business=int(hour_business),
                spike_prob=float(spike_prob),
            )

            after_neg = guard_result.final_pred
            actual_down = after_neg < float(base_after_spike)

            after_neg_list.append(after_neg)
            neg_applied_list.append(actual_down)

            if actual_down:
                neg_reason_list.append(REASON_CODES.NEGATIVE_DOWN)
            else:
                neg_reason_list.append(REASON_CODES.NEGATIVE_GUARDRAIL_SKIPPED)

            module_seq_list.append(
                format_module_sequence(
                    high_spike_applied=bool(spike_applied),
                    negative_applied=actual_down,
                )
            )

        df["after_negative_pred"] = after_neg_list
        df["negative_applied"] = neg_applied_list
        df["negative_reason_code"] = neg_reason_list
        df["module_sequence"] = module_seq_list

        return df

    # ── Internal: final guardrail and output assembly ──────────────────

    def _apply_final_guardrail(
        self,
        df: pd.DataFrame,
        profile: StackProfile,
    ) -> pd.DataFrame:
        """Apply the final guardrail and assemble canonical output columns.

        The final guardrail ensures:
        1. High-spike hours are not harmed by downward correction.
        2. Normal-hour degradation is bounded.
        3. All output columns are populated.
        """
        after_neg = df["after_negative_pred"].values
        after_spike = df["after_high_spike_pred"].values
        spike_applied = df["high_spike_applied"].values
        neg_applied = df["negative_applied"].values

        # Final pred: after_negative_pred (which may be after_spike if neg blocked)
        final_pred = after_neg.copy()

        # Guardrail 1: If high_spike was applied, never let final < after_spike
        spike_mask = spike_applied.astype(bool)
        final_pred[spike_mask] = np.maximum(
            final_pred[spike_mask], after_spike[spike_mask]
        )

        # Guardrail 2: Normal hours — limit degradation to 0.5 sMAPE equivalent
        # Use simple cap: final_pred should not be more than 20% lower than after_spike
        normal_hours = ~df["hour_business"].between(9, 16).values
        pred_after_spike = after_spike.copy()
        pred_after_spike[pred_after_spike == 0] = 1.0  # avoid div-by-zero
        degradation = (pred_after_spike - final_pred) / np.abs(pred_after_spike)
        over_degraded = normal_hours & (degradation > 0.20)
        final_pred[over_degraded] = after_spike[over_degraded] * 0.80

        df["final_pred"] = final_pred

        # ── Build correction_reason column ──────────────────────────
        reasons: list[str] = []
        for i in range(len(df)):
            parts: list[str] = []
            if spike_applied[i]:
                parts.append(REASON_CODES.SPIKE_LIFTED)
            if neg_applied[i]:
                parts.append(REASON_CODES.NEGATIVE_DOWN)
            if not spike_applied[i] and not neg_applied[i]:
                # Check if negative was blocked
                if df["negative_reason_code"].iloc[i] == REASON_CODES.SPIKE_BLOCKS_NEGATIVE:
                    parts.append(REASON_CODES.SPIKE_BLOCKS_NEGATIVE)
                else:
                    parts.append(REASON_CODES.NONE)
            reasons.append(" | ".join(parts))
        df["correction_reason"] = reasons

        # ── Ensure base_pred column ─────────────────────────────────
        if self.pred_col in df.columns:
            df["base_pred"] = df[self.pred_col].values
        else:
            df["base_pred"] = after_spike

        return df

    # ── Internal: deduplication ────────────────────────────────────────

    @staticmethod
    def _deduplicate_by_time(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate (business_day, hour_business) rows, keeping last."""
        keys = ["business_day", "hour_business"]
        present = [c for c in keys if c in df.columns]
        if len(present) < 2:
            return df
        before = len(df)
        df = df.drop_duplicates(subset=present, keep="last")
        if len(df) < before:
            logger.warning("Dropped %d duplicate time rows", before - len(df))
        return df


# ── Convenience entry-point ────────────────────────────────────────────


def run_residual_stack(
    prediction_pack_path: str | Path,
    spike_risk_path: str | Path | None = None,
    spike_profile: str = "medium",
    negative_profile: str = "conservative",
    history_df: Optional[pd.DataFrame] = None,
    out_dir: str | Path | None = None,
) -> StackResult:
    """One-shot convenience wrapper around :class:`ResidualStackOrchestrator`.

    Parameters
    ----------
    prediction_pack_path, spike_risk_path : as in :meth:`ResidualStackOrchestrator.run`.
    spike_profile : str
        Profile name for high-spike ('conservative' / 'medium' / 'aggressive').
    negative_profile : str
        Profile name for negative ('conservative' / 'moderate' / 'aggressive').
    history_df : pd.DataFrame | None
        Historical data for fitting quantiles.
    out_dir : str | Path | None
        If set, writes corrected CSV + metrics JSON to this directory.

    Returns
    -------
    StackResult
    """
    orch = ResidualStackOrchestrator()
    profile = StackProfile(
        name=f"{spike_profile}+{negative_profile}",
        spike_profile_name=spike_profile,
        negative_profile_name=negative_profile,
    )
    result = orch.run(
        prediction_pack_path=prediction_pack_path,
        spike_risk_path=spike_risk_path,
        profile=profile,
        history_df=history_df,
    )

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result.df.to_csv(out_path / "residual_stack_output.csv", index=False)

        import json
        report = {
            "profile": {
                "name": profile.name,
                "spike": profile.spike_profile_name,
                "negative": profile.negative_profile_name,
            },
            "metrics": result.metrics,
            "data_limited": result.data_limited,
            "output_columns": list(result.df.columns),
        }
        with open(out_path / "residual_stack_manifest.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return result
