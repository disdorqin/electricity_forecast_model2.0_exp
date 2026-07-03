# -*- coding: utf-8 -*-
"""
apply_negative_correction.py — Apply negative price / low valley correction to predictions.

Pipeline:
    1. Load and merge prediction data
    2. Engineer features (leakage-safe)
    3. Predict negative/low risk
    4. Check mutual exclusion with high_spike
    5. Apply downward residual correction
    6. Guardrail evaluation
    7. Return corrected DataFrame with metadata

Must NOT degrade high_spike correction performance.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from extreme.negative_price.features import engineer_negative_price_features
from extreme.negative_price.guardrail import (
    NegativeGuardrail,
    NegativeGuardrailConfig,
)
from extreme.negative_price.labels import add_all_labels
from extreme.negative_price.residual_correction import (
    NegativeResidualConfig,
    NegativeResidualCorrector,
    get_period,
)
from extreme.negative_price.risk_model import (
    NegativeRiskConfig,
    NegativeRiskModel,
    RiskTarget,
    compute_heuristic_v2_risk,
)


@dataclass
class NegativeCorrectionProfile:
    """Named configuration profile for negative price correction.

    Attributes:
        name: Profile name (conservative / moderate / aggressive).
        risk_threshold: Minimum risk to apply downward correction.
        max_downward_ratio: Max downward as fraction of base_pred.
        max_absolute_downward: Max absolute downward.
        min_pred_floor: Absolute minimum price.
        protect_9_16: Reduce correction during spike-prone hours.
        mode: 'normal' or 'relaxed'.
    """
    name: str = "conservative"
    risk_threshold: float = 0.4
    max_downward_ratio: float = 0.15
    max_absolute_downward: float = 20.0
    min_pred_floor: float = -100.0
    protect_9_16: bool = True
    mode: str = "normal"

    def to_risk_config(self) -> NegativeRiskConfig:
        return NegativeRiskConfig(
            prob_threshold_negative=self.risk_threshold,
            prob_threshold_low_valley=self.risk_threshold,
        )

    def to_residual_config(self) -> NegativeResidualConfig:
        return NegativeResidualConfig(
            risk_threshold=self.risk_threshold,
            max_downward_ratio=self.max_downward_ratio,
            max_absolute_downward=self.max_absolute_downward,
            min_pred_floor=self.min_pred_floor,
            period_9_16_protection=self.protect_9_16,
            mode=self.mode,
        )


# ── Default profiles ──────────────────────────────────────────────────

PROFILES: dict[str, NegativeCorrectionProfile] = {
    "conservative": NegativeCorrectionProfile(
        name="conservative",
        risk_threshold=0.5,
        max_downward_ratio=0.10,
        max_absolute_downward=15.0,
        min_pred_floor=-50.0,
        protect_9_16=True,
    ),
    "moderate": NegativeCorrectionProfile(
        name="moderate",
        risk_threshold=0.35,
        max_downward_ratio=0.20,
        max_absolute_downward=30.0,
        min_pred_floor=-100.0,
        protect_9_16=True,
    ),
    "aggressive": NegativeCorrectionProfile(
        name="aggressive",
        risk_threshold=0.25,
        max_downward_ratio=0.30,
        max_absolute_downward=50.0,
        min_pred_floor=-200.0,
        protect_9_16=False,
    ),
}


def get_profile(name: str) -> NegativeCorrectionProfile:
    """Get a named correction profile."""
    if name not in PROFILES:
        valid = list(PROFILES.keys())
        raise ValueError(f"Unknown profile '{name}'. Available: {valid}")
    return PROFILES[name]


# ── Core correction pipeline ──────────────────────────────────────────

def apply_negative_correction(
    prediction_pack_path: str | Path,
    risk_model: Optional[NegativeRiskModel] = None,
    history_df: Optional[pd.DataFrame] = None,
    profile: Optional[NegativeCorrectionProfile] = None,
    pred_col: str = "base_fused_pred",
    spike_prob_col: str = "high_spike_prob",
) -> pd.DataFrame:
    """Run the full negative price correction pipeline.

    Pipeline:
        1. Load prediction pack
        2. Engineer features
        3. Compute risk scores (or load from risk_model)
        4. Apply downward residual correction per row
        5. Guardrail evaluation
        6. Return augmented DataFrame

    Args:
        prediction_pack_path: Path to prediction pack CSV.
        risk_model: Optional pre-fitted NegativeRiskModel. If None,
                    uses a heuristic risk based on base_pred statistics.
        history_df: Optional historical DataFrame for fitting correction quantiles.
        profile: CorrectionProfile. If None, uses 'conservative'.
        pred_col: Column name for the base fused prediction.
        spike_prob_col: Column name for high-spike probability.

    Returns:
        DataFrame with added columns:
            negative_risk, low_valley_risk, downward_amount,
            negative_corrected_pred, final_pred, negative_reason_code
    """
    profile = profile or get_profile("conservative")

    df = pd.read_csv(prediction_pack_path)

    # Preserve y_true before feature engineering (features may drop it)
    _y_true_col = "y_true" if "y_true" in df.columns else None
    _y_true_values = df[_y_true_col].copy() if _y_true_col is not None else None

    # Engineer features
    feat_df = engineer_negative_price_features(df, pred_col=pred_col, history_df=history_df)

    # Restore y_true if it was dropped
    if _y_true_col is not None and _y_true_col not in feat_df.columns and _y_true_values is not None:
        feat_df[_y_true_col] = _y_true_values

    # ── Risk estimation ────────────────────────────────────────────
    if risk_model is not None and risk_model.is_fitted:
        risk_probas = risk_model.predict_proba(feat_df)
        feat_df["negative_risk"] = risk_probas
        feat_df["low_valley_risk"] = risk_probas  # combined model
    else:
        # Use heuristic_v2 (continuous, multi-signal) by default
        heur = compute_heuristic_v2_risk(
            df if history_df is None else df,
            history_df=history_df,
            pred_col=pred_col,
        )
        feat_df["negative_risk"] = heur["negative_prob"].values
        feat_df["low_valley_risk"] = heur["low_valley_prob"].values

    # ── Residual corrector ─────────────────────────────────────────
    corrector = NegativeResidualCorrector(profile.to_residual_config())
    if history_df is not None and not history_df.empty:
        corrector.fit_from_history(history_df, pred_col=pred_col)
    else:
        # Default modest downward candidates
        corrector.set_downward_candidates({"1_8": -15.0, "9_16": -5.0, "17_24": -10.0})

    # ── Guardrail ──────────────────────────────────────────────────
    guardrail_config = NegativeGuardrailConfig(
        spike_gate_active=True,
        spike_prob_threshold=0.5,
    )
    guardrail = NegativeGuardrail(guardrail_config)

    # ── Apply row-by-row ───────────────────────────────────────────
    downward_amounts: list[float] = []
    final_preds: list[float] = []
    reason_codes: list[str] = []
    negative_corrected_list: list[float] = []

    for _, row in feat_df.iterrows():
        base_pred = row.get(pred_col, 0.0)
        neg_risk = row.get("negative_risk", 0.0)
        lv_risk = row.get("low_valley_risk", 0.0)
        hour_business = row.get("hour_business", 12)
        spike_prob = row.get(spike_prob_col, 0.0)

        if pd.isna(base_pred):
            base_pred = 0.0
        if pd.isna(neg_risk):
            neg_risk = 0.0
        if pd.isna(lv_risk):
            lv_risk = 0.0
        if pd.isna(spike_prob):
            spike_prob = 0.0

        # Step 1: Downward correction
        correction_result = corrector.compute_downward_correction(
            base_pred=float(base_pred),
            negative_risk=float(neg_risk),
            low_valley_risk=float(lv_risk),
            hour_business=int(hour_business),
            high_spike_active=(spike_prob > 0.5),
        )

        # Step 2: Guardrail
        guard_result = guardrail.evaluate(
            base_pred=float(base_pred),
            corrected_pred=correction_result.corrected_pred,
            hour_business=int(hour_business),
            spike_prob=float(spike_prob),
        )

        negative_corrected_list.append(correction_result.corrected_pred)
        final_preds.append(guard_result.final_pred)
        reason_codes.append(guard_result.reason_code)
        downward_amounts.append(guard_result.final_pred - float(base_pred))

    feat_df["negative_risk"] = feat_df["negative_risk"]
    feat_df["low_valley_risk"] = feat_df["low_valley_risk"]
    feat_df["negative_corrected_pred"] = negative_corrected_list
    feat_df["downward_amount"] = downward_amounts
    feat_df["negative_reason_code"] = reason_codes
    feat_df["final_pred"] = final_preds

    return feat_df


def run_correction_with_profile(
    prediction_pack_path: str | Path,
    profile_name: str = "conservative",
    history_df: Optional[pd.DataFrame] = None,
    risk_model: Optional[NegativeRiskModel] = None,
) -> pd.DataFrame:
    """Run negative correction with a named profile."""
    profile = get_profile(profile_name)
    return apply_negative_correction(
        prediction_pack_path=prediction_pack_path,
        risk_model=risk_model,
        history_df=history_df,
        profile=profile,
    )


def compute_metrics(
    df: pd.DataFrame,
    y_true_col: str = "y_true",
    y_pred_before_col: str = "base_fused_pred",
    y_pred_after_col: str = "final_pred",
) -> dict[str, Any]:
    """Compute evaluation metrics for negative correction.

    Metrics:
        - negative_count / low_valley_count
        - negative_MAE_before / after
        - low_valley_MAE_before / after
        - negative_miss_before / after
        - low_valley_overestimate_before / after
        - overall_sMAPE_before / after / delta
        - high_spike_MAE_before / after / delta
        - normal_degradation

    Args:
        df: DataFrame with y_true, predictions before and after.
        y_true_col: Column name for actual values.
        y_pred_before_col: Column name for predictions before correction.
        y_pred_after_col: Column name for predictions after correction.

    Returns:
        Dict of metric name -> value.
    """
    metrics: dict[str, Any] = {}

    y_true = df[y_true_col].values.astype(float)
    before = df[y_pred_before_col].values.astype(float)
    after = df[y_pred_after_col].values.astype(float)

    # Overall periods
    is_9_16 = df["hour_business"].between(9, 16).values
    is_normal = ~is_9_16

    # Counts
    neg_mask = y_true < 0
    lv_mask = y_true <= 50
    metrics["negative_count"] = int(neg_mask.sum())
    metrics["low_valley_count"] = int(lv_mask.sum())

    # Negative price MAE
    if metrics["negative_count"] > 0:
        metrics["negative_MAE_before"] = float(np.mean(np.abs(y_true[neg_mask] - before[neg_mask])))
        metrics["negative_MAE_after"] = float(np.mean(np.abs(y_true[neg_mask] - after[neg_mask])))
    else:
        metrics["negative_MAE_before"] = 0.0
        metrics["negative_MAE_after"] = 0.0

    # Low valley MAE
    if metrics["low_valley_count"] > 0:
        metrics["low_valley_MAE_before"] = float(np.mean(np.abs(y_true[lv_mask] - before[lv_mask])))
        metrics["low_valley_MAE_after"] = float(np.mean(np.abs(y_true[lv_mask] - after[lv_mask])))
    else:
        metrics["low_valley_MAE_before"] = 0.0
        metrics["low_valley_MAE_after"] = 0.0

    # Negative miss (y_true < 0 but y_pred >= 0)
    metrics["negative_miss_before"] = int(np.sum((y_true < 0) & (before >= 0)))
    metrics["negative_miss_after"] = int(np.sum((y_true < 0) & (after >= 0)))

    # Low valley overestimate (y_pred - y_true >= 30 when y_true is low)
    if metrics["low_valley_count"] > 0:
        metrics["low_valley_overestimate_before"] = int(np.sum((before[lv_mask] - y_true[lv_mask]) >= 30))
        metrics["low_valley_overestimate_after"] = int(np.sum((after[lv_mask] - y_true[lv_mask]) >= 30))
    else:
        metrics["low_valley_overestimate_before"] = 0
        metrics["low_valley_overestimate_after"] = 0

    # Overall sMAPE (floor50)
    denom_before = (np.abs(y_true) + np.abs(before)) / 2.0
    denom_before = np.clip(denom_before, 50.0, None)
    smape_before = np.mean(np.abs(y_true - before) / denom_before * 100)
    metrics["overall_sMAPE_before"] = round(float(smape_before), 4)
    denom_after = (np.abs(y_true) + np.abs(after)) / 2.0
    denom_after = np.clip(denom_after, 50.0, None)
    smape_after = np.mean(np.abs(y_true - after) / denom_after * 100)
    metrics["overall_sMAPE_after"] = round(float(smape_after), 4)
    metrics["overall_sMAPE_delta"] = round(float(smape_after - smape_before), 4)

    # High spike MAE (y_true > 150)
    spike_mask = y_true > 150
    if spike_mask.sum() > 0:
        metrics["high_spike_MAE_before"] = float(np.mean(np.abs(y_true[spike_mask] - before[spike_mask])))
        metrics["high_spike_MAE_after"] = float(np.mean(np.abs(y_true[spike_mask] - after[spike_mask])))
    else:
        metrics["high_spike_MAE_before"] = 0.0
        metrics["high_spike_MAE_after"] = 0.0

    if metrics.get("high_spike_MAE_before", 0) > 0:
        metrics["high_spike_MAE_delta"] = round(
            (metrics["high_spike_MAE_after"] - metrics["high_spike_MAE_before"])
            / metrics["high_spike_MAE_before"] * 100, 2
        )
    else:
        metrics["high_spike_MAE_delta"] = 0.0

    # Normal degradation (sMAPE delta for non-9_16 hours)
    if is_normal.sum() > 0:
        yt_n = y_true[is_normal]
        bf_n = before[is_normal]
        af_n = after[is_normal]
        denom_n_before = (np.abs(yt_n) + np.abs(bf_n)) / 2.0
        denom_n_before = np.clip(denom_n_before, 50.0, None)
        smape_n_before = np.mean(np.abs(yt_n - bf_n) / denom_n_before * 100)
        denom_n_after = (np.abs(yt_n) + np.abs(af_n)) / 2.0
        denom_n_after = np.clip(denom_n_after, 50.0, None)
        smape_n_after = np.mean(np.abs(yt_n - af_n) / denom_n_after * 100)
        metrics["normal_degradation"] = round(float(smape_n_after - smape_n_before), 4)
    else:
        metrics["normal_degradation"] = 0.0

    return metrics
