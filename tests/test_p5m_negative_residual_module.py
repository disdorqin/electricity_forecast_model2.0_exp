#!/usr/bin/env python3
"""Tests for the P5M negative price / low valley residual correction module.

Coverage:
    1. Negative label generation
    2. Low-valley label generation (with max() rule)
    3. No actual-value leakage in features
    4. Downward-only correction invariant
    5. Mutual exclusion with high_spike correction
    6. Guardrail max lift/drop cap
    7. Metric computation (correct naming & values)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from extreme.negative_price.schema import (
    NEGATIVE_PRICE_COL,
    LOW_VALLEY_COL,
    OVERESTIMATE_LOW_COL,
    NEGATIVE_PRICE_THRESHOLD,
    LOW_VALLEY_ABSOLUTE_THRESHOLD,
)
from extreme.negative_price.labels import (
    generate_negative_price_labels,
    generate_low_valley_labels,
    generate_overestimate_low_labels,
    add_all_labels,
    compute_low_valley_percentile,
)
from extreme.negative_price.features import (
    engineer_negative_price_features,
    select_feature_columns,
)
from extreme.negative_price.risk_model import (
    NegativeRiskModel,
    NegativeRiskConfig,
    RiskTarget,
)
from extreme.negative_price.residual_correction import (
    NegativeResidualCorrector,
    NegativeResidualConfig,
    get_period,
    DOWNWARD_CORRECTION_APPLIED,
    DOWNWARD_NO_CORRECTION_LOW_RISK,
    DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE,
    DOWNWARD_NO_CORRECTION_ALREADY_LOW,
)
from extreme.negative_price.guardrail import (
    NegativeGuardrail,
    NegativeGuardrailConfig,
)
from extreme.negative_price.apply_negative_correction import (
    apply_negative_correction,
    compute_metrics,
    get_profile,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    np.random.seed(42)
    n = 48
    rows = []
    for h in range(1, n + 1):
        y_true = max(-50.0, 150.0 + np.random.randn() * 80)
        y_pred = max(0.0, 140.0 + np.random.randn() * 70)
        rows.append({
            "business_day": "2026-07-01" if h <= 24 else "2026-07-02",
            "hour_business": h if h <= 24 else h - 24,
            "ds": f"2026-07-01 {h-1}:00:00" if h <= 24 else f"2026-07-02 {h-25}:00:00",
            "y_true": y_true,
            "y_pred": y_pred,
            "base_fused_pred": y_pred,
            "high_spike_prob": np.random.rand() * 0.3,
            "风电总加预测值": np.random.rand() * 1000,
            "光伏总加预测值": np.random.rand() * 500,
            "直调负荷预测值": 5000 + np.random.randn() * 500,
            "竞价空间预测值": 2000 + np.random.randn() * 300,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_df_with_negatives(sample_df: pd.DataFrame) -> pd.DataFrame:
    df = sample_df.copy()
    df.loc[0, "y_true"] = -50.0   # negative price
    df.loc[0, "base_fused_pred"] = 30.0
    df.loc[1, "y_true"] = 20.0    # low valley (<= 50)
    df.loc[1, "base_fused_pred"] = 80.0
    df.loc[2, "y_true"] = 10.0    # low valley + overestimate
    df.loc[2, "base_fused_pred"] = 100.0
    return df


# ── 1. Negative label ────────────────────────────────────────────────

class TestNegativeLabel:
    def test_detects_negative_price(self, sample_df_with_negatives):
        labels = generate_negative_price_labels(sample_df_with_negatives)
        assert labels.iloc[0] == 1  # y_true = -50
        assert labels.iloc[1] == 0  # y_true = 20 > 0

    def test_all_non_negative(self, sample_df):
        """Negative label only fires for y_true < 0."""
        labels = generate_negative_price_labels(sample_df)
        # Verify every 1 has y_true < 0 (no false positives)
        assert (labels == 1).sum() == (sample_df["y_true"] < 0).sum()


# ── 2. Low-valley label (max() rule) ─────────────────────────────────

class TestLowValleyLabel:
    def test_detects_low_valley(self, sample_df_with_negatives):
        """low_valley = y_true <= 50."""
        labels = generate_low_valley_labels(sample_df_with_negatives)
        assert labels.iloc[0] == 1  # y_true = -50 <= 50
        assert labels.iloc[1] == 1  # y_true = 20 <= 50
        assert labels.iloc[2] == 1  # y_true = 10 <= 50

    def test_uses_max_rule(self):
        """effective = max(50, p10) per spec."""
        df = pd.DataFrame({"y_true": [30, 40, 60, 80, 100, 120]})
        p10 = compute_low_valley_percentile(df)
        # p10 of [30, 40, 60, 80, 100, 120] at 10% is between 30 and 40
        labels = generate_low_valley_labels(df, percentile_threshold=p10)
        # effective = max(50, p10) = 50 since p10 < 50
        assert labels.iloc[0] == 1  # 30 <= 50
        assert labels.iloc[2] == 0  # 60 > 50

    def test_compute_percentile(self, sample_df):
        p10 = compute_low_valley_percentile(sample_df)
        assert isinstance(p10, float)
        assert p10 > -200


# ── 3. No actual-value leakage ───────────────────────────────────────

class TestNoLeakage:
    def test_no_y_true_in_features(self, sample_df):
        feat_df = engineer_negative_price_features(sample_df)
        assert "y_true" not in feat_df.columns or feat_df["y_true"].isna().all()

    def test_target_leakage_cols_dropped(self, sample_df):
        df = sample_df.copy()
        df["residual"] = df["y_true"] - df["base_fused_pred"]
        feat_df = engineer_negative_price_features(df)
        assert "residual" not in feat_df.columns

    def test_no_realtime_price(self, sample_df):
        df = sample_df.copy()
        df["实时电价"] = df["y_true"]
        feat_df = engineer_negative_price_features(df)
        assert "实时电价" not in feat_df.columns

    def test_all_prediction_signals_are_predictions(self):
        """Verify feature columns never contain y_true or residual."""
        from extreme.negative_price.features import get_feature_columns
        cols = get_feature_columns()
        forbidden = {"y_true", "residual", "abs_error", "smape", "实时电价", "realtime_price"}
        assert forbidden.isdisjoint(set(cols)), f"Leakage in features: {forbidden & set(cols)}"


# ── 4. Downward-only correction ──────────────────────────────────────

class TestDownwardOnly:
    def test_downward_amount_never_positive(self):
        """Corrected prediction <= base_pred."""
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        for risk in [0.0, 0.3, 0.6, 0.9]:
            result = corrector.compute_downward_correction(
                base_pred=100.0, negative_risk=risk, low_valley_risk=risk,
                hour_business=10,
            )
            assert result.corrected_pred <= 100.0, f"risk={risk}: corrected > base"

    def test_final_pred_after_negative_never_exceeds_base(self, sample_df_with_negatives):
        """Integration check: final_after_negative <= final_before_negative."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "pack.csv"
            sample_df_with_negatives.to_csv(pack_path, index=False)
            result = apply_negative_correction(
                prediction_pack_path=pack_path, profile=get_profile("aggressive"),
                pred_col="base_fused_pred",
            )
            # final_pred should be <= base_fused_pred everywhere
            assert (result["final_pred"] <= result["base_fused_pred"] + 1e-6).all()


# ── 5. High-spike mutual exclusion ───────────────────────────────────

class TestHighSpikeMutualExclusion:
    def test_correction_skipped_when_high_spike_active(self):
        """When high_spike_active=True, downward correction is not applied."""
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        result = corrector.compute_downward_correction(
            base_pred=100.0, negative_risk=0.8, low_valley_risk=0.9,
            hour_business=10, high_spike_active=True,
        )
        assert result.downward_amount == 0.0
        assert result.reason_code == DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE

    def test_guardrail_rejects_when_spike_prob_high(self):
        """Guardrail reverts correction when spike_prob > threshold."""
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(spike_prob_threshold=0.5))
        result = guardrail.evaluate(
            base_pred=100.0, corrected_pred=80.0,
            hour_business=10, spike_prob=0.8,
        )
        assert result.final_pred == 100.0  # reverted to base
        assert "HIGH_SPIKE_GATE" in result.reason_code


# ── 6. Guardrail caps ────────────────────────────────────────────────

class TestGuardrailCap:
    def test_max_downward_by_ratio(self):
        """9_16 hours: max downward = base_pred * 0.10 = 10."""
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(
            max_downward_ratio_9_16=0.10, max_absolute_downward_9_16=100.0,
        ))
        result = guardrail.evaluate(
            base_pred=100.0, corrected_pred=50.0, hour_business=10,
        )
        # 100 * 0.10 = 10, so final = 100 - 10 = 90
        assert result.final_pred == 90.0

    def test_absolute_price_floor(self):
        """Prediction must stay above min_allowed_price."""
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(min_allowed_price=-200.0))
        result = guardrail.evaluate(
            base_pred=0.0, corrected_pred=-300.0, hour_business=1,
        )
        assert result.final_pred >= -200.0

    def test_1_8_period_cap(self):
        """1_8 hours have wider caps."""
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(
            max_downward_ratio_1_8=0.25, max_absolute_downward_1_8=40.0,
        ))
        result = guardrail.evaluate(
            base_pred=100.0, corrected_pred=50.0, hour_business=3,
        )
        # 100 * 0.25 = 25, so final = 100 - 25 = 75
        assert result.final_pred == 75.0


# ── 7. Metrics ───────────────────────────────────────────────────────

class TestMetrics:
    def test_all_required_metrics_present(self, sample_df_with_negatives):
        df = sample_df_with_negatives.copy()
        df["final_pred"] = df["base_fused_pred"].values * 0.9
        metrics = compute_metrics(df)

        required = [
            "negative_count", "low_valley_count",
            "negative_MAE_before", "negative_MAE_after",
            "low_valley_MAE_before", "low_valley_MAE_after",
            "negative_miss_before", "negative_miss_after",
            "low_valley_overestimate_before", "low_valley_overestimate_after",
            "overall_sMAPE_before", "overall_sMAPE_after", "overall_sMAPE_improvement",
            "high_spike_MAE_before", "high_spike_MAE_after", "high_spike_MAE_improvement",
            "normal_degradation",
        ]
        for key in required:
            assert key in metrics, f"Missing metric: {key}"

    def test_negative_MAE_computed_correctly(self):
        """Manually verify negative_MAE for one sample."""
        df = pd.DataFrame({
            "hour_business": [1, 2, 3, 4, 5],
            "y_true": [-50.0, -20.0, 10.0, 100.0, 200.0],
            "base_fused_pred": [30.0, 0.0, 40.0, 90.0, 180.0],
            "final_pred": [20.0, -10.0, 35.0, 95.0, 190.0],
        })
        metrics = compute_metrics(df)
        assert metrics["negative_count"] == 2
        # Before: | -50-30 | + | -20-0 | = 80+20 = 100, /2 = 50.0
        assert metrics["negative_MAE_before"] == 50.0
        # After: | -50-20 | + | -20-(-10) | = 70+10 = 80, /2 = 40.0
        assert metrics["negative_MAE_after"] == 40.0

    def test_low_valley_overestimate_count(self):
        """Verify overestimate counts with manual values."""
        df = pd.DataFrame({
            "hour_business": [1, 2, 3],
            "y_true": [10.0, 20.0, 60.0],
            "base_fused_pred": [100.0, 40.0, 70.0],
            "final_pred": [60.0, 30.0, 65.0],
        })
        metrics = compute_metrics(df)
        # low_valley = y_true <= 50 => rows 0, 1
        # overestimate = (pred - y_true) >= 30
        # row0 before: 100-10=90 >=30, after: 60-10=50 >=30
        # row1 before: 40-20=20 <30, after: 30-20=10 <30
        assert metrics["low_valley_overestimate_before"] == 1
        assert metrics["low_valley_overestimate_after"] == 1

    def test_smape_not_worsened(self, sample_df_with_negatives):
        """smoking test — correction should not drastically worsen sMAPE."""
        df = sample_df_with_negatives.copy()
        df["final_pred"] = df["base_fused_pred"].values * 0.95
        metrics = compute_metrics(df)
        assert abs(metrics["overall_sMAPE_improvement"]) < 10.0


# ── Integration tests ────────────────────────────────────────────────

class TestIntegration:
    def test_apply_negative_correction_pipeline(self, sample_df_with_negatives):
        with tempfile.TemporaryDirectory() as tmpdir:
            pack_path = Path(tmpdir) / "pack.csv"
            sample_df_with_negatives.to_csv(pack_path, index=False)
            result = apply_negative_correction(
                prediction_pack_path=pack_path,
                profile=get_profile("conservative"),
                pred_col="base_fused_pred",
            )
            assert "negative_corrected_pred" in result.columns
            assert "downward_amount" in result.columns
            assert "negative_reason_code" in result.columns
            assert "final_pred" in result.columns
            assert "final_pred_before_negative" in result.columns

    def test_profiles(self):
        for name in ["conservative", "moderate", "aggressive"]:
            profile = get_profile(name)
            assert profile.name == name
            assert profile.risk_threshold > 0
        with pytest.raises(ValueError):
            get_profile("nonexistent")

    def test_add_all_labels(self, sample_df_with_negatives):
        result = add_all_labels(sample_df_with_negatives)
        for col in [NEGATIVE_PRICE_COL, LOW_VALLEY_COL, OVERESTIMATE_LOW_COL]:
            assert col in result.columns

    def test_new_feature_names_present(self, sample_df):
        feat_df = engineer_negative_price_features(sample_df)
        expected_new = [
            "recent_negative_rate_by_hour", "recent_low_price_rate_by_hour",
            "model_disagreement", "final_pred_before_negative",
        ]
        for col in expected_new:
            assert col in feat_df.columns, f"Missing feature: {col}"


if __name__ == "__main__":
    pytest.main([__file__])
