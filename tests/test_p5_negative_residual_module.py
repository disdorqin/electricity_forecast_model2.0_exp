#!/usr/bin/env python3
"""Tests for the P5 negative price / low valley residual correction module."""

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
    df.loc[0, "y_true"] = -50.0  # negative price
    df.loc[0, "base_fused_pred"] = 30.0
    df.loc[1, "y_true"] = 20.0   # low valley
    df.loc[1, "base_fused_pred"] = 80.0
    df.loc[2, "y_true"] = 10.0   # low valley + overestimate
    df.loc[2, "base_fused_pred"] = 100.0
    return df


# ── Schema tests ──────────────────────────────────────────────────────

class TestSchema:
    def test_constants(self):
        assert NEGATIVE_PRICE_THRESHOLD == 0.0

    def test_label_cols_defined(self):
        assert NEGATIVE_PRICE_COL == "label_negative_price"
        assert LOW_VALLEY_COL == "label_low_valley"
        assert OVERESTIMATE_LOW_COL == "label_overestimate_low"


# ── Labels tests ─────────────────────────────────────────────────────

class TestLabels:
    def test_negative_price_labels(self, sample_df_with_negatives):
        labels = generate_negative_price_labels(sample_df_with_negatives)
        assert labels.iloc[0] == 1  # y_true = -50
        assert labels.iloc[1] == 0  # y_true = 20 > 0

    def test_low_valley_labels(self, sample_df_with_negatives):
        labels = generate_low_valley_labels(sample_df_with_negatives)
        assert labels.iloc[2] == 1  # y_true = 10 <= 50
        assert labels.iloc[3] == 0  # y_true > 50

    def test_overestimate_low_labels(self, sample_df_with_negatives):
        labels = generate_overestimate_low_labels(
            sample_df_with_negatives, threshold=30,
        )
        assert labels.iloc[2] == 1  # 100 - 10 = 90 >= 30

    def test_add_all_labels(self, sample_df_with_negatives):
        result = add_all_labels(sample_df_with_negatives)
        for col in [NEGATIVE_PRICE_COL, LOW_VALLEY_COL, OVERESTIMATE_LOW_COL]:
            assert col in result.columns

    def test_compute_percentile(self, sample_df):
        p10 = compute_low_valley_percentile(sample_df)
        assert isinstance(p10, float)
        assert p10 > -200


# ── Features tests ────────────────────────────────────────────────────

class TestFeatures:
    def test_engineer_features(self, sample_df):
        result = engineer_negative_price_features(sample_df)
        expected_cols = ["period", "prediction_spread", "renewable_ratio",
                         "negative_price_rate_hour"]
        for col in expected_cols:
            assert col in result.columns, f"Missing: {col}"

    def test_select_feature_columns(self, sample_df):
        feat_df = engineer_negative_price_features(sample_df)
        cols = select_feature_columns(feat_df)
        assert len(cols) > 0
        for col in cols:
            assert col in feat_df.columns

    def test_no_y_true_leakage(self, sample_df):
        feat_df = engineer_negative_price_features(sample_df)
        assert "y_true" not in feat_df.columns or feat_df["y_true"].isna().all()

    def test_prediction_spread(self, sample_df):
        feat_df = engineer_negative_price_features(sample_df)
        assert "prediction_spread" in feat_df.columns


# ── Risk model tests ──────────────────────────────────────────────────

class TestRiskModel:
    def test_fit_and_predict(self, sample_df_with_negatives):
        df = sample_df_with_negatives.copy()
        df = add_all_labels(df)
        model = NegativeRiskModel()
        model.fit(df)
        assert model.is_fitted
        probas = model.predict_proba(df)
        assert len(probas) == len(df)
        assert np.all((probas >= 0) & (probas <= 1))

    def test_not_fitted_returns_zero(self, sample_df):
        model = NegativeRiskModel()
        probas = model.predict_proba(sample_df)
        assert np.all(probas == 0)

    def test_lr_model(self, sample_df_with_negatives):
        df = sample_df_with_negatives.copy()
        df = add_all_labels(df)
        config = NegativeRiskConfig(model_type="lr")
        model = NegativeRiskModel(config)
        model.fit(df)
        assert model.is_fitted


# ── Residual correction tests ────────────────────────────────────────

class TestResidualCorrection:
    def test_fit_from_history(self, sample_df):
        corrector = NegativeResidualCorrector()
        corrector.fit_from_history(sample_df)
        quantiles = corrector.get_quantiles()
        assert "1_8" in quantiles
        assert "9_16" in quantiles
        assert "17_24" in quantiles

    def test_downward_correction_low_risk(self):
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        result = corrector.compute_downward_correction(
            base_pred=100.0,
            negative_risk=0.0,
            low_valley_risk=0.0,
            hour_business=10,
        )
        assert result.downward_amount == 0.0
        assert result.reason_code == DOWNWARD_NO_CORRECTION_LOW_RISK

    def test_downward_correction_applied(self):
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        result = corrector.compute_downward_correction(
            base_pred=100.0,
            negative_risk=0.8,
            low_valley_risk=0.9,
            hour_business=10,
        )
        assert result.downward_amount < 0
        assert result.reason_code in (DOWNWARD_CORRECTION_APPLIED, "DOWNWARD_CORRECTION_CAPPED")

    def test_high_spike_mutual_exclusion(self):
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        result = corrector.compute_downward_correction(
            base_pred=100.0,
            negative_risk=0.8,
            low_valley_risk=0.9,
            hour_business=10,
            high_spike_active=True,
        )
        assert result.downward_amount == 0.0
        assert result.reason_code == DOWNWARD_NO_CORRECTION_HIGH_SPIKE_GATE

    def test_already_low_no_correction(self):
        corrector = NegativeResidualCorrector()
        corrector.set_downward_candidates({"1_8": -10.0, "9_16": -5.0, "17_24": -8.0})
        result = corrector.compute_downward_correction(
            base_pred=-10.0,
            negative_risk=0.8,
            low_valley_risk=0.9,
            hour_business=10,
        )
        assert result.downward_amount == 0.0

    def test_get_period(self):
        assert get_period(3) == "1_8"
        assert get_period(12) == "9_16"
        assert get_period(20) == "17_24"
        with pytest.raises(ValueError):
            get_period(25)


# ── Guardrail tests ──────────────────────────────────────────────────

class TestGuardrail:
    def test_high_spike_gate_rejects(self):
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(spike_prob_threshold=0.5))
        result = guardrail.evaluate(
            base_pred=100.0,
            corrected_pred=80.0,
            hour_business=10,
            spike_prob=0.8,
        )
        assert result.final_pred == 100.0  # reverted to base
        assert "HIGH_SPIKE_GATE" in result.reason_code

    def test_no_gate_with_low_spike_prob(self):
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(spike_prob_threshold=0.5))
        result = guardrail.evaluate(
            base_pred=100.0,
            corrected_pred=80.0,
            hour_business=10,
            spike_prob=0.1,
        )
        # corrected_pred=80 but guardrail max downward for 9_16 is 100*0.10=10, so final=90
        assert result.final_pred == 90.0

    def test_min_price_floor(self):
        guardrail = NegativeGuardrail(NegativeGuardrailConfig(min_allowed_price=-200.0))
        result = guardrail.evaluate(
            base_pred=0.0,
            corrected_pred=-300.0,
            hour_business=10,
        )
        assert result.final_pred >= -200.0


# ── Apply correction tests ───────────────────────────────────────────

class TestApplyCorrection:
    def test_apply_negative_correction(self, sample_df_with_negatives):
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

    def test_compute_metrics(self, sample_df_with_negatives):
        df = sample_df_with_negatives.copy()
        df["final_pred"] = df["base_fused_pred"].values * 0.9  # slight downward shift
        metrics = compute_metrics(df)
        assert "negative_MAE_before" in metrics
        assert "negative_MAE_after" in metrics
        assert "overall_sMAPE_before" in metrics
        assert "overall_sMAPE_after" in metrics
        assert "high_spike_degradation" in metrics
        assert "normal_degradation" in metrics

    def test_profile_exists(self):
        profile = get_profile("conservative")
        assert profile.name == "conservative"
        assert profile.risk_threshold > 0

        profile = get_profile("aggressive")
        assert profile.name == "aggressive"
        assert profile.risk_threshold < 0.5

        with pytest.raises(ValueError):
            get_profile("nonexistent")


if __name__ == "__main__":
    pytest.main([__file__])
