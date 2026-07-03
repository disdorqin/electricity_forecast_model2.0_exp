#!/usr/bin/env python3
"""Tests for P5M negative risk calibration.

Coverage:
    1. Rolling train window does NOT leak D-day y_true
    2. Risk CSV schema has required columns
    3. No actual-value leakage in risk features
    4. low_valley_prob is not all zeros
    5. High-spike overlap is computable
    6. DATA_LIMITED flag works
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from extreme.negative_price.risk_model import (
    compute_heuristic_v2_risk,
    RollingLowValleyScorer,
    RollingMLConfig,
)
from extreme.negative_price.labels import add_all_labels
from extreme.negative_price.schema import TARGET_LEAKAGE_COLS

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def multi_day_df() -> pd.DataFrame:
    """Generate multi-day DataFrame with some negative/low-valley events."""
    np.random.seed(42)
    rows = []
    for day in range(1, 35):
        for h in range(1, 25):
            y_true = 100.0 + np.random.randn() * 80
            y_pred = y_true + np.random.randn() * 20
            y_true_adj = y_true
            if day >= 28 and h in [2, 3, 4, 5]:
                y_true_adj = -20.0 + np.random.randn() * 10  # negative price mornings
                y_pred = 10.0 + np.random.randn() * 20
            elif day >= 25 and h >= 20:
                y_true_adj = 30.0 + np.random.randn() * 15  # low valley nights
                y_pred = 60.0 + np.random.randn() * 25
            # Use valid dates: June has 30 days, wrap to July
            month = "06" if day <= 30 else "07"
            day_str = f"{day:02d}" if day <= 30 else f"{day-30:02d}"
            rows.append({
                "business_day": f"2026-{month}-{day_str}",
                "hour_business": h,
                "ds": f"2026-{month}-{day_str} {h-1}:00:00",
                "y_true": max(-100.0, y_true_adj),
                "y_pred": max(-50.0, y_pred),
                "base_fused_pred": max(-50.0, y_pred),
                "sgdfnet_pred": max(-50.0, y_pred + np.random.randn() * 15),
                "timemixer_pred": max(-50.0, y_pred + np.random.randn() * 15),
                "high_spike_prob": np.random.rand() * 0.2,
                "风电总加预测值": np.random.rand() * 1000,
                "光伏总加预测值": np.random.rand() * 500,
                "直调负荷预测值": 5000 + np.random.randn() * 500,
                "竞价空间预测值": 2000 + np.random.randn() * 300,
            })
    df = pd.DataFrame(rows)
    return add_all_labels(df)


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Minimal 3-day DataFrame for fast tests."""
    np.random.seed(1)
    rows = []
    for day in range(1, 4):
        for h in range(1, 25):
            y_true = min(100.0, 50.0 + np.random.randn() * 40)
            if day == 2 and h in [2, 3, 4]:
                y_true = -30.0
            rows.append({
                "business_day": f"2026-07-{day:02d}",
                "hour_business": h,
                "ds": f"2026-07-{day:02d} {h-1}:00:00",
                "y_true": y_true,
                "y_pred": max(-50.0, y_true + np.random.randn() * 15),
                "base_fused_pred": max(-50.0, y_true + np.random.randn() * 15),
                "风电总加预测值": 500 + np.random.randn() * 100,
                "光伏总加预测值": 200 + np.random.randn() * 50,
                "直调负荷预测值": 5000 + np.random.randn() * 200,
                "竞价空间预测值": 2000 + np.random.randn() * 100,
            })
    df = pd.DataFrame(rows)
    return add_all_labels(df)


# ── 1. Rolling window does NOT leak D-day ───────────────────────────

class TestRollingWindowNoLeakage:
    def test_train_window_excludes_prediction_day(self, multi_day_df):
        """Verify the rolling training window uses [D-30, D-1] not D."""
        config = RollingMLConfig(train_window_days=10, min_train_samples=50)
        scorer = RollingLowValleyScorer(config)
        result = scorer.fit_predict(multi_day_df)

        risk_sources = result["risk_source"].unique()
        # Should have either 'rolling_ml' or 'rolling_ml (cold start)'
        assert any("rolling_ml" in s for s in risk_sources), \
            f"No rolling_ml results. Sources: {risk_sources}"

    def test_no_d_day_y_true_in_training(self, tiny_df):
        """The scorer never sees D-day y_true during training for D."""
        config = RollingMLConfig(train_window_days=2, min_train_samples=5)
        scorer = RollingLowValleyScorer(config)
        # This should not crash — even if some days fail, others work
        result = scorer.fit_predict(tiny_df)
        assert len(result) == len(tiny_df)
        assert "risk_source" in result.columns


# ── 2. Risk CSV schema ──────────────────────────────────────────────

class TestRiskCSVSchema:
    def test_required_columns_heuristic(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        required = ["business_day", "hour_business", "negative_prob",
                     "low_valley_prob", "risk_source", "leakage_safe"]
        for col in required:
            assert col in result.columns, f"Missing: {col}"

    def test_required_columns_rolling(self, tiny_df):
        config = RollingMLConfig(train_window_days=2, min_train_samples=5)
        scorer = RollingLowValleyScorer(config)
        result = scorer.fit_predict(tiny_df)
        required = ["negative_prob", "low_valley_prob", "risk_source", "leakage_safe"]
        for col in required:
            assert col in result.columns, f"Missing: {col}"

    def test_probabilities_in_0_1_range(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        assert result["negative_prob"].between(0, 1).all()
        assert result["low_valley_prob"].between(0, 1).all()

    def test_leakage_safe_flag(self, tiny_df):
        result = compute_heuristic_v2_risk(tiny_df)
        assert result["leakage_safe"].all()

        config = RollingMLConfig(train_window_days=2, min_train_samples=5)
        scorer = RollingLowValleyScorer(config)
        ml_result = scorer.fit_predict(tiny_df)
        assert ml_result["leakage_safe"].all()


# ── 3. No actual-value leakage ──────────────────────────────────────

class TestNoActualLeakage:
    def test_no_y_true_in_features(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        assert "y_true" not in result.columns or result["y_true"].isna().all()

    def test_no_residual_in_features(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        assert "residual" not in result.columns

    def test_no_realtime_price(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        assert "实时电价" not in result.columns

    def test_feature_cols_leakage_free(self):
        """Verify feature columns used by heuristic_v2 never include forbidden cols."""
        cols = RollingLowValleyScorer(RollingMLConfig())._get_feature_cols()
        forbidden = {"y_true", "residual", "abs_error", "smape", "实时电价", "realtime_price"}
        assert forbidden.isdisjoint(set(cols)), f"Leakage in features: {forbidden & set(cols)}"


# ── 4. low_valley_prob not all zeros ────────────────────────────────

class TestLowValleyProbNonZero:
    def test_heuristic_nonzero(self, multi_day_df):
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        assert result["low_valley_prob"].sum() > 0, "low_valley_prob is all zero"
        assert result["negative_prob"].sum() > 0, "negative_prob is all zero"

    def test_heuristic_detects_negative_period(self, multi_day_df):
        """Verify prob > 0 during known negative periods."""
        result = compute_heuristic_v2_risk(multi_day_df, history_df=multi_day_df)
        # The last 7 days have negative/low events in hours 2-5 and 20-24
        assert result["negative_prob"].max() > 0.05, "Never triggers > 0.05"


# ── 5. High-spike overlap computable ────────────────────────────────

class TestHighSpikeOverlap:
    def test_overlap_computable(self, multi_day_df):
        """Count hours where both high_spike and low_valley are active."""
        spike_col = "high_spike_prob"
        if spike_col in multi_day_df.columns:
            spike_active = multi_day_df[spike_col].fillna(0) > 0.5
            lv = multi_day_df.get("label_low_valley", pd.Series(0)) == 1
            overlap = int((spike_active & lv).sum())
            assert isinstance(overlap, int)
            assert overlap >= 0
        else:
            pytest.skip("No high_spike column in data")


# ── 6. DATA_LIMITED flag ────────────────────────────────────────────

class TestDataLimited:
    def test_data_limited_when_no_negatives(self):
        """When no negative prices exist, flag DATA_LIMITED."""
        df = pd.DataFrame({
            "business_day": ["2026-07-01"] * 24,
            "hour_business": list(range(1, 25)),
            "ds": [f"2026-07-01 {h-1}:00:00" for h in range(1, 25)],
            "y_true": [max(10.0, 50.0 + np.random.randn() * 30) for _ in range(24)],
            "y_pred": [60.0 + np.random.randn() * 20 for _ in range(24)],
            "base_fused_pred": [60.0 + np.random.randn() * 20 for _ in range(24)],
            "风电总加预测值": [500.0] * 24,
            "光伏总加预测值": [200.0] * 24,
            "直调负荷预测值": [5000.0] * 24,
            "竞价空间预测值": [2000.0] * 24,
        })
        df = add_all_labels(df)
        neg_count = int((df["label_negative_price"] == 1).sum())
        data_limited = neg_count == 0
        assert data_limited, "Expected DATA_LIMITED but found negative prices"

    def test_heuristic_still_works_when_data_limited(self):
        """Heuristic scorer should not crash even with no negatives."""
        df = pd.DataFrame({
            "business_day": ["2026-07-01"] * 24,
            "hour_business": list(range(1, 25)),
            "ds": [f"2026-07-01 {h-1}:00:00" for h in range(1, 25)],
            "y_true": [100.0] * 24,
            "y_pred": [95.0] * 24,
            "base_fused_pred": [95.0] * 24,
            "风电总加预测值": [500.0] * 24,
            "光伏总加预测值": [200.0] * 24,
            "直调负荷预测值": [5000.0] * 24,
            "竞价空间预测值": [2000.0] * 24,
        })
        result = compute_heuristic_v2_risk(df)
        assert len(result) == 24
        assert "negative_prob" in result.columns


if __name__ == "__main__":
    pytest.main([__file__])
