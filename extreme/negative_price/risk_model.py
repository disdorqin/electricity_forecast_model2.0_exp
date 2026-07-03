# -*- coding: utf-8 -*-
"""
risk_model.py — Negative/low price risk estimation model.

Provides:
    - heuristic_v2: Rule-based continuous risk scorer (no model training)
    - RollingLowValleyScorer: Daily walk-forward ML risk scorer
    - NegativeRiskModel: Original single-fit sklearn classifier (kept for compat)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from extreme.negative_price.features import (
    engineer_negative_price_features,
    select_feature_columns,
)
from extreme.negative_price.schema import (
    NEGATIVE_PRICE_COL,
    LOW_VALLEY_COL,
    NEGATIVE_PRICE_THRESHOLD,
)


# ── Heuristic V2: rule-based continuous risk scorer ──────────────────────

def compute_heuristic_v2_risk(
    df: pd.DataFrame,
    pred_col: str = "base_fused_pred",
    history_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute continuous negative/low-valley risk scores using only
    prediction-time safe features.

    Uses:
        hour_business, period, base_fused_pred, prediction_spread,
        renewable_ratio, 直调负荷预测值, 竞价空间预测值,
        recent_low_price_rate_by_hour, recent_negative_rate_by_hour,
        recent_mean_low_residual_by_hour

    Returns DataFrame with columns: negative_prob, low_valley_prob.
    Both are continuous 0-1 scores, not hard thresholds.
    """
    feat = engineer_negative_price_features(df, history_df=history_df, pred_col=pred_col)

    # Preserve only identification columns + risk scores (no y_true leakage)
    id_cols = []
    for c in ["business_day", "hour_business", "ds", "timestamp"]:
        if c in df.columns:
            id_cols.append(c)

    pred = feat.get(pred_col, pd.Series(0.0, index=df.index)).fillna(0.0)
    recent_neg_rate = feat.get("recent_negative_rate_by_hour", pd.Series(0.0, index=df.index)).fillna(0.0)
    recent_low_rate = feat.get("recent_low_price_rate_by_hour", pd.Series(0.0, index=df.index)).fillna(0.0)
    recent_low_res = feat.get("recent_mean_low_residual_by_hour", pd.Series(0.0, index=df.index)).fillna(0.0)
    renewable_ratio = feat.get("renewable_ratio", pd.Series(0.0, index=df.index)).fillna(0.0)
    spread = feat.get("prediction_spread", pd.Series(0.0, index=df.index)).fillna(0.0)

    # Hour factors: early morning (1-8) and late night (17-24) have higher low risk
    hour = feat.get("hour_business", pd.Series(12, index=df.index)).fillna(12)
    hour_factor = np.where(
        (hour >= 1) & (hour <= 8), 0.15,
        np.where((hour >= 17) & (hour <= 24), 0.10, 0.05)
    )

    # Pred level factor: lower pred → higher risk
    pred_factor = np.clip(1.0 - pred / 200.0, 0.0, 1.0)

    # Recent negative/low rate factors
    neg_rate_factor = recent_neg_rate.values * 0.30
    low_rate_factor = recent_low_rate.values * 0.25

    # Residual factor: if history shows overprediction in low regime → higher risk
    res_factor = np.clip(-recent_low_res.values / 50.0, 0.0, 0.20)  # negative residual = overprediction

    # Renewable ratio factor: high renewable → higher low-price risk
    ren_factor = renewable_ratio.values * 0.05

    # Spread factor: high disagreement → uncertainty
    spread_factor = np.clip(spread.values / 100.0, 0.0, 0.05)

    # Build result from identification columns only (no y_true leakage)
    result = df[id_cols].copy() if id_cols else pd.DataFrame(index=df.index)

    # Composite negative probability
    neg_prob = np.clip(
        hour_factor * 0.10
        + pred_factor * 0.20
        + neg_rate_factor * 0.25
        + res_factor * 0.15
        + ren_factor * 0.10
        + spread_factor * 0.05,
        0.0, 1.0,
    )

    # Composite low-valley probability (broader, includes more signals)
    lv_prob = np.clip(
        hour_factor * 0.10
        + pred_factor * 0.25
        + low_rate_factor * 0.20
        + neg_rate_factor * 0.10
        + res_factor * 0.15
        + ren_factor * 0.10
        + spread_factor * 0.05,
        0.0, 1.0,
    )

    result["negative_prob"] = neg_prob
    result["low_valley_prob"] = lv_prob
    result["risk_source"] = "heuristic_v2"
    result["leakage_safe"] = True
    return result


# ── Rolling low-valley ML scorer ────────────────────────────────────────

@dataclass
class RollingMLConfig:
    """Configuration for rolling walk-forward risk scoring.

    Attributes:
        train_window_days: Rolling training window.
        target_label: Which label to predict ('low_valley', 'negative_price', 'combined').
        rf_n_estimators: Number of trees.
        rf_max_depth: Max tree depth.
        prob_threshold_low_valley: Decision threshold for low_valley.
        min_train_samples: Minimum samples to attempt training.
    """
    train_window_days: int = 30
    target_label: str = "combined"
    rf_n_estimators: int = 100
    rf_max_depth: int = 6
    prob_threshold_low_valley: float = 0.3
    prob_threshold_negative: float = 0.3
    min_train_samples: int = 50


class RollingLowValleyScorer:
    """Daily walk-forward ML risk scorer.

    For each day D:
        train on [D - train_window_days, D - 1]
        predict D

    Uses low_valley (or combined) label as target.
    No D-day y_true, actual columns, or residuals used at prediction time.
    """

    def __init__(self, config: Optional[RollingMLConfig] = None):
        self.config = config or RollingMLConfig()
        self._models: dict[str, Any] = {}  # business_day -> model
        self._feature_cols: list[str] = []

    def fit_predict(
        self,
        df: pd.DataFrame,
        business_day_col: str = "business_day",
        pred_col: str = "base_fused_pred",
    ) -> pd.DataFrame:
        """Run walk-forward training and return risk scores for all rows.

        Args:
            df: Full DataFrame with labels (label_negative_price, label_low_valley).
            business_day_col: Column name for business day.
            pred_col: Column name for base prediction.

        Returns:
            DataFrame with added columns:
                negative_prob, low_valley_prob, overestimate_low_prob,
                risk_source, leakage_safe
        """
        df = df.copy()
        days = sorted(df[business_day_col].unique())
        self._feature_cols = self._get_feature_cols()

        result_rows: list[pd.DataFrame] = []

        for i, day in enumerate(days):
            if i == 0:
                # First day: no history, use heuristic fallback
                day_df = df[df[business_day_col] == day].copy()
                heur = compute_heuristic_v2_risk(day_df, pred_col=pred_col)
                heur["risk_source"] = "rolling_ml (cold start)"
                result_rows.append(heur)
                continue

            # Training window: [i - train_window, i - 1]
            train_days = days[max(0, i - self.config.train_window_days):i]
            train_df = df[df[business_day_col].isin(train_days)].copy()
            test_df = df[df[business_day_col] == day].copy()

            # Train model
            model = self._train_day_model(train_df)
            if model is None:
                # Fallback to heuristic
                heur = compute_heuristic_v2_risk(test_df, pred_col=pred_col)
                heur["risk_source"] = "rolling_ml (train failed)"
                result_rows.append(heur)
                continue

            # Predict
            result = self._predict_day(test_df, model, pred_col=pred_col)
            result["risk_source"] = "rolling_ml"
            result_rows.append(result)

        if not result_rows:
            return compute_heuristic_v2_risk(df, pred_col=pred_col)

        final = pd.concat(result_rows, ignore_index=True)
        final["leakage_safe"] = True
        return final

    def _get_feature_cols(self) -> list[str]:
        """Return the feature columns used by the ML model."""
        base_features = [
            "hour_business", "hour",
            "base_fused_pred", "dayahead_proxy", "prediction_spread",
            "renewable_ratio",
            "recent_negative_rate_by_hour", "recent_low_price_rate_by_hour",
            "recent_mean_low_residual_by_hour",
            "min_pred_last_24h",
        ]
        # Add forecast exogenous if available
        exog = [
            "风电总加预测值", "光伏总加预测值", "新能源总加预测值",
            "直调负荷预测值", "竞价空间预测值",
        ]
        return base_features + exog

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        """Prepare target label for training."""
        if self.config.target_label == "negative_price":
            return df.get(NEGATIVE_PRICE_COL, pd.Series(0, index=df.index)).fillna(0).astype(int)
        elif self.config.target_label == "low_valley":
            return df.get(LOW_VALLEY_COL, pd.Series(0, index=df.index)).fillna(0).astype(int)
        else:  # combined
            neg = df.get(NEGATIVE_PRICE_COL, pd.Series(0, index=df.index)).fillna(0).astype(int)
            low = df.get(LOW_VALLEY_COL, pd.Series(0, index=df.index)).fillna(0).astype(int)
            return ((neg + low) > 0).astype(int)

    def _train_day_model(
        self, train_df: pd.DataFrame,
    ) -> Optional[Any]:
        """Train a RandomForest for a single day."""
        feat_df = engineer_negative_price_features(train_df)
        available = [c for c in self._feature_cols if c in feat_df.columns]
        if len(available) < 3:
            return None

        X = feat_df[available].fillna(0.0).values.astype(np.float32)
        y = self._prepare_target(train_df).values

        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X, y = X[mask], y[mask]
        if len(X) < self.config.min_train_samples:
            return None

        pos_ratio = y.mean()
        if pos_ratio == 0 or pos_ratio == 1:
            return None  # no variation

        model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=1,
        )
        model.fit(X, y)
        return model

    def _predict_day(
        self, test_df: pd.DataFrame, model: Any,
        pred_col: str = "base_fused_pred",
    ) -> pd.DataFrame:
        """Predict risk for one day."""
        feat_df = engineer_negative_price_features(test_df)
        available = [c for c in self._feature_cols if c in feat_df.columns]
        X = np.zeros((len(feat_df), len(available)), dtype=np.float32)
        for i, col in enumerate(available):
            X[:, i] = feat_df[col].fillna(0.0).values

        probas = model.predict_proba(X)
        lv_prob = probas[:, 1] if probas.shape[1] >= 2 else probas[:, 0]

        result = test_df.copy()
        result["negative_prob"] = np.clip(lv_prob * 0.5, 0.0, 1.0)  # heuristic scaling
        result["low_valley_prob"] = lv_prob
        result["overestimate_low_prob"] = np.clip(lv_prob * 0.7, 0.0, 1.0)
        result["leakage_safe"] = True
        return result


# ── Legacy NegativeRiskModel (kept for backward compat) ─────────────────

class RiskTarget:
    NEGATIVE_PRICE = "negative_price"
    LOW_VALLEY = "low_valley"
    COMBINED = "combined"


@dataclass
class NegativeRiskConfig:
    target: str = RiskTarget.COMBINED
    model_type: str = "rf"
    feature_extra: Optional[list[str]] = None
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    lr_c: float = 1.0
    prob_threshold_negative: float = 0.3
    prob_threshold_low_valley: float = 0.3


class NegativeRiskModel:
    """Legacy single-fit NegativeRiskModel. Prefer RollingLowValleyScorer for
    walk-forward scoring or compute_heuristic_v2_risk for rule-based."""

    def __init__(self, config: Optional[NegativeRiskConfig] = None):
        self.config = config or NegativeRiskConfig()
        self._model: Any = None
        self._feature_cols: list[str] = []
        self._fitted: bool = False

    def fit(self, df: pd.DataFrame, label_df: Optional[pd.DataFrame] = None) -> "NegativeRiskModel":
        feat_df = engineer_negative_price_features(df)
        self._feature_cols = select_feature_columns(feat_df, extra_cols=self.config.feature_extra)
        train_df = feat_df[self._feature_cols].copy()
        if label_df is not None:
            y = self._prepare_target(label_df)
        else:
            y = self._prepare_target(df)

        mask = train_df.notna().all(axis=1) & y.notna()
        X, y = train_df[mask].values.astype(np.float32), y[mask].values.astype(int)
        if len(X) < 10:
            warnings.warn(f"Only {len(X)} training samples")
            self._fitted = False
            return self

        self._model = RandomForestClassifier(
            n_estimators=self.config.rf_n_estimators,
            max_depth=self.config.rf_max_depth, random_state=42,
            class_weight="balanced", n_jobs=1,
        )
        self._model.fit(X, y)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted or self._model is None:
            return np.zeros(len(df))
        feat_df = engineer_negative_price_features(df)
        X = np.zeros((len(feat_df), len(self._feature_cols)), dtype=np.float32)
        for i, col in enumerate(self._feature_cols):
            if col in feat_df.columns:
                X[:, i] = feat_df[col].fillna(0).values
        probas = self._model.predict_proba(X)
        return probas[:, 1] if probas.shape[1] >= 2 else probas[:, 0]

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        if self.config.target == RiskTarget.NEGATIVE_PRICE:
            return df.get(NEGATIVE_PRICE_COL, 0).astype(int)
        elif self.config.target == RiskTarget.LOW_VALLEY:
            return df.get(LOW_VALLEY_COL, 0).astype(int)
        else:
            neg = df.get(NEGATIVE_PRICE_COL, 0).astype(int)
            low = df.get(LOW_VALLEY_COL, 0).astype(int)
            return ((neg + low) > 0).astype(int)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_cols)


def fit_risk_model(df: pd.DataFrame, config: Optional[NegativeRiskConfig] = None) -> NegativeRiskModel:
    model = NegativeRiskModel(config)
    model.fit(df)
    return model
