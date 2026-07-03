# -*- coding: utf-8 -*-
"""
risk_model.py — Negative/low price risk estimation model.

Provides:
    - NegativeRiskConfig: Configuration for risk estimation
    - NegativeRiskModel: Predicts probability of negative/low price events
    - fit_risk_model: Convenience function to train on historical data

The risk model is used to decide WHEN to apply downward residual correction.
It must be leakage-safe: no y_true or actual values at inference.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from extreme.negative_price.features import (
    engineer_negative_price_features,
    select_feature_columns,
)
from extreme.negative_price.schema import (
    NEGATIVE_PRICE_COL,
    LOW_VALLEY_COL,
)


class RiskTarget:
    """Target label to predict."""
    NEGATIVE_PRICE = "negative_price"
    LOW_VALLEY = "low_valley"
    COMBINED = "combined"


@dataclass
class NegativeRiskConfig:
    """Configuration for negative/low price risk estimation.

    Attributes:
        target: Which label to predict.
        model_type: 'rf' (RandomForest) or 'lr' (LogisticRegression).
        feature_extra: Extra feature columns to include.
        rf_n_estimators: Number of trees for RF.
        rf_max_depth: Max depth for RF.
        lr_c: Inverse regularization for LR.
        prob_threshold_negative: Threshold for negative price alert.
        prob_threshold_low_valley: Threshold for low valley alert.
    """
    target: str = RiskTarget.COMBINED
    model_type: str = "rf"
    feature_extra: Optional[list[str]] = None
    rf_n_estimators: int = 100
    rf_max_depth: int = 8
    lr_c: float = 1.0
    prob_threshold_negative: float = 0.3
    prob_threshold_low_valley: float = 0.3


class NegativeRiskModel:
    """Risk model for negative/low price events.

    Predicts probability that the next prediction will be in a negative
    or low-valley price regime.
    """

    def __init__(self, config: Optional[NegativeRiskConfig] = None):
        self.config = config or NegativeRiskConfig()
        self._model: Any = None
        self._feature_cols: list[str] = []
        self._fitted: bool = False

    def _build_model(self) -> Any:
        if self.config.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=42,
                class_weight="balanced",
                n_jobs=1,
            )
        elif self.config.model_type == "lr":
            return LogisticRegression(
                C=self.config.lr_c,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
        else:
            raise ValueError(f"Unknown model_type: {self.config.model_type}")

    def _get_target_col(self) -> str:
        if self.config.target == RiskTarget.NEGATIVE_PRICE:
            return NEGATIVE_PRICE_COL
        elif self.config.target == RiskTarget.LOW_VALLEY:
            return LOW_VALLEY_COL
        else:  # COMBINED
            # Use either label as positive
            return NEGATIVE_PRICE_COL  # handled in _prepare_target

    def _prepare_target(self, df: pd.DataFrame) -> pd.Series:
        if self.config.target == RiskTarget.COMBINED:
            neg = df.get(NEGATIVE_PRICE_COL, 0)
            low = df.get(LOW_VALLEY_COL, 0)
            return ((neg.astype(int) + low.astype(int)) > 0).astype(int)
        return df[self._get_target_col()].astype(int)

    def fit(
        self,
        df: pd.DataFrame,
        label_df: Optional[pd.DataFrame] = None,
    ) -> "NegativeRiskModel":
        """Fit the risk model on historical data.

        Args:
            df: Raw historical DataFrame (features engineered internally).
            label_df: Optional pre-computed label DataFrame.
                      If None, labels must already be in df (before feature engineering).

        Returns:
            Self for chaining.
        """
        # Engineer features
        feat_df = engineer_negative_price_features(df)

        # Get feature columns
        self._feature_cols = select_feature_columns(
            feat_df, extra_cols=self.config.feature_extra,
        )

        # Remove NaN rows
        train_df = feat_df[self._feature_cols].copy()
        if label_df is not None:
            y = self._prepare_target(label_df)
        else:
            # Labels must be in original df before engineer (which drops leakage cols)
            y = self._prepare_target(df)

        mask = train_df.notna().all(axis=1) & y.notna()
        X_train = train_df[mask].values.astype(np.float32)
        y_train = y[mask].values.astype(int)

        if len(X_train) < 10:
            warnings.warn(f"Only {len(X_train)} training samples — risk model may be unreliable")
            self._fitted = False
            return self

        self._model = self._build_model()
        self._model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict risk probabilities for given data.

        Args:
            df: DataFrame with required features.

        Returns:
            Array of probabilities (shape: n_samples,) for the positive class.
        """
        if not self._fitted or self._model is None:
            return np.zeros(len(df))

        feat_df = engineer_negative_price_features(df)
        available = [c for c in self._feature_cols if c in feat_df.columns]
        missing = set(self._feature_cols) - set(available)
        if missing:
            warnings.warn(f"Missing feature columns: {missing}. Filling with 0.")

        X = np.zeros((len(feat_df), len(self._feature_cols)), dtype=np.float32)
        for i, col in enumerate(self._feature_cols):
            if col in feat_df.columns:
                X[:, i] = feat_df[col].fillna(0).values
            # else stays 0

        probas = self._model.predict_proba(X)
        if probas.shape[1] >= 2:
            return probas[:, 1]
        return probas[:, 0]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_columns(self) -> list[str]:
        return list(self._feature_cols)


def fit_risk_model(
    df: pd.DataFrame,
    config: Optional[NegativeRiskConfig] = None,
) -> NegativeRiskModel:
    """Convenience function to fit a negative risk model."""
    model = NegativeRiskModel(config)
    model.fit(df)
    return model
