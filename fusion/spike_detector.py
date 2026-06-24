from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


def _safe_rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).mean()


def _safe_rolling_std(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).std()


def _safe_rolling_quantile(series: pd.Series, window: int, q: float, min_periods: int = 1) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).quantile(q)


@dataclass
class SpikeDetectorConfig:
    spike_threshold: float = 100.0
    extreme_high: float = 800.0
    extreme_low: float = -80.0
    prob_threshold: float = 0.35
    n_estimators: int = 300
    max_depth: int = 5
    learning_rate: float = 0.05
    min_samples_leaf: int = 15
    momentum_factor: float = 0.85
    stage2_enabled: bool = True
    gray_low: float = 0.25
    gray_high: float = 0.65


class SpikeDetector:
    def __init__(self, config: SpikeDetectorConfig | None = None):
        self.config = config or SpikeDetectorConfig()
        self._stage1_model: GradientBoostingClassifier | None = None
        self._stage2_model: GradientBoostingClassifier | None = None
        self._imputer1: SimpleImputer | None = None
        self._imputer2: SimpleImputer | None = None
        self._feature_names: list[str] = []
        self._stage2_feature_names: list[str] = []
        self._hour_stats: dict[int, float] = {}
        self._is_fitted = False

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        ds = pd.to_datetime(work["时刻"])

        work["hour"] = ds.dt.hour.replace({0: 24})
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["weekday"] = ds.dt.dayofweek
        work["is_weekend"] = (work["weekday"] >= 5).astype(int)
        work["month"] = ds.dt.month
        work["day_of_month"] = ds.dt.day
        work["is_month_start"] = (work["day_of_month"] <= 3).astype(int)
        work["is_month_end"] = (work["day_of_month"] >= 28).astype(int)
        work["is_peak_hour"] = work["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
        work["is_solar_hour"] = work["hour"].isin(range(9, 17)).astype(int)

        da = pd.to_numeric(work.get("日前电价", pd.Series(dtype=float)), errors="coerce")
        rt = pd.to_numeric(work.get("实时电价", pd.Series(dtype=float)), errors="coerce")
        work["da"] = da
        work["rt"] = rt

        for col in ["直调负荷预测值", "新能源总加预测值", "竞价空间预测值"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        work["da_prev1"] = da.shift(1)
        work["da_prev24"] = da.shift(24)
        work["da_prev48"] = da.shift(48)
        work["da_prev168"] = da.shift(168)
        work["rt_prev1"] = rt.shift(1)
        work["rt_prev24"] = rt.shift(24)

        work["da_ramp_1"] = da.diff(1)
        work["da_ramp_24"] = da - da.shift(24)
        work["da_ma_24"] = _safe_rolling_mean(da, 24)
        work["da_std_24"] = _safe_rolling_std(da, 24)
        work["da_q95_168"] = _safe_rolling_quantile(da, 168, 0.95)
        work["da_q05_168"] = _safe_rolling_quantile(da, 168, 0.05)
        work["da_ma_168"] = _safe_rolling_mean(da, 168)

        work["spread"] = rt - da
        work["abs_spread"] = work["spread"].abs()
        work["spread_ma_24"] = _safe_rolling_mean(work["spread"], 24)
        work["spread_std_24"] = _safe_rolling_std(work["spread"], 24)
        work["spread_ramp_1"] = work["spread"].diff(1)

        work["da_volatility"] = work["da_std_24"] / (work["da_ma_24"].abs() + 1e-5)
        work["da_position"] = (da - work["da_q05_168"]) / (work["da_q95_168"] - work["da_q05_168"] + 1e-5)
        work["da_position"] = work["da_position"].clip(0, 2)

        if "竞价空间预测值" in work.columns:
            bidding = work["竞价空间预测值"]
            work["bidding_lag_1"] = bidding.shift(1)
            work["bidding_lag_24"] = bidding.shift(24)
            work["bidding_ramp_1"] = bidding.diff(1)
            work["bidding_ma_24"] = _safe_rolling_mean(bidding, 24)
            work["bidding_negative"] = (bidding < 0).astype(int)
            work["bidding_extreme"] = (bidding > 23000).astype(int)

        if "直调负荷预测值" in work.columns:
            load = work["直调负荷预测值"]
            work["load_ma_24"] = _safe_rolling_mean(load, 24)
            work["load_ramp_1"] = load.diff(1)

        if "新能源总加预测值" in work.columns:
            renewable = work["新能源总加预测值"]
            work["renewable_ma_24"] = _safe_rolling_mean(renewable, 24)
            work["renewable_ramp_1"] = renewable.diff(1)
            if "直调负荷预测值" in work.columns:
                work["renewable_ratio"] = renewable / (work["直调负荷预测值"] + 1e-5)

        return work

    def _get_stage1_features(self, work: pd.DataFrame) -> list[str]:
        features = [
            "hour_sin", "hour_cos", "weekday", "is_weekend", "month",
            "is_peak_hour", "is_solar_hour",
            "da", "da_prev1", "da_prev24", "da_prev48", "da_prev168",
            "da_ramp_1", "da_ramp_24",
            "da_ma_24", "da_std_24", "da_ma_168",
            "da_volatility", "da_position",
            "rt_prev1", "rt_prev24",
            "spread_ma_24", "spread_std_24",
        ]
        extra = [
            "bidding_lag_1", "bidding_lag_24", "bidding_ramp_1",
            "bidding_ma_24", "bidding_negative", "bidding_extreme",
            "load_ma_24", "load_ramp_1",
            "renewable_ma_24", "renewable_ramp_1", "renewable_ratio",
        ]
        for f in extra:
            if f in work.columns:
                features.append(f)
        return features

    def _get_stage2_features(self, work: pd.DataFrame) -> list[str]:
        return self._get_stage1_features(work) + ["p1_prob"]

    def fit(self, history_df: pd.DataFrame) -> SpikeDetector:
        work = self._build_features(history_df)

        da = pd.to_numeric(work["日前电价"], errors="coerce")
        rt = pd.to_numeric(work["实时电价"], errors="coerce")
        spread = rt - da

        spike_label = (
            (spread.abs() > self.config.spike_threshold)
            | (da > self.config.extreme_high)
            | (rt > self.config.extreme_high)
            | (da < self.config.extreme_low)
            | (rt < self.config.extreme_low)
        ).astype(int)

        work["spike_label"] = spike_label

        for h in range(1, 25):
            mask = work["hour"] == h
            if mask.sum() > 0:
                self._hour_stats[h] = float(spike_label[mask].mean())

        self._feature_names = self._get_stage1_features(work)
        valid_mask = work[self._feature_names + ["spike_label"]].notna().all(axis=1)
        X = work.loc[valid_mask, self._feature_names].values
        y = work.loc[valid_mask, "spike_label"].values

        if len(X) < 200:
            logger.warning("SpikeDetector: only %d valid samples", len(X))
            self._is_fitted = True
            return self

        self._imputer1 = SimpleImputer(strategy="median")
        X_imp = self._imputer1.fit_transform(X)

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        spw = (neg_count / max(pos_count, 1)) * 0.4

        self._stage1_model = GradientBoostingClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            subsample=0.8,
            random_state=42,
        )
        self._stage1_model.fit(X_imp, y)

        p1_probs = self._stage1_model.predict_proba(X_imp)[:, 1]

        if self.config.stage2_enabled:
            gray_mask = (p1_probs >= self.config.gray_low) & (p1_probs <= self.config.gray_high)
            if gray_mask.sum() > 100:
                work_gray = work.loc[valid_mask].copy()
                work_gray["p1_prob"] = p1_probs
                self._stage2_feature_names = self._get_stage2_features(work_gray)
                X2 = work_gray.loc[gray_mask, self._stage2_feature_names].values
                y2 = work_gray.loc[gray_mask, "spike_label"].values

                self._imputer2 = SimpleImputer(strategy="median")
                X2_imp = self._imputer2.fit_transform(X2)

                self._stage2_model = GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=42,
                )
                self._stage2_model.fit(X2_imp, y2)
                logger.info("Stage2 trained on %d gray-zone samples", gray_mask.sum())

        train_pred = self._stage1_model.predict(X_imp)
        train_acc = (train_pred == y).mean()
        logger.info(
            "SpikeDetector fitted: %d samples, spike_rate=%.2f%%, train_acc=%.2f%%",
            len(y), y.mean() * 100, train_acc * 100,
        )
        self._is_fitted = True
        return self

    def predict_spike_probability(self, df: pd.DataFrame) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("SpikeDetector not fitted.")

        work = self._build_features(df)

        if self._stage1_model is not None and self._imputer1 is not None:
            X = work[self._feature_names].values
            valid_mask = ~np.isnan(X).all(axis=1)
            probs = np.full(len(X), 0.5)
            if valid_mask.sum() > 0:
                X_imp = self._imputer1.transform(X[valid_mask])
                p1_probs = self._stage1_model.predict_proba(X_imp)[:, 1]
                probs[valid_mask] = p1_probs

                if self.config.stage2_enabled and self._stage2_model is not None and self._imputer2 is not None:
                    work_valid = work[valid_mask].copy()
                    work_valid["p1_prob"] = p1_probs
                    gray_mask = (p1_probs >= self.config.gray_low) & (p1_probs <= self.config.gray_high)
                    if gray_mask.sum() > 0:
                        X2 = work_valid.loc[gray_mask, self._stage2_feature_names].values
                        X2_imp = self._imputer2.transform(X2)
                        p2_probs = self._stage2_model.predict_proba(X2_imp)[:, 1]
                        probs[valid_mask][np.where(gray_mask)[0]] = p2_probs

            ds = pd.to_datetime(df["时刻"])
            hours = ds.dt.hour.replace({0: 24}).astype(int)
            for i in range(len(probs)):
                h = hours.iloc[i] if i < len(hours) else 12
                if probs[i] < 0.01:
                    probs[i] = max(probs[i], self._hour_stats.get(h, 0.1) * 0.3)

            return pd.Series(probs, index=df.index, name="spike_prob")
        else:
            ds = pd.to_datetime(df["时刻"])
            hours = ds.dt.hour.replace({0: 24}).astype(int)
            probs = [self._hour_stats.get(h, 0.1) for h in hours]
            return pd.Series(probs, index=df.index, name="spike_prob")

    def apply_momentum(self, probs: pd.Series) -> pd.Series:
        result = probs.copy()
        for i in range(1, len(result)):
            if result.iloc[i - 1] >= self.config.prob_threshold:
                result.iloc[i] = min(result.iloc[i] * (1 + (1 - self.config.momentum_factor)), 1.0)
        return result

    def predict(self, df: pd.DataFrame, threshold: float | None = None, use_momentum: bool = True) -> pd.Series:
        probs = self.predict_spike_probability(df)
        if use_momentum:
            probs = self.apply_momentum(probs)
        t = threshold if threshold is not None else self.config.prob_threshold
        return (probs >= t).astype(int).rename("is_spike")
