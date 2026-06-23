"""
Improved meta-learner: GradientBoosting with hour-level features and cross-validation.
Ensures fusion is never worse than the best individual model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit

from .contracts import build_wide_frame
from .metrics import smape_floor50


BASE_ID_COLS = ["task", "target_day", "ds", "period", "hour_business", "y_true"]


@dataclass
class SegmentMetaModel:
    task: str
    period: str
    model_names: list[str]
    imputer: SimpleImputer
    estimator: GradientBoostingRegressor
    best_single_model: str = ""
    best_single_smape: float = 0.0
    fit_smape: float = 0.0
    cv_smape: float = 0.0
    use_learner: bool = True


def _build_features(group: pd.DataFrame, model_cols: list[str]) -> pd.DataFrame:
    features = group[model_cols].copy()
    features["hour_sin"] = np.sin(2 * np.pi * group["hour_business"].values / 24)
    features["hour_cos"] = np.cos(2 * np.pi * group["hour_business"].values / 24)

    if "ds" in group.columns:
        ds = pd.to_datetime(group["ds"])
        features["day_of_week"] = ds.dt.dayofweek.values
        features["is_weekend"] = (ds.dt.dayofweek >= 5).astype(int).values
        features["month"] = ds.dt.month.values

    for col in model_cols:
        if col in group.columns:
            features[f"{col}_lag1"] = group[col].shift(1).fillna(group[col]).values

    for i, col_a in enumerate(model_cols):
        for col_b in model_cols[i + 1:]:
            if col_a in group.columns and col_b in group.columns:
                features[f"{col_a}_x_{col_b}"] = group[col_a].values * group[col_b].values
    return features


def _feature_names(model_cols: list[str]) -> list[str]:
    names = list(model_cols)
    names.extend(["hour_sin", "hour_cos", "day_of_week", "is_weekend", "month"])
    for col in model_cols:
        names.append(f"{col}_lag1")
    for i, col_a in enumerate(model_cols):
        for col_b in model_cols[i + 1:]:
            names.append(f"{col_a}_x_{col_b}")
    return names


def _find_best_single_model(clean_group: pd.DataFrame, model_cols: list[str]) -> tuple[str, float]:
    y_true = clean_group["y_true"].to_numpy(dtype=float)
    best_name, best_smape = "", float("inf")
    for col in model_cols:
        if col not in clean_group.columns:
            continue
        preds = clean_group[col].to_numpy(dtype=float)
        valid = ~(np.isnan(y_true) | np.isnan(preds))
        if valid.sum() == 0:
            continue
        s = smape_floor50(y_true[valid], preds[valid])
        if s < best_smape:
            best_smape = s
            best_name = col
    return best_name, best_smape


def fit_meta_learners_from_long_table(
    df: pd.DataFrame,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    min_samples_leaf: int = 10,
    cv_folds: int = 3,
) -> tuple[dict[tuple[str, str], SegmentMetaModel], pd.DataFrame]:
    wide = build_wide_frame(df)
    model_cols = [column for column in wide.columns if column not in BASE_ID_COLS]
    if not model_cols:
        raise ValueError("No model columns found after pivoting prediction table")

    models: dict[tuple[str, str], SegmentMetaModel] = {}
    report_rows: list[dict[str, object]] = []

    for (task, period), group in wide.groupby(["task", "period"], sort=True):
        active_model_cols = [column for column in model_cols if column in group.columns and group[column].notna().any()]
        clean_group = group.dropna(subset=["y_true"]).copy()
        if not active_model_cols or clean_group.empty:
            continue

        best_name, best_smape = _find_best_single_model(clean_group, active_model_cols)

        feature_frame = _build_features(clean_group, active_model_cols)
        fnames = _feature_names(active_model_cols)
        imputer = SimpleImputer(strategy="median")
        x_all = imputer.fit_transform(feature_frame[fnames])
        y_all = clean_group["y_true"].to_numpy(dtype=float)

        if len(clean_group) >= cv_folds * 2:
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            cv_smapes = []
            for train_idx, val_idx in tscv.split(x_all):
                x_tr, x_vl = x_all[train_idx], x_all[val_idx]
                y_tr, y_vl = y_all[train_idx], y_all[val_idx]
                est = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                )
                est.fit(x_tr, y_tr)
                y_hat = est.predict(x_vl)
                cv_smapes.append(smape_floor50(y_vl, y_hat))
            cv_smape = float(np.mean(cv_smapes))
        else:
            cv_smape = float("inf")

        estimator = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
        )
        estimator.fit(x_all, y_all)
        y_fit = estimator.predict(x_all)
        fit_smape = float(smape_floor50(y_all, y_fit))

        use_learner = cv_smape < best_smape

        segment_model = SegmentMetaModel(
            task=str(task),
            period=str(period),
            model_names=active_model_cols,
            imputer=imputer,
            estimator=estimator,
            best_single_model=best_name,
            best_single_smape=best_smape,
            fit_smape=fit_smape,
            cv_smape=cv_smape if cv_smape != float("inf") else fit_smape,
            use_learner=use_learner,
        )
        models[(str(task), str(period))] = segment_model

        importance_map = dict(zip(fnames, estimator.feature_importances_.tolist()))
        report_rows.append(
            {
                "task": task,
                "period": period,
                "sample_count": int(len(clean_group)),
                "cv_smape": cv_smape,
                "fit_smape": fit_smape,
                "best_single_model": best_name,
                "best_single_smape": best_smape,
                "use_learner": use_learner,
                **{f"importance_{name}": float(importance_map.get(name, 0.0)) for name in fnames},
            }
        )

    return models, pd.DataFrame(report_rows)


def apply_meta_learners(
    df: pd.DataFrame,
    models: dict[tuple[str, str], SegmentMetaModel],
    *,
    task: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    from .contracts import infer_period as _infer_period

    work = df.copy()
    work["ds"] = pd.to_datetime(work["ds"], errors="coerce")

    if "model_name" in work.columns and "y_pred" in work.columns:
        work["hour_business"] = work["ds"].dt.hour.replace({0: 24}).astype(int)
        work["period"] = work["hour_business"].map(_infer_period)
        work["target_day"] = work["ds"].dt.normalize().where(
            work["ds"].dt.hour != 0, work["ds"].dt.normalize() - pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
        work["task"] = task

        id_cols = ["task", "target_day", "ds", "period", "hour_business"]
        if "y_true" in work.columns:
            id_cols_with_truth = id_cols + ["y_true"]
        else:
            id_cols_with_truth = id_cols

        truth_df = work[id_cols_with_truth].drop_duplicates(subset=id_cols)
        wide_pred = work.pivot_table(
            index=id_cols, columns="model_name", values="y_pred", aggfunc="last"
        ).reset_index()
        wide_pred.columns.name = None
        work = truth_df.merge(wide_pred, on=id_cols, how="inner")

    if "target_day" not in work.columns:
        work["hour_business"] = work["ds"].dt.hour.replace({0: 24}).astype(int)
        work["period"] = work["hour_business"].map(_infer_period)
        work["target_day"] = work["ds"].dt.normalize().where(
            work["ds"].dt.hour != 0, work["ds"].dt.normalize() - pd.Timedelta(days=1)
        ).dt.strftime("%Y-%m-%d")
    if "period" not in work.columns:
        work["hour_business"] = work["ds"].dt.hour.replace({0: 24}).astype(int)
        work["period"] = work["hour_business"].map(_infer_period)
    if "task" not in work.columns:
        work["task"] = task

    task_days = pd.to_datetime(work["target_day"])
    task_df = work[(task_days >= pd.Timestamp(test_start)) & (task_days <= pd.Timestamp(test_end))].copy()
    if task_df.empty:
        raise RuntimeError(f"No test rows found for task={task}.")

    model_cols = [c for c in task_df.columns if c not in ["task", "target_day", "ds", "period", "hour_business", "y_true", "y_pred", "y_fused"]]

    fused_parts: list[pd.DataFrame] = []
    for period, group in task_df.groupby("period", sort=True):
        key = (task, period)
        if key not in models:
            logger.warning("Missing meta learner for task=%s, period=%s — using equal weights", task, period)
            if model_cols:
                out = group.copy()
                out["y_fused"] = group[model_cols].mean(axis=1)
                fused_parts.append(out)
            continue
        meta_model = models[key]

        available_models = [m for m in meta_model.model_names if m in group.columns and group[m].notna().any()]

        if not available_models:
            out = group.copy()
            out["y_fused"] = np.nan
            fused_parts.append(out)
            continue

        best_name = meta_model.best_single_model
        if best_name not in available_models:
            best_name = available_models[0]
        best_preds = group[best_name].to_numpy(dtype=float) if best_name in group.columns else None

        if not meta_model.use_learner:
            y_pred = best_preds if best_preds is not None else np.full(len(group), np.nan)
            out = group.copy()
            out["y_fused"] = y_pred
            fused_parts.append(out)
            continue

        if meta_model.cv_smape < meta_model.best_single_smape and len(available_models) >= max(1, len(meta_model.model_names) * 0.5):
            feature_frame = _build_features(group, available_models)
            fnames = _feature_names(available_models)
            x_test = meta_model.imputer.transform(feature_frame[fnames])
            gbm_preds = meta_model.estimator.predict(x_test)
        else:
            gbm_preds = None

        if gbm_preds is not None:
            y_pred = gbm_preds
        elif best_preds is not None:
            y_pred = best_preds
        else:
            y_pred = np.full(len(group), np.nan)

        out = group.copy()
        out["y_fused"] = y_pred
        fused_parts.append(out)

    return pd.concat(fused_parts, ignore_index=True).sort_values(["target_day", "hour_business"]).reset_index(drop=True)
