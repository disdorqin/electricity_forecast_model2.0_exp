# -*- coding: utf-8 -*-
"""阶段 C: 最终陪跑预测模块。

用截止到最新日期的数据训练所有基础模型，预测明天的 24 小时电价。
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from rolling_oof.contracts import RollingOriginConfig
from rolling_oof.output_layout import OofRunLayout

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def run_escort_prediction(
    config: RollingOriginConfig,
    layout: OofRunLayout,
    target_date: str,
) -> pd.DataFrame:
    """最终陪跑预测：用最新可用数据训练所有模型，预测 target_date。

    Parameters
    ----------
    config : RollingOriginConfig
        编排器配置。
    layout : OofRunLayout
        输出目录布局。
    target_date : str
        预测目标日，格式 YYYY-MM-DD。

    Returns
    -------
    pd.DataFrame
        标准 long-table 格式的预测结果。
    """
    from runners.registry import get_model_pipeline
    from pipelines.base import BaseModelPipeline

    target_dt = pd.Timestamp(target_date)
    train_start = target_dt - pd.DateOffset(months=config.training_months)

    # 使用 models 来区分日前/实时
    all_frames: list[pd.DataFrame] = []

    for model_name in config.models_list:
        try:
            pipeline = get_model_pipeline(model_name)
        except KeyError:
            logger.warning("[escort] Unknown model: %s", model_name)
            continue

        for task in config.tasks_list:
            logger.info("[escort] %s/%s: predicting %s", model_name, task, target_date)
            try:
                result = _predict_one_model(
                    pipeline=pipeline,
                    model_name=model_name,
                    task=task,
                    train_start=train_start,
                    target_date=target_date,
                    data_path=config.data_path,
                    **{
                        "training_months": config.training_months,
                        "val_ratio": config.val_ratio,
                        "output_root": f"daily_runs/{target_date}",
                    },
                )
                if result is not None:
                    all_frames.append(result)
            except Exception as e:
                logger.error("[escort] %s/%s failed: %s", model_name, task, e, exc_info=True)

    if not all_frames:
        logger.warning("[escort] No predictions generated for %s", target_date)
        return pd.DataFrame()

    # 合并
    escort_df = pd.concat(all_frames, axis=0, ignore_index=True)

    # 保存
    layout.escort_dir.mkdir(parents=True, exist_ok=True)
    escort_df.to_csv(layout.escort_long_path(target_date), index=False)

    # 按 task 分别保存
    for task in config.tasks_list:
        task_df = escort_df[escort_df["task"] == task] if "task" in escort_df.columns else None
        if task_df is not None and not task_df.empty:
            task_df.to_csv(layout.escort_task_path(target_date, task), index=False)

    logger.info("[escort] Phase C complete: %d rows for %s", len(escort_df), target_date)
    return escort_df


def _predict_one_model(
    pipeline,
    model_name: str,
    task: str,
    train_start: pd.Timestamp,
    target_date: str,
    data_path: str,
    **kwargs,
) -> Optional[pd.DataFrame]:
    """使用现有 pipeline 接口预测单个模型的单日电价。

    优先使用 predict_range(kwargs) 接口，fallback 使用 predict(kwargs)。
    """
    try:
        predict_kwargs = {
            "target": task,
            "data_path": data_path,
            "predict_date": target_date,
            "start": target_date,
            "end": target_date,
            **kwargs,
        }

        if hasattr(pipeline, "predict_range"):
            result = pipeline.predict_range(**predict_kwargs)
        elif hasattr(pipeline, "predict"):
            result = pipeline.predict(**predict_kwargs)
        else:
            logger.warning("[escort] %s: no predict method", model_name)
            return None

        if result is None:
            return None

        # 提取 DataFrame
        df = None
        if hasattr(result, "frame"):
            df = result.frame
        elif isinstance(result, pd.DataFrame):
            df = result

        if df is None or df.empty:
            return None

        # 标准化为 long-table 格式
        import numpy as np
        from datetime import datetime as dt

        target_dt = pd.Timestamp(target_date)

        # 确保 ds 列
        if "ds" not in df.columns and "时刻" in df.columns:
            df["ds"] = pd.to_datetime(df["时刻"])
        if "ds" not in df.columns:
            df["ds"] = None

        # 提取预测列
        if "y_pred" not in df.columns:
            for col in ["预测值", "prediction", "y_pred"]:
                if col in df.columns:
                    df["y_pred"] = pd.to_numeric(df[col], errors="coerce")
                    break
            else:
                # fallback: 使用第一列数值型数据
                for col in df.columns:
                    if col not in ("task", "model_name", "ds", "时刻", "target_day"):
                        df["y_pred"] = pd.to_numeric(df[col], errors="coerce")
                        break

        # 标准化字段
        df["task"] = task
        df["model_name"] = model_name
        df["fold_id"] = -1  # escort 无 fold
        df["train_start"] = train_start.strftime("%Y-%m-%d")
        df["train_end"] = (target_dt - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        df["test_start"] = target_date
        df["test_end"] = target_date
        df["target_day"] = target_date
        df["source"] = "escort_forecast"
        df["run_mode"] = "escort"
        df["created_at"] = dt.now().isoformat()

        # business_day / period / hour_business
        if "ds" in df.columns and df["ds"].notna().any():
            ds_dt = pd.to_datetime(df["ds"])
            df["hour_business"] = ds_dt.apply(
                lambda t: 24 if t.hour == 0 else t.hour
            )
            df["business_day"] = ds_dt.apply(
                lambda t: (t - pd.Timedelta(days=1) if t.hour == 0 else t).strftime("%Y-%m-%d")
            )
            from rolling_oof.contracts import assign_period
            df["period"] = df["hour_business"].apply(assign_period)
        else:
            df["business_day"] = target_date
            df["hour_business"] = None
            df["period"] = None

        # y_true: escort 时不填（未来日无真实值）
        if "y_true" not in df.columns:
            df["y_true"] = None

        # 确保所有 contract 列存在
        from rolling_oof.contracts import LONG_TABLE_COLUMNS
        for col in LONG_TABLE_COLUMNS:
            if col not in df.columns:
                df[col] = None

        return df[LONG_TABLE_COLUMNS]

    except Exception as e:
        logger.error("[escort] %s/%s: %s", model_name, task, e)
        return None
