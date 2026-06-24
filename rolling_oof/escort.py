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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

        # 标准化
        df["task"] = task
        df["model_name"] = model_name
        df["source"] = "escort_forecast"
        df["run_mode"] = "escort"

        return df

    except Exception as e:
        logger.error("[escort] %s/%s: %s", model_name, task, e)
        return None
