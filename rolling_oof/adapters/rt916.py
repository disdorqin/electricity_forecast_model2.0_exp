# -*- coding: utf-8 -*-
"""RT916/SpikeFusionNet 模型的 rolling-origin 适配器。

关键修复：
1. 强制 retrain_daily=True（daily_walk_forward）
2. 训练时调用 apply_asof_cutoff_for_inference 防止 lag 特征泄露
3. 修复 00:00 时间对齐（统一为 business_hour=24）
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


TARGET_MAP = {"dayahead": "日前电价", "realtime": "实时电价"}


class RT916RollingAdapter(BaseRollingAdapter):
    """RT916 rolling-origin 适配器（修复版）。"""

    model_name: str = "rt916"
    device_type: str = "gpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """RT916 对单个 fold 执行训练+预测。

        强制使用 retrain_daily=True 实现 daily_walk_forward。
        """
        logger.info(
            "[rt916/%s] fold %d: daily walk-forward %s -> %s (cutoff=%s)",
            task,
            fold_spec.fold_id,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        try:
            # 设置环境变量传递训练窗口参数
            os.environ["SPIKE_TRAIN_START_DATE"] = fold_spec.train_start.isoformat()
            os.environ["SPIKE_TRAIN_END_DATE"] = fold_spec.train_end.isoformat()
            os.environ["SPIKE_RETRAIN_DAILY"] = "1"

            result_df = _run_rt916_fold(data_path, fold_spec, task)
            if result_df is None or result_df.empty:
                return FoldResult(
                    fold_id=fold_spec.fold_id,
                    model_name=self.model_name,
                    task=task,
                    fold_spec=fold_spec,
                    success=False,
                    error_message="No predictions generated",
                )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                predictions_df=result_df,
                success=True,
            )
        except Exception as e:
            logger.error(
                "[rt916/%s] fold %d failed: %s", task, fold_spec.fold_id, e, exc_info=True
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )
        finally:
            # 清理环境变量
            for key in (
                "SPIKE_TRAIN_START_DATE",
                "SPIKE_TRAIN_END_DATE",
                "SPIKE_RETRAIN_DAILY",
            ):
                os.environ.pop(key, None)


def _run_rt916_fold(
    data_path: str, fold_spec: FoldSpec, task: str
) -> pd.DataFrame | None:
    """运行 RT916 fold。"""
    from RT916_SpikeFusionNet.pipeline import ModelPipeline

    pipeline = ModelPipeline()

    result = pipeline.predict_range(
        target=task,
        data_path=data_path,
        start=fold_spec.test_start.isoformat(),
        end=fold_spec.test_end.isoformat(),
        predict_date=None,
        output_root="oof_runs/rt916_temp",
        retrain_daily=True,  # 强制 daily_walk_forward
        asof_hour=15,
    )

    if result is not None and hasattr(result, "frame"):
        df = result.frame
        # 修复时间对齐：确保 00:00 映射为 business_hour=24
        if "ds" in df.columns and "hour_business" not in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
            df["hour"] = df["ds"].dt.hour
            df["hour_business"] = df["hour"].apply(lambda h: 24 if h == 0 else h)
        return df
    return result
