# -*- coding: utf-8 -*-
"""SGDFNet 模型的 rolling-origin 适配器。

SGDFNet 已经是 daily_walk_forward + strict cutoff protocol，基本合格。
本适配器将其 YAML 配置接口统一映射到 FoldSpec 参数。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class SGDFNetRollingAdapter(BaseRollingAdapter):
    """SGDFNet rolling-origin 适配器。"""

    model_name: str = "sgdfnet"
    device_type: str = "cpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """SGDFNet 对单个 fold 执行训练+预测。

        SGDFNet 只在 realtime 上有效。
        将 fold_spec 参数转换为 SGDFNet 的 YAML 配置格式。
        """
        if task != "realtime":
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message="SGDFNet only supports realtime task",
            )

        logger.info(
            "[sgdfnet] fold %d: daily walk-forward %s -> %s (cutoff=%s)",
            fold_spec.fold_id,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        try:
            result_df = _run_sgdfnet_fold(data_path, fold_spec)
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
            logger.error("[sgdfnet] fold %d failed: %s", fold_spec.fold_id, e, exc_info=True)
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )


def _run_sgdfnet_fold(data_path: str, fold_spec: FoldSpec) -> pd.DataFrame | None:
    """将 FoldSpec 映射为 SGDFNet pipeline 调用。"""
    from SGDFNet.pipeline import ModelPipeline
    from datetime import date as dt_date

    pipeline = ModelPipeline()

    # SGDFNet 的 predict_range 接受 start/end 参数
    # 内部通过 _build_decision_days 生成 decision_days 列表
    result = pipeline.predict_range(
        target="realtime",
        data_path=data_path,
        start=fold_spec.test_start.isoformat(),
        end=fold_spec.test_end.isoformat(),
        predict_date=None,
        output_root="oof_runs/sgdfnet_temp",
    )

    if result is not None and hasattr(result, "frame"):
        return result.frame
    return result
