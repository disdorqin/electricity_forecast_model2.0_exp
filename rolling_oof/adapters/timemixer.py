# -*- coding: utf-8 -*-
"""TimeMixer 模型的 rolling-origin 适配器。

支持三种滚动模式：
- window_once: 训练一次，预测整个 fold（基准对比）
- block: 每 block_days 训练一次，预测一个 block
- daily: 每天训练一次，逐日预测（最严格）

推荐默认使用 daily 模式以保证严格性。
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# TimeMixer 子模块路径
_TIMEMIXER_ROOT = _PROJECT_ROOT / "TimeMixer"
if str(_TIMEMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(_TIMEMIXER_ROOT))


class TimeMixerRollingAdapter(BaseRollingAdapter):
    """TimeMixer rolling-origin 适配器。"""

    model_name: str = "timemixer"
    device_type: str = "gpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """TimeMixer 对单个 fold 执行训练+预测。

        根据 rolling_mode 参数选择训练模式。
        """
        rolling_mode = kwargs.get("rolling_mode", "daily")
        block_days = kwargs.get("block_days", 7)

        logger.info(
            "[timemixer/%s] fold %d: mode=%s, %s -> %s (cutoff=%s)",
            task,
            fold_spec.fold_id,
            rolling_mode,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        try:
            result_df = _run_timemixer_fold(
                data_path=data_path,
                fold_spec=fold_spec,
                task=task,
                rolling_mode=rolling_mode,
                block_days=block_days,
            )

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
                "[timemixer/%s] fold %d failed: %s",
                task,
                fold_spec.fold_id,
                e,
                exc_info=True,
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )


def _run_timemixer_fold(
    data_path: str,
    fold_spec: FoldSpec,
    task: str,
    rolling_mode: str = "daily",
    block_days: int = 7,
) -> pd.DataFrame | None:
    """对 TimeMixer 执行单个 fold。

    通过构造 RunConfig 调用 run_monthly_reproduction。
    """
    from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction

    # 映射 rolling_mode 到 TimeMixer 的 training_mode
    # "window_once" -> "rolling" (原逻辑)
    # "block" -> "block" (新增)
    # "daily" -> "daily" (新增)
    mode_map = {
        "window_once": "rolling",
        "block": "block",
        "daily": "daily",
    }
    tm_mode = mode_map.get(rolling_mode, "daily")

    run_cfg = RunConfig(
        data_path=data_path,
        output_dir=f"oof_runs/timemixer_temp/fold_{fold_spec.fold_id}_{task}",
        month=fold_spec.target_month,
        test_start=fold_spec.test_start.isoformat(),
        test_end_exclusive=(fold_spec.test_end + timedelta(days=1)).isoformat(),
        training_mode=tm_mode,
        block_days=block_days,
        segment_training=True,
        segment_count=3,
    )

    result = run_monthly_reproduction(run_cfg)

    if result is None:
        return None

    # 从输出中提取 long-table
    # run_monthly_reproduction 返回 dict，包含 da_predictions, rt_predictions 等
    pred_key = "da_predictions" if task == "dayahead" else "rt_predictions"
    predictions = result.get(pred_key)

    if predictions is None:
        return None

    if isinstance(predictions, pd.DataFrame):
        return predictions

    # predictions 可能是 list[DataFrame]（segment_training=True 时）
    if isinstance(predictions, list):
        return pd.concat(predictions, axis=0) if predictions else None

    return None
