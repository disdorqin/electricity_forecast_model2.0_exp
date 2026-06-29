# -*- coding: utf-8 -*-
"""TimeMixer 模型的 rolling-origin 适配器。

支持四种滚动模式：
- window_once: 训练一次，预测整个 fold（基准对比）
- block: 每 block_days 训练一次，预测一个 block
- daily: 每天训练一次，逐日预测（最严格）
- online: base train + online update（推荐，最快）

推荐默认使用 online 模式。
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
        online 模式: 一次调用完成 base train + 所有 block 的 predict+update。
        """
        rolling_mode = kwargs.get("rolling_mode", "online")
        block_days = kwargs.get("block_days", 3)

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
                training_months=kwargs.get("training_months", 6),
                checkpoint_dir=kwargs.get("checkpoint_dir"),
                online_epochs=kwargs.get("online_epochs", 3),
                online_lr=kwargs.get("online_lr"),
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
    rolling_mode: str = "online",
    block_days: int = 3,
    training_months: int = 6,
    checkpoint_dir: str | None = None,
    online_epochs: int = 3,
    online_lr: float | None = None,
) -> pd.DataFrame | None:
    """对 TimeMixer 执行单个 fold。

    通过构造 RunConfig 调用 run_monthly_reproduction。
    """
    logger.debug("[timemixer fold %d] About to import repro_pipeline...", fold_spec.fold_id)
    from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction
    logger.debug("[timemixer fold %d] Import successful, setting env vars...", fold_spec.fold_id)

    # Windows 下多进程 DataLoader 不稳定，强制单进程 (0)
    import os as _os
    _os.environ["OPTIM_NUM_WORKERS"] = "0"
    _os.environ["OPTIM_PIN_MEMORY"] = "0"

    # 映射 rolling_mode 到 TimeMixer 的 training_mode
    mode_map = {
        "window_once": "rolling",
        "block": "block",
        "daily": "daily",
        "online": "online",
    }
    tm_mode = mode_map.get(rolling_mode, "online")

    output_dir = f"oof_runs/timemixer_temp/fold_{fold_spec.fold_id}_{task}"

    run_cfg = RunConfig(
        data_path=data_path,
        output_dir=output_dir,
        month=fold_spec.target_month,
        test_start=fold_spec.test_start.isoformat(),
        test_end_exclusive=(fold_spec.test_end + timedelta(days=1)).isoformat(),
        training_mode=tm_mode,
        block_days=block_days,
        train_months=training_months,
        checkpoint_dir=checkpoint_dir,
        online_epochs=online_epochs,
        online_lr=online_lr,
    )

    logger.debug("[timemixer fold %d] About to call run_monthly_reproduction (mode=%s)...", fold_spec.fold_id, tm_mode)
    result = run_monthly_reproduction(run_cfg)
    logger.debug("[timemixer fold %d] run_monthly_reproduction returned, type=%s", fold_spec.fold_id, type(result))

    if result is None:
        return None

    pred_key = "da_predictions" if task == "dayahead" else "rt_predictions"
    predictions = result.get(pred_key)

    if predictions is None:
        return None

    if isinstance(predictions, pd.DataFrame):
        return predictions

    if isinstance(predictions, list):
        return pd.concat(predictions, axis=0) if predictions else None

    return None
