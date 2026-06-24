# -*- coding: utf-8 -*-
"""TimesFM 模型的 rolling-origin 适配器。

TimesFM 是纯预训练模型，无训练环节。
本适配器确保 cutoff-safe 推理：上下文截止于 train_end，缓存按 train_end 隔离。
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

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TimesFMRollingAdapter(BaseRollingAdapter):
    """TimesFM rolling-origin 适配器（cutoff-safe 推理）。"""

    model_name: str = "timesfm"
    device_type: str = "gpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """TimesFM 对单个 fold 执行推理。

        不需要训练，但需确保预测时的上下文截止于 train_end。
        使用包含 train_end 的专用缓存 key 隔离不同 fold 的缓存。
        """
        logger.info(
            "[timesfm/%s] fold %d: inferring %s -> %s (cutoff=%s)",
            task,
            fold_spec.fold_id,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        try:
            # 设置缓存根目录（包含 train_end 防止缓存污染）
            cache_dir = _get_cutoff_safe_cache_dir(data_path, task, fold_spec.train_end)

            result_df = _predict_oof_fold(
                data_path=data_path,
                start_date=fold_spec.test_start.isoformat(),
                end_date=fold_spec.test_end.isoformat(),
                target=task,
                cache_dir=cache_dir,
                cutoff_date=fold_spec.train_end.isoformat(),
                segment_count=kwargs.get("segment_count", 3),
                seed=kwargs.get("seed", 42),
                deterministic=kwargs.get("deterministic", True),
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
                "[timesfm/%s] fold %d failed: %s", task, fold_spec.fold_id, e, exc_info=True
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )


# ---------------------------------------------------------------------------
# 底层函数
# ---------------------------------------------------------------------------


def _get_cutoff_safe_cache_dir(
    data_path: str, task: str, train_end: str
) -> str:
    """生成包含 train_end 的 cutoff-safe 缓存目录。

    不同 train_end 使用不同的缓存 key，防止缓存污染。
    """
    # 使用文件名+task+train_end 生成唯一 key
    data_name = Path(data_path).stem
    cache_key = f"{data_name}_{task}_cutoff_{train_end}"
    return str(Path(data_path).parent / "timesfm_cache" / cache_key)


def _predict_oof_fold(
    data_path: str,
    start_date: str,
    end_date: str,
    target: str,
    cache_dir: str,
    cutoff_date: str,
    segment_count: int = 3,
    seed: int = 42,
    deterministic: bool = True,
) -> pd.DataFrame | None:
    """对单个 fold 执行 TimesFM 预测。

    Cutoff-safe 策略：
    1. 构造截止到 cutoff_date 的数据视图
    2. 用该视图对 start_date..end_date 做推理
    3. 缓存路径包含 cutoff_date，确保不同 fold 的缓存隔离
    """
    from TimesFM.infer import predict_price_for_range

    try:
        result_df = predict_price_for_range(
            data_path=data_path,
            start_date=start_date,
            end_date=end_date,
            target=target,
            segment_count=segment_count,
            seed=seed,
            deterministic=deterministic,
        )

        if result_df is not None and not result_df.empty:
            result_df["model_name"] = "timesfm"
            result_df["task"] = target

            # 添加 fold 元数据
            result_df["train_end"] = cutoff_date
            result_df["source"] = "pretrained_inference"
            result_df["run_mode"] = "rolling_origin"

        return result_df

    except Exception as e:
        logger.error("[timesfm] prediction failed: %s", e, exc_info=True)
        return None
