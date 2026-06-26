# -*- coding: utf-8 -*-
"""SGDFNet 模型的 rolling-origin 适配器。

SGDFNet 已经是 daily_walk_forward + strict cutoff protocol，基本合格。
本适配器将其 YAML 配置接口统一映射到 FoldSpec 参数。

Phase 4 新增：
- 支持两种 fold 策略："10x3"（原行为）和 "3x10"（时间预算 fallback）。
- 自动策略选择：前 3 个 fold 若各超 30 分钟则切换到 3x10。
- 3x10 策略下将 10 天预测拆分到 3 天 sub-block 并标记 tap_block_id。
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 自动策略切换的全局时间阈值（秒）
_AUTO_SWITCH_THRESHOLD_SEC: float = 30 * 60  # 30 分钟


class SGDFNetRollingAdapter(BaseRollingAdapter):
    """SGDFNet rolling-origin 适配器，支持 10x3 / 3x10 fold 策略。"""

    model_name: str = "sgdfnet"
    device_type: str = "cpu"
    supported_tasks: tuple[str, ...] = ("realtime",)

    def __init__(self) -> None:
        super().__init__()
        # 类级别状态：跟踪 fold 耗时以支持 auto 策略切换
        self._fold_timings: dict[int, float] = {}  # fold_id -> elapsed seconds
        self._strategy_override: str | None = None  # "3x10" if auto-switch triggered

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """SGDFNet 对单个 fold 执行训练+预测。

        Parameters
        ----------
        fold_strategy : str
            "auto"（默认）：先尝试 10x3，前 3 fold 超时时自动切换到 3x10。
            "10x3"：强制 10 fold × 3 day。
            "3x10"：强制 3 fold × 10 day。
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

        fold_strategy: str = kwargs.get("fold_strategy", "3x10")
        effective_strategy = self._resolve_strategy(fold_strategy, fold_spec)

        logger.info(
            "[sgdfnet] fold %d: strategy=%s (requested=%s), %s -> %s (cutoff=%s)",
            fold_spec.fold_id,
            effective_strategy,
            fold_strategy,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        t0 = time.monotonic()
        try:
            if effective_strategy == "3x10":
                result = self._run_3x10_fold(task, fold_spec, data_path, **kwargs)
            else:
                result = self._run_10x3_fold(task, fold_spec, data_path, **kwargs)

            elapsed = time.monotonic() - t0
            self._record_timing(fold_spec.fold_id, elapsed)

            # 为结果添加策略标记
            if result.success and result.predictions_df is not None:
                result.predictions_df["fold_strategy"] = effective_strategy

            return result

        except Exception as e:
            elapsed = time.monotonic() - t0
            self._record_timing(fold_spec.fold_id, elapsed)
            logger.error(
                "[sgdfnet] fold %d failed: %s", fold_spec.fold_id, e, exc_info=True,
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # 策略解析
    # ------------------------------------------------------------------

    def _resolve_strategy(
        self, fold_strategy: str, fold_spec: FoldSpec,
    ) -> str:
        """确定当前 fold 实际使用的策略。"""
        if fold_strategy == "10x3":
            return "10x3"
        if fold_strategy == "3x10":
            return "3x10"

        # "auto" 逻辑
        # 1. 如果已经触发过切换，后续 fold 全部使用 3x10
        if self._strategy_override == "3x10":
            return "3x10"

        # 2. 根据 fold 跨度推断：>5 天说明外部已生成 3x10 fold_specs
        if fold_spec.test_days_count > 5:
            return "3x10"

        # 3. 检查前 3 个 fold 的耗时，决定是否切换
        self._check_auto_switch()

        return "10x3"

    def _check_auto_switch(self) -> None:
        """检查是否需要从 10x3 切换到 3x10。"""
        if self._strategy_override is not None:
            return  # 已经切换过了

        # 至少需要 3 个 fold 的计时数据
        if len(self._fold_timings) < 3:
            return

        # 取前 3 个完成的 fold
        sorted_timings = sorted(self._fold_timings.items())[:3]
        recent_times = [t for _, t in sorted_timings]

        # 如果前 3 个 fold 每个都超过阈值 → 切换
        if all(t > _AUTO_SWITCH_THRESHOLD_SEC for t in recent_times):
            self._strategy_override = "3x10"
            logger.warning(
                "[sgdfnet] Auto-switch triggered: first 3 folds averaged "
                "%.1f min each (> %.0f min threshold). "
                "Subsequent folds will use 3x10 strategy. "
                "Use convert_10x3_to_3x10_specs() to regenerate fold_specs.",
                sum(recent_times) / len(recent_times) / 60,
                _AUTO_SWITCH_THRESHOLD_SEC / 60,
            )

    def _record_timing(self, fold_id: int, elapsed: float) -> None:
        """记录 fold 耗时。"""
        self._fold_timings[fold_id] = elapsed
        logger.info(
            "[sgdfnet] fold %d completed in %.1f sec (%.1f min)",
            fold_id, elapsed, elapsed / 60,
        )

    # ------------------------------------------------------------------
    # 10x3 策略（原有行为）
    # ------------------------------------------------------------------

    def _run_10x3_fold(
        self, task: str, fold_spec: FoldSpec, data_path: str, **kwargs,
    ) -> FoldResult:
        """10x3 策略：单个 3 天 fold 的标准训练+预测。"""
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

        # 标记元数据
        result_df["tap_source"] = "rolling_cutoff"
        result_df["source_confidence"] = 1.0
        result_df["fold_strategy"] = "10x3"

        return FoldResult(
            fold_id=fold_spec.fold_id,
            model_name=self.model_name,
            task=task,
            fold_spec=fold_spec,
            predictions_df=result_df,
            success=True,
        )

    # ------------------------------------------------------------------
    # 3x10 策略
    # ------------------------------------------------------------------

    def _run_3x10_fold(
        self, task: str, fold_spec: FoldSpec, data_path: str, **kwargs,
    ) -> FoldResult:
        """3x10 策略：单个 10 天 fold 的训练+预测，结果拆分到 sub-blocks。

        10 天预测覆盖约 3.3 个 block，每行按日期映射到正确的 tap_block_id。
        """
        result_df = _run_sgdfnet_fold(data_path, fold_spec)

        if result_df is None or result_df.empty:
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message="No predictions generated (3x10 strategy)",
            )

        # 标记元数据
        result_df["tap_source"] = "rolling_cutoff"
        result_df["source_confidence"] = 1.0
        result_df["fold_strategy"] = "3x10"

        # 推断 D（目标月第一天）并添加 block 映射
        result_df = _annotate_block_columns(result_df, fold_spec)

        return FoldResult(
            fold_id=fold_spec.fold_id,
            model_name=self.model_name,
            task=task,
            fold_spec=fold_spec,
            predictions_df=result_df,
            success=True,
        )

    # ------------------------------------------------------------------
    # 静态工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def convert_10x3_to_3x10_specs(
        fold_specs_10x3: list[FoldSpec],
    ) -> list[FoldSpec]:
        """将 10 个 3 天 fold_specs 转换为 3 个 10 天 fold_specs。

        转换规则（D = 目标月第一天）：
            新 fold 0: D-30 ~ D-21 (10 天)
            新 fold 1: D-20 ~ D-11 (10 天)
            新 fold 2: D-10 ~ D-1  (10 天)

        新 fold 的 train_end 取对应时间段的第一个子 fold 的 train_end。
        """
        if not fold_specs_10x3:
            return []

        specs = sorted(fold_specs_10x3, key=lambda fs: fs.test_start)

        # 推断 D（目标月第一天）= max(test_end) + 1 day
        D = specs[-1].test_end + timedelta(days=1)

        # 定义 3 个 10 天区间
        ranges = [
            (D - timedelta(days=30), D - timedelta(days=21)),  # D-30 ~ D-21
            (D - timedelta(days=20), D - timedelta(days=11)),  # D-20 ~ D-11
            (D - timedelta(days=10), D - timedelta(days=1)),   # D-10 ~ D-1
        ]

        result: list[FoldSpec] = []
        for new_id, (range_start, range_end) in enumerate(ranges):
            # 找到覆盖该时间段的原始 fold_specs
            overlapping = [
                fs for fs in specs
                if fs.test_start <= range_end and fs.test_end >= range_start
            ]
            if not overlapping:
                logger.warning(
                    "[sgdfnet] No overlapping folds for 3x10 fold %d (%s ~ %s)",
                    new_id, range_start, range_end,
                )
                continue

            # train_end 取第一个重叠 fold 的 train_end
            # （对于 fold 0 用最早的 train_end，确保 cutoff-safe）
            train_end = min(fs.train_end for fs in overlapping)
            train_start = overlapping[0].train_start
            target_month = overlapping[0].target_month

            new_spec = FoldSpec(
                fold_id=new_id,
                train_start=train_start,
                train_end=train_end,
                test_start=range_start,
                test_end=range_end,
                target_month=target_month,
            )
            result.append(new_spec)
            logger.info(
                "[sgdfnet] 3x10 fold %d: %s -> %s (10 days, merged from %d sub-folds)",
                new_id, range_start, range_end, len(overlapping),
            )

        return result


# ---------------------------------------------------------------------------
# 底层函数
# ---------------------------------------------------------------------------


def _run_sgdfnet_fold(data_path: str, fold_spec: FoldSpec) -> pd.DataFrame | None:
    """将 FoldSpec 映射为 SGDFNet pipeline 调用。"""
    from SGDFNet.pipeline import ModelPipeline

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


def _annotate_block_columns(
    df: pd.DataFrame, fold_spec: FoldSpec, predict_date: str = "",
) -> pd.DataFrame:
    """为 3x10 fold 的预测结果添加 tap_block_id / age_block / horizon_day。

    通过解析每行的日期，计算其属于哪个 3 天 block。
    D = predict_date (预测日), Block 划分：
        Block 0: D-30, D-29, D-28
        Block 1: D-27, D-26, D-25
        ...
        Block 9: D-3,  D-2,  D-1
    """
    result = df.copy()

    # 推断 D：优先使用 predict_date，其次 target_month，最后 fallback
    if predict_date:
        D = pd.Timestamp(predict_date).date()
    else:
        try:
            D = pd.Timestamp(fold_spec.target_month + "-01").date()
        except Exception:
            # fallback: D = test_end + 1 day (仅对最后一个 fold 准确)
            D = fold_spec.test_end + timedelta(days=1)

    # 检测日期列
    date_col = None
    for candidate in ("business_day", "target_day", "ds", "timestamp", "时刻", "鏃跺埢"):
        if candidate in result.columns:
            date_col = candidate
            break

    if date_col is None:
        logger.warning(
            "[sgdfnet] No date column found for block annotation. "
            "Columns: %s", list(result.columns),
        )
        result["tap_block_id"] = None
        result["age_block"] = None
        result["horizon_day"] = None
        return result

    # 解析日期
    dates = pd.to_datetime(result[date_col], errors="coerce").dt.date

    def _compute_block(d) -> int | None:
        if d is None or pd.isna(d):
            return None
        k = (D - d).days  # D-1 → k=1, D-30 → k=30
        if k < 1 or k > 30:
            return None
        return (30 - k) // 3

    result["tap_block_id"] = dates.apply(_compute_block)
    result["age_block"] = result["tap_block_id"].apply(
        lambda b: 9 - b if b is not None and not pd.isna(b) else None,
    )
    result["horizon_day"] = dates.apply(
        lambda d: (D - d).days if d is not None and not pd.isna(d) else None,
    )

    return result
