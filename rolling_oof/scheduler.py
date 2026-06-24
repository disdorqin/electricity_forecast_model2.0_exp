# -*- coding: utf-8 -*-
"""核心编排器：RollingOriginOrchestrator。

统一调度 rolling-origin OOF 池的完整流程：
  阶段 A: 按 fold 顺序执行所有模型的训练+预测
  阶段 B: 汇编 OOF long-table
  阶段 C: 最终陪跑预测
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.audit import audit_single_fold, audit_cross_model_alignment
from rolling_oof.contracts import (
    AuditCheck,
    FoldResult,
    FoldSpec,
    OofPoolManifest,
    RollingOriginConfig,
    generate_fold_specs,
    normalize_long_table,
)
from rolling_oof.output_layout import OofRunLayout

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 适配器注册表（延迟导入以避免循环依赖）
ADAPTER_REGISTRY: dict[str, type[BaseRollingAdapter]] = {}


def _init_registry():
    """延迟初始化适配器注册表。"""
    if ADAPTER_REGISTRY:
        return
    from rolling_oof.adapters.lightgbm import LightGBMRollingAdapter
    from rolling_oof.adapters.timemixer import TimeMixerRollingAdapter
    from rolling_oof.adapters.timesfm import TimesFMRollingAdapter
    from rolling_oof.adapters.sgdfnet import SGDFNetRollingAdapter
    from rolling_oof.adapters.rt916 import RT916RollingAdapter

    ADAPTER_REGISTRY.update({
        "lightgbm": LightGBMRollingAdapter,
        "timemixer": TimeMixerRollingAdapter,
        "timesfm": TimesFMRollingAdapter,
        "sgdfnet": SGDFNetRollingAdapter,
        "rt916": RT916RollingAdapter,
    })


class RollingOriginOrchestrator:
    """rolling-origin OOF 池编排器。

    Parameters
    ----------
    config : RollingOriginConfig
        编排器配置。
    """

    def __init__(self, config: RollingOriginConfig):
        self.config = config
        _init_registry()

        # 生成 fold 列表
        self.folds: list[FoldSpec] = generate_fold_specs(
            start_month=config.start_month,
            end_month=config.end_month,
            expanding=config.expanding,
            train_min_months=config.train_min_months,
        )

        # 输出布局
        self.layout = OofRunLayout(
            pool_root=Path(config.output_root) / config.pool_id,
            pool_id=config.pool_id,
        )
        self.layout.ensure_dirs()

        logger.info(
            "RollingOriginOrchestrator initialized: pool=%s, folds=%d, models=%d, tasks=%d",
            self.config.pool_id,
            len(self.folds),
            len(self.config.models_list),
            len(self.config.tasks_list),
        )

    # ------------------------------------------------------------------
    # 阶段 A
    # ------------------------------------------------------------------

    def run_phase_a(self) -> list[FoldResult]:
        """阶段 A: 按 fold 顺序执行所有模型的所有 fold。

        调度策略:
        - 按 fold_id 升序排列
        - 同一 fold 内，CPU/GPU 模型可并行
        - 不同 fold 之间严格顺序（避免 future data leak）
        """
        all_results: list[FoldResult] = []
        n_folds = len(self.folds)

        for fold in self.folds:
            logger.info(
                "[rolling-oof] Phase A: Fold %d/%d (train~%s → test %s)",
                fold.fold_id + 1,
                n_folds,
                fold.train_end,
                fold.target_month,
            )

            # 确保 fold 目录存在
            self.layout.ensure_dirs(fold.fold_id)

            # 保存 fold_spec.json
            import json as _json
            spec_path = self.layout.fold_spec_path(fold.fold_id)
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            _fold_data = {
                "fold_id": fold.fold_id,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "target_month": fold.target_month,
                "is_expanding": fold.is_expanding,
            }
            with open(spec_path, "w", encoding="utf-8") as _sf:
                _json.dump(_fold_data, _sf, ensure_ascii=False, indent=2)

            for model_name in self.config.models_list:
                for task in self.config.tasks_list:
                    logger.info(
                        "[rolling-oof]   %s/%s: training...",
                        model_name,
                        task,
                    )
                    result = self.execute_fold(fold, model_name, task)
                    all_results.append(result)

                    if result.success:
                        logger.info(
                            "[rolling-oof]   %s/%s: done",
                            model_name,
                            task,
                        )
                    else:
                        logger.error(
                            "[rolling-oof]   %s/%s: FAILED - %s",
                            model_name,
                            task,
                            result.error_message,
                        )

            logger.info("[rolling-oof]   Fold %d/%d complete", fold.fold_id + 1, n_folds)

        return all_results

    def execute_fold(
        self, fold: FoldSpec, model_name: str, task: str
    ) -> FoldResult:
        """执行单个 fold 的训练和预测。

        1. 根据 model_name 获取对应的适配器
        2. 调用 fold_train_predict()
        3. 执行审计检查
        4. 保存结果到 oof_runs/
        """
        adapter_cls = ADAPTER_REGISTRY.get(model_name)
        if adapter_cls is None:
            return FoldResult(
                fold_id=fold.fold_id,
                model_name=model_name,
                task=task,
                fold_spec=fold,
                success=False,
                error_message=f"Unknown model: {model_name}",
            )

        # 跳过模型不支持的 task（如 SGDFNet 只支持 realtime）
        adapter_inst = adapter_cls()
        if task not in adapter_inst.supported_tasks:
            logger.info(
                "[rolling-oof]   %s/%s: skipped (model only supports %s)",
                model_name, task, adapter_inst.supported_tasks,
            )
            return FoldResult(
                fold_id=fold.fold_id,
                model_name=model_name,
                task=task,
                fold_spec=fold,
                success=True,
                error_message=f"Task {task} not supported by {model_name} (supports {adapter_inst.supported_tasks})",
            )

        # 构造适配器的额外参数
        extra_kwargs = {
            "training_months": self.config.training_months,
            "val_ratio": self.config.val_ratio,
            "rolling_mode": self.config.timemixer_rolling_mode,
            "block_days": self.config.timemixer_block_days,
        }

        adapter = adapter_cls()
        result = adapter.fold_train_predict(
            task=task,
            fold_spec=fold,
            data_path=self.config.data_path,
            **extra_kwargs,
        )

        # 审计检查
        if not self.config.skip_audit and result.predictions_df is not None:
            audit_result = audit_single_fold(
                result.predictions_df,
                fold_id=fold.fold_id,
                model_name=model_name,
                task=task,
                output_path=result.output_path,
            )
            # 保存审计报告
            audit_path = self.layout.fold_audit_path(fold.fold_id, model_name, task)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_data = {
                "overall_risk": audit_result.overall_risk,
                "checks": [
                    {
                        "name": c.name,
                        "passed": c.passed,
                        "severity": c.severity,
                        "detail": c.detail,
                    }
                    for c in audit_result.checks
                ],
            }
            with open(audit_path, "w", encoding="utf-8") as f:
                json.dump(audit_data, f, ensure_ascii=False, indent=2)

            if audit_result.overall_risk == "high":
                logger.warning(
                    "[rolling-oof]   %s/%s: audit HIGH risk - %d errors",
                    model_name,
                    task,
                    audit_result.error_count,
                )

        # 保存预测结果
        if result.success and result.predictions_df is not None:
            output_path = self.layout.fold_raw_path(fold.fold_id, model_name, task)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result.predictions_df.to_csv(output_path, index=False)
            result.output_path = str(output_path)

        return result

    # ------------------------------------------------------------------
    # 阶段 B
    # ------------------------------------------------------------------

    def run_phase_b(self, fold_results: list[FoldResult]) -> pd.DataFrame:
        """阶段 B: 汇编 OOF long-table。

        1. 收集所有 fold 的 predictions_df
        2. 合并为统一的 long-table
        3. 保存
        """
        all_frames: list[pd.DataFrame] = []

        for result in fold_results:
            if not result.success or result.predictions_df is None:
                continue
            df = result.predictions_df.copy()
            # 标准化
            df = normalize_long_table(
                df,
                task=result.task,
                model_name=result.model_name,
                fold_id=result.fold_id,
                train_start=result.fold_spec.train_start.isoformat(),
                train_end=result.fold_spec.train_end.isoformat(),
                test_start=result.fold_spec.test_start.isoformat(),
                test_end=result.fold_spec.test_end.isoformat(),
            )
            all_frames.append(df)

        if not all_frames:
            logger.warning("No successful fold results to assemble")
            return pd.DataFrame()

        long_table = pd.concat(all_frames, axis=0, ignore_index=True)
        # 去重
        long_table = long_table.drop_duplicates(
            subset=["task", "model_name", "target_day", "ds", "fold_id"],
            keep="last",
        )

        # 保存
        output_path = self.layout.long_table_path
        long_table.to_csv(output_path, index=False)
        logger.info(
            "[rolling-oof] Phase B: OOF long-table saved (%d rows) -> %s",
            len(long_table),
            output_path,
        )

        return long_table

    # ------------------------------------------------------------------
    # 阶段 C
    # ------------------------------------------------------------------

    def run_phase_c(self, target_date: str) -> pd.DataFrame:
        """阶段 C: 最终陪跑预测。

        用截止到 target_date-1 的所有可用数据训练，预测 target_date 的 24 小时。
        """
        from rolling_oof.escort import run_escort_prediction

        escort_df = run_escort_prediction(
            config=self.config,
            layout=self.layout,
            target_date=target_date,
        )
        return escort_df

    # ------------------------------------------------------------------
    # 全程运行
    # ------------------------------------------------------------------

    def run_all(
        self, escort_date: Optional[str] = None
    ) -> dict:
        """完整运行阶段 A+B+C。

        Returns
        -------
        dict
            包含 pool_id, manifest_path, long_table_path, fold_results 等。
        """
        start_time = datetime.now()

        # 阶段 A
        fold_results = self.run_phase_a()

        # 阶段 B
        long_table = self.run_phase_b(fold_results)

        # 全局审计
        global_audit = self._run_global_audit(fold_results, long_table)
        audit_path = self.layout.audit_path
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(global_audit, f, ensure_ascii=False, indent=2, default=str)

        # Manifest
        manifest = OofPoolManifest(
            oof_pool_id=self.config.pool_id,
            generated_at=datetime.now().isoformat(),
            date_range_start=str(self.folds[0].test_start) if self.folds else "",
            date_range_end=str(self.folds[-1].test_end) if self.folds else "",
            folds=self.folds,
            models=self.config.models_list,
            tasks=self.config.tasks_list,
            expanding=self.config.expanding,
            train_min_months=self.config.train_min_months,
        )
        manifest_data = {
            "oof_pool_id": manifest.oof_pool_id,
            "generated_at": manifest.generated_at,
            "date_range_start": manifest.date_range_start,
            "date_range_end": manifest.date_range_end,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "train_start": str(f.train_start),
                    "train_end": str(f.train_end),
                    "test_start": str(f.test_start),
                    "test_end": str(f.test_end),
                    "target_month": f.target_month,
                }
                for f in self.folds
            ],
            "models": manifest.models,
            "tasks": manifest.tasks,
            "expanding": manifest.expanding,
            "train_min_months": manifest.train_min_months,
        }
        with open(self.layout.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, ensure_ascii=False, indent=2, default=str)

        # 阶段 C（可选）
        escort_df = pd.DataFrame()
        if escort_date:
            escort_df = self.run_phase_c(escort_date)

        elapsed = (datetime.now() - start_time).total_seconds()

        return {
            "pool_id": self.config.pool_id,
            "manifest_path": str(self.layout.manifest_path),
            "long_table_path": str(self.layout.long_table_path),
            "audit_path": str(self.layout.audit_path),
            "fold_results": fold_results,
            "long_table": long_table,
            "escort_predictions": escort_df,
            "elapsed_seconds": elapsed,
            "n_folds": len(self.folds),
            "n_models": len(self.config.models_list),
            "n_tasks": len(self.config.tasks_list),
        }

    # ------------------------------------------------------------------
    # 审计
    # ------------------------------------------------------------------

    def _run_global_audit(
        self,
        fold_results: list[FoldResult],
        long_table: pd.DataFrame,
    ) -> dict:
        """执行全局审计。"""
        checks: list[AuditCheck] = []

        # 检查是否有 fold 失败
        failed = [r for r in fold_results if not r.success]
        if failed:
            checks.append(
                AuditCheck(
                    name="all_folds_success",
                    passed=False,
                    severity="error",
                    detail=f"{len(failed)} fold(s) failed",
                    evidence={
                        "failed": [
                            f"{r.model_name}/{r.task}/fold_{r.fold_id}: {r.error_message}"
                            for r in failed
                        ]
                    },
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="all_folds_success",
                    passed=True,
                    severity="info",
                    detail="All folds completed successfully",
                )
            )

        # 检查 long-table 非空
        if long_table.empty:
            checks.append(
                AuditCheck(
                    name="long_table_nonempty",
                    passed=False,
                    severity="error",
                    detail="OOF long-table is empty",
                )
            )
        else:
            checks.append(
                AuditCheck(
                    name="long_table_nonempty",
                    passed=True,
                    severity="info",
                    detail=f"Long table has {len(long_table)} rows",
                )
            )

        # 跨模型对齐
        model_predictions: dict[tuple[str, str], pd.DataFrame] = {}
        for r in fold_results:
            if r.success and r.predictions_df is not None:
                model_predictions[(r.model_name, r.task)] = r.predictions_df
        alignment_checks = audit_cross_model_alignment(model_predictions)
        checks.extend(alignment_checks)

        # 汇总
        error_count = sum(1 for c in checks if not c.passed and c.severity == "error")

        return {
            "pool_id": self.config.pool_id,
            "generated_at": datetime.now().isoformat(),
            "overall_risk": "high" if error_count > 0 else "low",
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "detail": c.detail,
                }
                for c in checks
            ],
            "summary": {
                "total_folds": len(self.folds),
                "total_results": len(fold_results),
                "successful": len([r for r in fold_results if r.success]),
                "failed": len(failed),
            },
        }
