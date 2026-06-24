# -*- coding: utf-8 -*-
"""rolling-origin OOF 预测池 CLI 调度入口。"""

from __future__ import annotations

import logging
from argparse import Namespace

from rolling_oof.contracts import RollingOriginConfig
from rolling_oof.scheduler import RollingOriginOrchestrator

logger = logging.getLogger(__name__)


def run_rolling_oof(args: Namespace) -> dict:
    """从 argparse 参数构造配置并运行 rolling-origin 流程。

    Parameters
    ----------
    args : Namespace
        CLI 解析结果。

    Returns
    -------
    dict
        包含 pool_id, long_table_path, manifest_path 等。
    """
    # 解析模型列表
    models = _resolve_models(args)

    # 解析任务列表
    tasks = _resolve_tasks(args)

    # 确定起止月份
    if args.oof_start_month and args.oof_end_month:
        start_month = args.oof_start_month
        end_month = args.oof_end_month
    elif args.date:
        # 使用 --date 推断（默认最近1个月）
        end_month = args.date[:7]  # "2026-08" from "2026-08-31"
        start_month = end_month  # 默认1个月
    else:
        raise ValueError("Must specify --oof-start-month and --oof-end-month, or --date")

    # 构造配置
    config = RollingOriginConfig(
        data_path=args.data_path,
        output_root=args.oof_output_root or "oof_runs",
        start_month=start_month,
        end_month=end_month,
        models=tuple(models),
        tasks=tuple(tasks),
        expanding=args.oof_expanding,
        train_min_months=getattr(args, "oof_train_min_months", 6),
        max_cpu_workers=getattr(args, "max_cpu_workers", 2),
        max_gpu_workers=getattr(args, "max_gpu_workers", 1),
        skip_audit=getattr(args, "skip_oof_audit", False),
        timemixer_rolling_mode=getattr(args, "timemixer_rolling_mode", "daily"),
        timemixer_block_days=getattr(args, "timemixer_block_days", 7),
        training_months=getattr(args, "training_months", 12),
        val_ratio=getattr(args, "val_ratio", 0.2),
    )

    # 运行
    orchestrator = RollingOriginOrchestrator(config)
    escort_date = getattr(args, "escort_date", None)

    result = orchestrator.run_all(escort_date=escort_date)

    # 打印摘要
    print_summary(result)

    return result


def print_summary(result: dict) -> None:
    """打印运行摘要。"""
    print()
    print("=" * 60)
    print("  Rolling-Origin OOF Pool 生成完成")
    print("=" * 60)
    print(f"  Pool ID:      {result.get('pool_id', 'N/A')}")
    print(f"  Long table:   {result.get('long_table_path', 'N/A')}")
    print(f"  Manifest:     {result.get('manifest_path', 'N/A')}")
    print(f"  Audit:        {result.get('audit_path', 'N/A')}")
    print(f"  Fold count:   {result.get('n_folds', 0)}")
    print(f"  Model count:  {result.get('n_models', 0)}")
    print(f"  Task count:   {result.get('n_tasks', 0)}")
    print(f"  Elapsed:      {result.get('elapsed_seconds', 0):.0f}s")
    print()

    # 统计成功率
    fold_results = result.get("fold_results", [])
    successful = sum(1 for r in fold_results if r.success)
    total = len(fold_results)
    if total > 0:
        print(f"  Success rate: {successful}/{total} ({100 * successful / total:.1f}%)")
        if successful < total:
            failed = [r for r in fold_results if not r.success]
            print(f"  Failures:")
            for f in failed:
                print(f"    - {f.model_name}/{f.task}/fold_{f.fold_id}: {f.error_message}")

    # 阶段C
    escort_df = result.get("escort_predictions")
    if escort_df is not None and not escort_df.empty:
        print(f"\n  Escort predictions: {len(escort_df)} rows")

    print("=" * 60)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _resolve_models(args: Namespace) -> list[str]:
    """解析 --models 参数。"""
    raw = getattr(args, "models", "all")
    if raw == "all" or not raw:
        return list(RollingOriginConfig().models)
    return [m.strip().lower() for m in raw.split(",")]


def _resolve_tasks(args: Namespace) -> list[str]:
    """解析 --target 参数。"""
    raw = getattr(args, "target", "both")
    if raw == "both":
        return ["dayahead", "realtime"]
    return [raw]
