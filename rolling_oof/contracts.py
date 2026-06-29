# -*- coding: utf-8 -*-
"""rolling-origin OOF 预测池统一数据合约。

定义所有核心 dataclass：FoldSpec, FoldResult, RollingOriginConfig 等。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 常数
# ---------------------------------------------------------------------------

# 标准 long-table 列（与 fusion/contracts.py 保持一致）
LONG_TABLE_COLUMNS: list[str] = [
    "task",
    "model_name",
    "fold_id",
    "train_start",
    "train_end",
    "test_start",
    "test_end",
    "target_day",
    "business_day",
    "ds",
    "period",
    "hour_business",
    "y_true",
    "y_pred",
    "source",
    "run_mode",
    "created_at",
]

# 时段映射
PERIOD_MAP: dict[range, str] = {
    range(1, 9): "1_8",
    range(9, 17): "9_16",
    range(17, 25): "17_24",
}


def assign_period(hour_business) -> str:
    """为商业小时(1-24)分配时段标签。"""
    if pd.isna(hour_business):
        return "unknown"
    for rng, label in PERIOD_MAP.items():
        if hour_business in rng:
            return label
    return "unknown"


# ---------------------------------------------------------------------------
# 核心 dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldSpec:
    """定义单个 rolling-origin fold 的参数。

    Attributes
    ----------
    fold_id : int
        从 0 开始的 fold 编号。
    train_start : date
        训练数据起始日期（inclusive）。
    train_end : date
        训练数据结束日期（inclusive），即 cutoff date。
        不允许包含 test 期间的数据。
    test_start : date
        预测目标起始日期（inclusive）。
    test_end : date
        预测目标结束日期（inclusive）。
    target_month : str
        目标月份标签，如 "2026-08"。
    """

    fold_id: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    target_month: str

    def __post_init__(self):
        # 协议校验
        if self.train_end >= self.test_start:
            raise ValueError(
                f"Fold {self.fold_id}: train_end ({self.train_end}) must be < "
                f"test_start ({self.test_start})"
            )
        if self.test_start > self.test_end:
            raise ValueError(
                f"Fold {self.fold_id}: test_start ({self.test_start}) must be <= "
                f"test_end ({self.test_end})"
            )
        if self.train_start > self.train_end:
            raise ValueError(
                f"Fold {self.fold_id}: train_start ({self.train_start}) must be <= "
                f"train_end ({self.train_end})"
            )

    @property
    def is_expanding(self) -> bool:
        """train_start 是否固定不变（expanding window）。"""
        return True  # 本项目默认 expanding

    @property
    def test_days_count(self) -> int:
        """预测天数。"""
        return (self.test_end - self.test_start).days + 1

    @property
    def fold_label(self) -> str:
        """人类可读的 fold 标签。"""
        return f"fold_{self.fold_id}_train~{self.train_end}_test_{self.target_month}"


@dataclass
class FoldResult:
    """单个 fold 的执行结果。

    Attributes
    ----------
    fold_id : int
        fold 编号。
    model_name : str
        模型名称。
    task : str
        "dayahead" 或 "realtime"。
    fold_spec : FoldSpec
        该 fold 的参数。
    predictions_df : pd.DataFrame
        标准化 long-table 格式的预测结果。
    train_metrics : dict
        训练指标 {mae, smape, ...}。
    success : bool
        是否成功完成。
    error_message : str
        失败原因（如有）。
    output_path : str
        原始输出文件路径。
    """

    fold_id: int
    model_name: str
    task: str
    fold_spec: FoldSpec
    predictions_df: Optional[pd.DataFrame] = None
    train_metrics: dict[str, float] = field(default_factory=dict)
    success: bool = True
    error_message: str = ""
    output_path: str = ""


# ---------------------------------------------------------------------------
# 审计 dataclass
# ---------------------------------------------------------------------------


@dataclass
class AuditCheck:
    """单个审计检查项。"""

    name: str  # 检查项名称
    passed: bool
    severity: str  # "info" | "warning" | "error"
    detail: str
    evidence: Optional[dict] = None
    model_name: str = ""
    task: str = ""


@dataclass
class FoldAuditResult:
    """单个 fold 的审计结果。"""

    fold_id: int
    model_name: str
    task: str
    checks: list[AuditCheck] = field(default_factory=list)
    overall_risk: str = "low"  # low / medium / high

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed and c.severity == "warning")


# ---------------------------------------------------------------------------
# 编排器配置
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RollingOriginConfig:
    """rolling-origin 编排器的配置。

    Attributes
    ----------
    data_path : str
        原始数据文件路径。
    output_root : str
        输出根目录。
    start_month : str
        第一个目标月，如 "2026-08"。
    end_month : str
        最后一个目标月，如 "2026-12"。
    models : list[str]
        参与 OOF 的模型列表。
    tasks : list[str]
        ["dayahead", "realtime"] 或其一。
    expanding : bool
        True=expanding 窗口, False=sliding 窗口。
    train_min_months : int
        滑动窗口的最小训练月数。
    max_cpu_workers : int
        CPU 模型并行数。
    max_gpu_workers : int
        GPU 模型并行数（通常是 1）。
    skip_audit : bool
        是否跳过审计。
    timemixer_rolling_mode : str
        TimeMixer 专用："window_once" | "block" | "daily"。
    timemixer_block_days : int
        block 模式下每 block 的天数。
    training_months : int
        基础模型的训练月数（仅对支持该参数的模型生效）。
    val_ratio : float
        内部验证集比例。
    """

    data_path: str
    output_root: str = "oof_runs"
    start_month: str = "2026-08"
    end_month: str = "2026-08"
    models: tuple[str, ...] = ("lightgbm", "sgdfnet", "rt916", "timemixer", "timesfm")
    tasks: tuple[str, ...] = ("dayahead", "realtime")
    expanding: bool = True
    train_min_months: int = 6
    max_cpu_workers: int = 2
    max_gpu_workers: int = 1
    skip_audit: bool = False
    timemixer_rolling_mode: str = "daily"
    timemixer_block_days: int = 7
    training_months: int = 12
    val_ratio: float = 0.2

    def __post_init__(self):
        if self.timemixer_rolling_mode not in ("window_once", "block", "daily"):
            raise ValueError(
                f"Invalid timemixer_rolling_mode: {self.timemixer_rolling_mode}"
            )

    @property
    def pool_id(self) -> str:
        """生成 OOF 池标识符。"""
        parts = [
            f"oof_{self.start_month}_to_{self.end_month}",
            "expanding" if self.expanding else f"sliding_{self.train_min_months}m",
        ]
        return "_".join(parts)

    @property
    def models_list(self) -> list[str]:
        return list(self.models)

    @property
    def tasks_list(self) -> list[str]:
        return list(self.tasks)


# ---------------------------------------------------------------------------
# OOF 池清单
# ---------------------------------------------------------------------------


@dataclass
class OofPoolManifest:
    """整个 OOF 池的清单。"""

    oof_pool_id: str
    generated_at: str
    date_range_start: str
    date_range_end: str
    folds: list = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    expanding: bool = True
    train_min_months: int = 6


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def generate_fold_specs(
    start_month: str,
    end_month: str,
    data_start: Optional[date] = None,
    expanding: bool = True,
    train_min_months: int = 6,
) -> list[FoldSpec]:
    """根据起止月份生成 FoldSpec 列表。

    Parameters
    ----------
    start_month : str
        第一个目标月，如 "2026-08"。
    end_month : str
        最后一个目标月，如 "2026-12"。
    data_start : date, optional
        数据最早可用日期。如果为 None，使用 2023-01-01。
    expanding : bool
        True=expanding 窗口（train_start 固定），False=sliding 窗口。
    train_min_months : int
        sliding 窗口的最小训练月数。

    Returns
    -------
    list[FoldSpec]
    """
    if data_start is None:
        data_start = date(2023, 1, 1)

    start_dt = _parse_month(start_month)
    end_dt = _parse_month(end_month)

    folds: list[FoldSpec] = []
    current = start_dt
    fold_id = 0

    while current <= end_dt:
        # 计算本月最后一天
        next_month = (
            date(current.year + (current.month // 12), (current.month % 12) + 1, 1)
            if current.month < 12
            else date(current.year + 1, 1, 1)
        )
        test_end = next_month - timedelta(days=1)

        if expanding:
            train_start = data_start
        else:
            train_start = max(
                data_start,
                date(current.year, current.month, 1)
                - timedelta(days=train_min_months * 30),
            )

        train_end = date(current.year, current.month, 1) - timedelta(days=1)

        fold = FoldSpec(
            fold_id=fold_id,
            train_start=train_start,
            train_end=train_end,
            test_start=date(current.year, current.month, 1),
            test_end=test_end,
            target_month=current.strftime("%Y-%m"),
        )
        folds.append(fold)

        current = next_month
        fold_id += 1

    return folds


def _parse_month(month_str: str) -> date:
    """解析月份字符串为当月的第一天。"""
    return date(int(month_str[:4]), int(month_str[5:7]), 1)


def normalize_long_table(
    df: pd.DataFrame,
    task: str,
    model_name: str,
    fold_id: int,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> pd.DataFrame:
    """将原始预测 DataFrame 标准化为统一 long-table 格式。

    保证输出包含 LONG_TABLE_COLUMNS 中所有列，缺失列用空值填充。
    """
    result = df.copy()
    result["task"] = task
    result["model_name"] = model_name
    result["fold_id"] = fold_id
    result["train_start"] = train_start
    result["train_end"] = train_end
    result["test_start"] = test_start
    result["test_end"] = test_end
    result["created_at"] = datetime.now().isoformat()

    # 自动计算 business_day / hour_business / period
    if "ds" in result.columns:
        ds_series = pd.to_datetime(result["ds"])
        # hour_business: 00:00 -> 24, else actual hour
        result["hour_business"] = ds_series.apply(
            lambda t: 24 if pd.notna(t) and t.hour == 0 else (t.hour if pd.notna(t) else pd.NA)
        ).astype("Int64")  # nullable int to tolerate NaN from missing ds
        # business_day: 00:00 归属前一天
        result["business_day"] = ds_series.apply(
            lambda t: (t - pd.Timedelta(days=1) if pd.notna(t) and t.hour == 0 else t).strftime("%Y-%m-%d") if pd.notna(t) else None
        )
        # period: 1_8 / 9_16 / 17_24
        result["period"] = result["hour_business"].apply(assign_period)

    # 列名自动映射：各模型输出列名各不相同，统一到 long-table 标准名
    _COLUMN_RENAME_MAP: dict[str, str] = {
        # 真实值列
        "y": "y_true",
        "actual": "y_true",
        # 预测值列
        "pred_y": "y_pred",
        "price": "y_pred",
        "predictions": "y_pred",
        "预测值": "y_pred",
        "prediction": "y_pred",
        # 时间列
        "timestamp": "ds",
        "时刻": "ds",
    }
    for src, dst in _COLUMN_RENAME_MAP.items():
        if src in result.columns and dst not in result.columns:
            result[dst] = result[src]

    # 补充缺失列
    for col in LONG_TABLE_COLUMNS:
        if col not in result.columns:
            result[col] = None

    # 确保输出只包含 contract 列
    return result[LONG_TABLE_COLUMNS]
