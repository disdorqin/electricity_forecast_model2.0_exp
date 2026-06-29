# -*- coding: utf-8 -*-
"""Prediction Ledger Schema — 列定义、dtype、校验。

每行对应一个模型对一个业务日某小时的预测。
账本从每日 pipeline 产出物读取并追加，满 30 天后可启用权重学习。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


# ── 列名常量 ────────────────────────────────────────────────────────────

RUN_DATE = "run_date"                      # pipeline 运行日期 (YYYY-MM-DD)
FORECAST_DATE = "forecast_date"             # 业务日 (business_day, YYYY-MM-DD)
HOUR_BUSINESS = "hour_business"             # 业务小时 1~24
TIMESTAMP = "timestamp"                     # 自然时间戳 (pd.Timestamp)
TARGET = "target"                           # "dayahead" | "realtime"
MODEL_NAME = "model_name"                   # 模型名
Y_PRED = "y_pred"                           # 模型原始预测
BASE_FUSED_PRED = "base_fused_pred"         # 融合预测（不含 spike 修正）
SPIKE_CORRECTED_PRED = "spike_corrected_pred"  # spike 修正后预测（可空）
FINAL_PRED = "final_pred"                   # 最终输出预测
Y_TRUE = "y_true"                           # 真实值（后续回填）
PERIOD = "period"                           # "1_8" | "9_16" | "17_24"
AVAILABLE_DATA_CUTOFF = "available_data_cutoff"  # 数据截止时间描述
PIPELINE_VERSION = "pipeline_version"       # pipeline 版本标识
SOURCE_FILE = "source_file"                 # 来源文件路径（相对项目根）
CREATED_AT = "created_at"                   # 账本行创建时间


# ── 完整列定义（顺序即 CSV 列顺序） ────────────────────────────────────

LEDGER_COLUMNS: list[str] = [
    RUN_DATE,
    FORECAST_DATE,
    HOUR_BUSINESS,
    TIMESTAMP,
    TARGET,
    MODEL_NAME,
    Y_PRED,
    BASE_FUSED_PRED,
    SPIKE_CORRECTED_PRED,
    FINAL_PRED,
    Y_TRUE,
    PERIOD,
    AVAILABLE_DATA_CUTOFF,
    PIPELINE_VERSION,
    SOURCE_FILE,
    CREATED_AT,
]

# ── 推荐 dtypes（读 CSV 时可传入 dtype=...） ──────────────────────────

LEDGER_DTYPES: dict[str, type | Any] = {
    RUN_DATE: str,
    FORECAST_DATE: str,
    HOUR_BUSINESS: "Int64",       # nullable int
    TIMESTAMP: str,                # 存为 ISO 字符串，读后解析
    TARGET: str,
    MODEL_NAME: str,
    Y_PRED: "float64",
    BASE_FUSED_PRED: "float64",
    SPIKE_CORRECTED_PRED: "float64",  # NaN 表示未修正
    FINAL_PRED: "float64",
    Y_TRUE: "float64",
    PERIOD: str,
    AVAILABLE_DATA_CUTOFF: str,
    PIPELINE_VERSION: str,
    SOURCE_FILE: str,
    CREATED_AT: str,
}

# ── 非空列（不允许 NaN） ──────────────────────────────────────────────

REQUIRED_COLUMNS: list[str] = [
    RUN_DATE,
    FORECAST_DATE,
    HOUR_BUSINESS,
    TIMESTAMP,
    TARGET,
    MODEL_NAME,
    Y_PRED,
    FINAL_PRED,
    PERIOD,
    PIPELINE_VERSION,
    CREATED_AT,
]

# ── 校验函数 ────────────────────────────────────────────────────────────


def validate_ledger_schema(df: pd.DataFrame, *, strict: bool = True) -> list[str]:
    """校验 DataFrame 是否符合账本 schema。

    Parameters
    ----------
    df : pd.DataFrame
        待校验的 DataFrame。
    strict : bool
        严格模式：检查所有列是否都存在。否则只检查必要列。

    Returns
    -------
    list[str]
        错误信息列表。为空表示校验通过。
    """
    errors: list[str] = []

    # 1. 检查列存在
    if strict:
        missing = [c for c in LEDGER_COLUMNS if c not in df.columns]
    else:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]

    if missing:
        errors.append(f"Missing columns: {missing}")

    if missing:
        return errors  # 无法进一步校验

    # 2. 检查 hour_business 范围 (1~24)
    if HOUR_BUSINESS in df.columns:
        hb = pd.to_numeric(df[HOUR_BUSINESS], errors="coerce")
        out_of_range = hb.isna() | (hb < 1) | (hb > 24)
        n_bad = out_of_range.sum()
        if n_bad > 0:
            errors.append(f"{n_bad} rows have hour_business outside [1, 24]")
            if strict:
                bad_indices = df.index[out_of_range].tolist()[:10]
                errors.append(f"  Sample bad indices: {bad_indices}")

    # 3. 检查非空列
    for col in REQUIRED_COLUMNS:
        n_null = df[col].isna().sum()
        if n_null > 0:
            errors.append(f"Column '{col}' has {n_null} null values")

    # 4. 检查 target 值
    if TARGET in df.columns:
        valid_targets = {"dayahead", "realtime"}
        bad_targets = set(df[TARGET].unique()) - valid_targets
        if bad_targets:
            errors.append(f"Invalid target values: {bad_targets}")

    # 5. 检查 period 值
    if PERIOD in df.columns:
        valid_periods = {"1_8", "9_16", "17_24"}
        bad_periods = set(df[PERIOD].unique()) - valid_periods
        if bad_periods:
            errors.append(f"Invalid period values: {bad_periods}")

    if errors:
        logger.warning("Schema validation found %d issues", len(errors))
    else:
        logger.info("Schema validation passed (%d cols, %d rows)", len(df.columns), len(df))

    return errors


# ── 账本路径约定 ──────────────────────────────────────────────────────

DEFAULT_LEDGER_DIR = "data/local_ledger"
"""默认账本写入目录（已在 .gitignore 中忽略）。"""


def ledger_path(target: str | None = None) -> str:
    """返回账本 CSV 路径。

    Parameters
    ----------
    target : str or None
        "dayahead" / "realtime" / None（返回合并账本路径）。
    """
    if target:
        return f"{DEFAULT_LEDGER_DIR}/ledger_{target}.csv"
    return f"{DEFAULT_LEDGER_DIR}/ledger.csv"


def make_empty_ledger() -> pd.DataFrame:
    """创建空账本 DataFrame（仅有列名）。"""
    return pd.DataFrame({col: pd.Series(dtype=LEDGER_DTYPES.get(col, "object"))
                         for col in LEDGER_COLUMNS})


# ── 业务小时 / 时间戳 映射工具 ────────────────────────────────────────


def timestamp_from_business_hour(
    business_day: str,
    hour_business: int,
) -> pd.Timestamp:
    """根据业务日和业务小时计算自然时间戳。

    hour_business 1~23 映射为当天 hour 1~23（自然时间 01:00~23:00）。
    hour_business 24 映射为第二天 00:00。
    """
    bd = pd.Timestamp(business_day)
    if hour_business == 24:
        return bd + pd.Timedelta(days=1)  # 00:00 of next day
    else:
        return bd + pd.Timedelta(hours=int(hour_business))


def business_hour_from_timestamp(ts: pd.Timestamp) -> tuple[str, int]:
    """从自然时间戳反算 (business_day, hour_business)。

    自然时间 00:00 → 上一业务日 hour_business=24。
    自然时间 01:00~23:00 → 当日 hour_business=1~23。
    """
    if ts.hour == 0:
        bd = (ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        hb = 24
    else:
        bd = ts.strftime("%Y-%m-%d")
        hb = ts.hour
    return bd, hb
