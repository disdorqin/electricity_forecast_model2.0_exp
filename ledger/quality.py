# -*- coding: utf-8 -*-
"""Ledger Quality Check — 账本质量检查。

检查项：
  1. 每个 target/business_day 是否有 24 行
  2. hour_business 是否在 1~24 范围内
  3. timestamp 00:00 是否正确映射到上一业务日 hour 24
  4. 是否重复写入
  5. 是否缺模型
  6. 是否缺 final_pred
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import pandas as pd
import numpy as np
from ledger.schema import (
    LEDGER_COLUMNS, RUN_DATE, FORECAST_DATE, HOUR_BUSINESS,
    TIMESTAMP, TARGET, MODEL_NAME, Y_PRED, BASE_FUSED_PRED,
    SPIKE_CORRECTED_PRED, FINAL_PRED, Y_TRUE, PERIOD,
    AVAILABLE_DATA_CUTOFF, PIPELINE_VERSION, SOURCE_FILE,
    CREATED_AT, DEFAULT_LEDGER_DIR, ledger_path,
    business_hour_from_timestamp,
)

logger = logging.getLogger(__name__)

FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["sgdfnet", "timemixer", "rt916", "timesfm"]
FORMAL_MODELS_BY_TASK = {
    "dayahead": FORMAL_DAYAHEAD_MODELS,
    "realtime": FORMAL_REALTIME_MODELS,
}

@dataclass
class LedgerQualityReport:
    """质量检查报告。"""
    ledger_path: str = ""
    total_rows: int = 0
    n_targets: int = 0
    n_forecast_dates: int = 0
    n_models: int = 0
    rows_per_day_target: dict[str, dict[str, int]] = field(default_factory=dict)
    hour_business_out_of_range: list[dict] = field(default_factory=list)
    timestamp_mapping_errors: list[dict] = field(default_factory=list)
    duplicate_rows: list[dict] = field(default_factory=list)
    missing_models: list[dict] = field(default_factory=list)
    missing_final_pred: int = 0
    n_nulls_per_column: dict[str, int] = field(default_factory=dict)
    passed: bool = True
    all_errors: list[str] = field(default_factory=list)
    all_warnings: list[str] = field(default_factory=list)
    checked_at: str = ""

    def summary(self) -> str:
        """返回人类可读的摘要。"""
        lines = [
            f"Ledger Quality Report: {self.ledger_path}",
            f"  Total rows: {self.total_rows}",
            f"  Targets: {self.n_targets}",
            f"  Forecast dates: {self.n_forecast_dates}",
            f"  Models: {self.n_models}",
            f"  Missing final_pred: {self.missing_final_pred}",
            f"  Duplicate rows: {len(self.duplicate_rows)}",
            f"  hour_business out of range: {len(self.hour_business_out_of_range)}",
            f"  Timestamp mapping errors: {len(self.timestamp_mapping_errors)}",
            f"  Missing models: {len(self.missing_models)}",
            f"  PASSED: {self.passed}",
        ]
        if self.all_errors:
            lines.append(f"  Errors ({len(self.all_errors)}):")
            for e in self.all_errors[:10]:
                lines.append(f"    - {e}")
        if self.all_warnings:
            lines.append(f"  Warnings ({len(self.all_warnings)}):")
            for w in self.all_warnings[:5]:
                lines.append(f"    - {w}")
        return "\n".join(lines)


def run_ledger_quality_check(
    ledger: str | Path | pd.DataFrame,
    *,
    expected_models: dict[str, list[str]] | None = None,
    strict: bool = True,
) -> LedgerQualityReport:
    """对账本执行全面质量检查。

    Parameters
    ----------
    ledger : str or Path or pd.DataFrame
        账本路径或 DataFrame。
    expected_models : dict or None
        每个 target 的期望模型列表。
        默认使用 FORMAL_MODELS_BY_TASK。
    strict : bool
        严格模式：检查所有列。

    Returns
    -------
    LedgerQualityReport
    """
    report = LedgerQualityReport()
    report.checked_at = datetime.now().isoformat()

    # 1. 加载账本
    if isinstance(ledger, (str, Path)):
        ledger_path = Path(ledger)
        report.ledger_path = str(ledger_path)
        if not ledger_path.exists():
            report.passed = False
            report.all_errors.append(f"Ledger file not found: {ledger_path}")
            return report
        df = pd.read_csv(ledger_path)
    else:
        df = ledger.copy()
        report.ledger_path = "<DataFrame>"

    report.total_rows = len(df)
    if df.empty:
        report.all_errors.append("Ledger is empty")
        report.passed = False
        return report

    # 2. 基本列检查
    from ledger.schema import validate_ledger_schema
    schema_errors = validate_ledger_schema(df, strict=strict)
    if schema_errors:
        for e in schema_errors:
            report.all_errors.append(e)
        report.passed = False

    # 3. 每 target/business_day 的行数
    report.n_targets = df[TARGET].nunique() if TARGET in df.columns else 0
    report.n_forecast_dates = df[FORECAST_DATE].nunique() if FORECAST_DATE in df.columns else 0
    report.n_models = df[MODEL_NAME].nunique() if MODEL_NAME in df.columns else 0
    if TARGET in df.columns and FORECAST_DATE in df.columns:
        grouped = df.groupby([TARGET, FORECAST_DATE]).size()
        for (target, fdate), cnt in grouped.items():
            if report.rows_per_day_target.setdefault(str(target), {}) is None:
                report.rows_per_day_target[str(target)] = {}
            report.rows_per_day_target[str(target)][str(fdate)] = int(cnt)
            expected = 24 * (report.n_models if report.n_models > 0 else 1)
            if cnt != 24 and cnt != 24 * max(report.n_models, 1):
                # Simple check: should be 24 * n_models for that target
                models_for_target = FORMAL_MODELS_BY_TASK.get(target, [])
                n_m = len(models_for_target)
                expected_cnt = 24 * n_m if n_m > 0 else 24
                if cnt != expected_cnt:
                    msg = f"{target}/{fdate}: {cnt} rows (expected {expected_cnt})"
                    report.all_errors.append(msg)
                    report.passed = False
    # 4. hour_business 范围检查 (1~24)
    if HOUR_BUSINESS in df.columns:
        hb = pd.to_numeric(df[HOUR_BUSINESS], errors="coerce")
        bad = hb.isna() | (hb < 1) | (hb > 24)
        bad_indices = df.index[bad].tolist()
        for idx in bad_indices[:20]:  # 最多记录 20 条
            report.hour_business_out_of_range.append({
                "row": int(idx),
                FORECAST_DATE: str(df.loc[idx, FORECAST_DATE]) if FORECAST_DATE in df.columns else "?",
                HOUR_BUSINESS: df.loc[idx, HOUR_BUSINESS],
                TARGET: str(df.loc[idx, TARGET]) if TARGET in df.columns else "?",
            })
        if bad_indices:
            report.all_errors.append(f"{len(bad_indices)} rows have hour_business outside [1, 24]")
            report.passed = False
    # 5. timestamp 映射检查：00:00 → 上一业务日 hour 24
    if TIMESTAMP in df.columns and FORECAST_DATE in df.columns and HOUR_BUSINESS in df.columns:
        ts = pd.to_datetime(df[TIMESTAMP], errors="coerce")
        for idx in df.index:
            ts_val = ts.loc[idx]
            if pd.isna(ts_val):
                continue
            hb = df.loc[idx, HOUR_BUSINESS]
            fdate = str(df.loc[idx, FORECAST_DATE])
            # hour 24 should have timestamp 00:00 of next day
            if hb == 24:
                expected_ts = pd.Timestamp(fdate) + pd.Timedelta(days=1)
                expected_ts_str = expected_ts.strftime("%Y-%m-%d %H:%M:%S")
                actual_ts_str = str(ts_val)
                if ts_val.hour != 0 or ts_val.date() != expected_ts.date():
                    report.timestamp_mapping_errors.append({
                        "row": int(idx),
                        FORECAST_DATE: fdate,
                        HOUR_BUSINESS: int(hb),
                        "timestamp": actual_ts_str,
                        "expected_timestamp": expected_ts_str,
                    })
        if report.timestamp_mapping_errors:
            report.all_errors.append(
                f"{len(report.timestamp_mapping_errors)} timestamp mapping errors "
                f"(hour 24 should map to 00:00 of next day)"
            )
            report.passed = False
    # 6. 重复写入检查
    if all(c in df.columns for c in [RUN_DATE, FORECAST_DATE, HOUR_BUSINESS, TARGET, MODEL_NAME]):
        dup_keys = df.groupby([RUN_DATE, FORECAST_DATE, HOUR_BUSINESS, TARGET, MODEL_NAME]).size()
        dup_keys = dup_keys[dup_keys > 1]
        for (rd, fd, hb, tg, md), cnt in dup_keys.items():
            report.duplicate_rows.append({
                RUN_DATE: rd,
                FORECAST_DATE: fd,
                HOUR_BUSINESS: hb,
                TARGET: tg,
                MODEL_NAME: md,
                "count": int(cnt),
            })
        if report.duplicate_rows:
            report.all_errors.append(f"{len(report.duplicate_rows)} duplicate key groups found")
            report.passed = False
    # 7. 缺模型检查
    if MODEL_NAME in df.columns and TARGET in df.columns and FORECAST_DATE in df.columns:
        exp_models = expected_models or FORMAL_MODELS_BY_TASK
        for target in df[TARGET].unique():
            target_df = df[df[TARGET] == target]
            expected = exp_models.get(target, [])
            for fdate in target_df[FORECAST_DATE].unique():
                day_df = target_df[target_df[FORECAST_DATE] == fdate]
                present = set(day_df[MODEL_NAME].unique())
                missing = [m for m in expected if m not in present]
                if missing:
                    report.missing_models.append({
                        TARGET: target,
                        FORECAST_DATE: fdate,
                        "missing": missing,
                        "present": sorted(present),
                    })
        if report.missing_models:
            n_missing = len(report.missing_models)
            report.all_errors.append(f"{n_missing} target/date combinations have missing models")
            report.passed = False
    # 8. 缺 final_pred 检查
    if FINAL_PRED in df.columns:
        missing_fp = df[FINAL_PRED].isna().sum()
        report.missing_final_pred = int(missing_fp)
        if missing_fp > 0:
            report.all_warnings.append(f"{missing_fp} rows missing final_pred")
            if strict:
                report.passed = False
    # 9. 每列空值统计
    report.n_nulls_per_column = {col: int(df[col].isna().sum()) for col in df.columns}
    return report
