# -*- coding: utf-8 -*-
"""滚动起点 OOF 预测池 —— 审计模块。

对模型输出进行10项审计检查：
1. 每个 target_day 是否 24 行
2. 是否有重复 ds
3. 是否缺失 00:00 或 01:00
4. hour_business 是否为 1-24
5. period 是否正确
6. business_day 是否正确
7. y_pred 是否 NaN
8. y_pred 是否存在极端异常值
9. 每个 task/fold 的覆盖率
10. 多模型之间同一 target_day/ds 是否对齐
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from rolling_oof.contracts import AuditCheck, FoldAuditResult, assign_period

# 极端异常值阈值（欧元/MWh）
MAD_PRICE_UPPER: float = 1000.0  # 绝对上界
MAD_PRICE_LOWER: float = -500.0  # 绝对下界


def audit_single_fold(
    df: pd.DataFrame,
    fold_id: int,
    model_name: str,
    task: str,
    output_path: str = "",
) -> FoldAuditResult:
    """对单个 fold 的输出进行全面审计。

    Parameters
    ----------
    df : pd.DataFrame
        模型输出的预测结果，必须包含 task, model_name, target_day,
        ds, period, hour_business, y_true, y_pred 等列。
    fold_id : int
        fold 编号。
    model_name : str
        模型名称。
    task : str
        任务类型。
    output_path : str
        输出路径（用于记录）。

    Returns
    -------
    FoldAuditResult
    """
    result = FoldAuditResult(
        fold_id=fold_id,
        model_name=model_name,
        task=task,
    )
    checks: list[AuditCheck] = []

    # --- 1. 每天24行 ---
    checks.append(_check_24_rows_per_day(df, model_name, task))

    # --- 2. 重复ds ---
    checks.append(_check_duplicate_ds(df, model_name, task))

    # --- 3. 缺失00:00或01:00 ---
    checks.append(_check_missing_hours(df, model_name, task))

    # --- 4. hour_business 1-24 ---
    checks.append(_check_hour_business_range(df, model_name, task))

    # --- 5. period 正确 ---
    checks.append(_check_period_labels(df, model_name, task))

    # --- 6. business_day 正确 ---
    checks.append(_check_business_day(df, model_name, task))

    # --- 7. y_pred NaN ---
    checks.append(_check_pred_nan(df, model_name, task))

    # --- 8. y_pred 极端异常值 ---
    checks.append(_check_pred_outliers(df, model_name, task))

    # --- 9. 覆盖率 ---
    checks.append(_check_coverage(df, model_name, task))

    # --- 综合评估 ---
    error_count = sum(1 for c in checks if not c.passed and c.severity == "error")
    warning_count = sum(1 for c in checks if not c.passed and c.severity == "warning")

    if error_count > 0:
        result.overall_risk = "high"
    elif warning_count > 2:
        result.overall_risk = "medium"
    else:
        result.overall_risk = "low"

    result.checks = checks
    return result


def audit_cross_model_alignment(
    all_predictions: dict[tuple[str, str], pd.DataFrame],
) -> list[AuditCheck]:
    """审计多个模型之间的时间对齐。

    Parameters
    ----------
    all_predictions : dict[(model_name, task), DataFrame]
        各模型的预测结果。

    Returns
    -------
    list[AuditCheck]
    """
    checks: list[AuditCheck] = []

    # 收集所有模型的 target_day+ds 集合
    keys_by_model: dict[str, set] = {}
    for (model_name, task), df in all_predictions.items():
        if df is None or df.empty:
            checks.append(
                AuditCheck(
                    name="cross_model_alignment",
                    passed=False,
                    severity="error",
                    detail=f"{model_name}/{task}: empty predictions",
                    model_name=model_name,
                    task=task,
                )
            )
            continue
        keys = set(zip(df.get("target_day", []), df.get("ds", [])))
        keys_by_model[f"{model_name}/{task}"] = keys

    if len(keys_by_model) <= 1:
        checks.append(
            AuditCheck(
                name="cross_model_alignment",
                passed=True,
                severity="info",
                detail="Only one model present, no cross-alignment check needed",
            )
        )
        return checks

    # 计算交集
    all_keys = list(keys_by_model.values())
    intersection = all_keys[0].intersection(*all_keys[1:])

    misaligned: list[str] = []
    for label, keys in keys_by_model.items():
        missing = intersection - keys
        extra = keys - intersection
        if missing or extra:
            misaligned.append(
                f"{label}: missing {len(missing)} points, extra {len(extra)} points"
            )

    if misaligned:
        checks.append(
            AuditCheck(
                name="cross_model_alignment",
                passed=False,
                severity="error",
                detail=f"Alignment issues: {'; '.join(misaligned)}",
                evidence={"intersection_size": len(intersection)},
            )
        )
    else:
        checks.append(
            AuditCheck(
                name="cross_model_alignment",
                passed=True,
                severity="info",
                detail=f"All models aligned on {len(intersection)} target_day+ds pairs",
            )
        )

    return checks


# ---------------------------------------------------------------------------
# 单项检查
# ---------------------------------------------------------------------------


def _check_24_rows_per_day(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="24h_per_day", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    counts = df.groupby("target_day").size()
    bad_days = counts[counts != 24]
    if bad_days.empty:
        return AuditCheck(
            name="24h_per_day", passed=True, severity="info",
            detail=f"All {len(counts)} days have exactly 24 hours",
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="24h_per_day", passed=False, severity="error",
        detail=f"{len(bad_days)} days have != 24 hours: {dict(bad_days)}",
        model_name=model_name, task=task,
        evidence={"bad_days": bad_days.to_dict()},
    )


def _check_duplicate_ds(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="duplicate_ds", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    dupes = df[df.duplicated(subset=["target_day", "ds"], keep=False)]
    if dupes.empty:
        return AuditCheck(
            name="duplicate_ds", passed=True, severity="info",
            detail="No duplicate (target_day, ds) pairs",
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="duplicate_ds", passed=False, severity="error",
        detail=f"Found {len(dupes)} duplicate (target_day, ds) rows",
        model_name=model_name, task=task,
    )


def _check_missing_hours(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="missing_midnight", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    has_midnight_prev_day = any(df["hour_business"] == 24)
    has_1am = any(df["hour_business"] == 1)
    issues = []
    if not has_midnight_prev_day:
        issues.append("missing hour_business=24 (midnight)")
    if not has_1am:
        issues.append("missing hour_business=1 (01:00)")
    if issues:
        return AuditCheck(
            name="missing_midnight", passed=False, severity="error",
            detail="; ".join(issues),
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="missing_midnight", passed=True, severity="info",
        detail="Both hour_business=1 and 24 present",
        model_name=model_name, task=task,
    )


def _check_hour_business_range(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="hour_business_range", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    hours = df["hour_business"].dropna().astype(int)
    bad = hours[(hours < 1) | (hours > 24)]
    if bad.empty:
        return AuditCheck(
            name="hour_business_range", passed=True, severity="info",
            detail="All hour_business values in 1-24",
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="hour_business_range", passed=False, severity="error",
        detail=f"Found {len(bad)} values outside 1-24 range: {bad.unique().tolist()}",
        model_name=model_name, task=task,
    )


def _check_period_labels(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="period_labels", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    if "period" not in df.columns:
        return AuditCheck(
            name="period_labels", passed=False, severity="error",
            detail="period column missing",
            model_name=model_name, task=task,
        )
    if "hour_business" not in df.columns:
        return AuditCheck(
            name="period_labels", passed=False, severity="error",
            detail="hour_business column missing, cannot verify period",
            model_name=model_name, task=task,
        )

    df_check = df.dropna(subset=["hour_business"]).copy()
    if df_check.empty:
        return AuditCheck(
            name="period_labels", passed=False, severity="error",
            detail="All hour_business values are NaN",
            model_name=model_name, task=task,
        )

    df_check["_expected_period"] = df_check["hour_business"].apply(assign_period)
    mismatched = df_check[df_check["period"] != df_check["_expected_period"]]

    if mismatched.empty:
        return AuditCheck(
            name="period_labels", passed=True, severity="info",
            detail="All period labels match hour_business",
            model_name=model_name, task=task,
        )
    bad_summary = mismatched.groupby(["hour_business", "period"]).size().head(10).to_dict()
    return AuditCheck(
        name="period_labels", passed=False, severity="warning",
        detail=f"Found {len(mismatched)} mismatched period labels",
        model_name=model_name, task=task,
        evidence={"mismatches": {str(k): v for k, v in bad_summary.items()}},
    )


def _check_business_day(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    """验证 business_day（如果存在）与实际日期的关系。

    01:00-23:00 属于当天 business_day；次日 00:00 是 hour_business=24，
    归属于前一天 business_day。
    """
    if df is None or df.empty:
        return AuditCheck(
            name="business_day", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    if "business_day" not in df.columns:
        return AuditCheck(
            name="business_day", passed=True, severity="info",
            detail="No business_day column (using target_day)",
            model_name=model_name, task=task,
        )
    # 基本检查：business_day 不应在 target_day 之后
    if "target_day" in df.columns:
        df_check = df.dropna(subset=["business_day", "target_day"]).copy()
        if not df_check.empty:
            invalid = df_check[df_check["business_day"] > df_check["target_day"]]
            if not invalid.empty:
                return AuditCheck(
                    name="business_day", passed=False, severity="warning",
                    detail=f"Found {len(invalid)} rows with business_day > target_day",
                    model_name=model_name, task=task,
                )
    return AuditCheck(
        name="business_day", passed=True, severity="info",
        detail="business_day column present and consistent",
        model_name=model_name, task=task,
    )


def _check_pred_nan(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="y_pred_nan", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    if "y_pred" not in df.columns:
        return AuditCheck(
            name="y_pred_nan", passed=False, severity="error",
            detail="y_pred column missing",
            model_name=model_name, task=task,
        )
    nan_count = df["y_pred"].isna().sum()
    if nan_count == 0:
        return AuditCheck(
            name="y_pred_nan", passed=True, severity="info",
            detail="No NaN values in y_pred",
            model_name=model_name, task=task,
        )
    nan_pct = nan_count / len(df) * 100
    return AuditCheck(
        name="y_pred_nan", passed=False,
        severity="error" if nan_pct > 10 else "warning",
        detail=f"Found {nan_count} NaN values ({nan_pct:.1f}%) in y_pred",
        model_name=model_name, task=task,
        evidence={"nan_count": nan_count, "nan_pct": nan_pct},
    )


def _check_pred_outliers(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="y_pred_outliers", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    if "y_pred" not in df.columns:
        return AuditCheck(
            name="y_pred_outliers", passed=False, severity="error",
            detail="y_pred column missing",
            model_name=model_name, task=task,
        )
    y_pred = df["y_pred"].dropna()
    if y_pred.empty:
        return AuditCheck(
            name="y_pred_outliers", passed=False, severity="error",
            detail="All y_pred values are NaN",
            model_name=model_name, task=task,
        )

    high = y_pred[y_pred > MAD_PRICE_UPPER]
    low = y_pred[y_pred < MAD_PRICE_LOWER]
    extreme_count = len(high) + len(low)

    if extreme_count == 0:
        return AuditCheck(
            name="y_pred_outliers", passed=True, severity="info",
            detail=f"y_pred range: [{y_pred.min():.1f}, {y_pred.max():.1f}]",
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="y_pred_outliers", passed=False, severity="warning",
        detail=(
            f"Found {extreme_count} extreme values "
            f"(> {MAD_PRICE_UPPER}: {len(high)}, < {MAD_PRICE_LOWER}: {len(low)})"
        ),
        model_name=model_name, task=task,
        evidence={
            "high_count": len(high),
            "low_count": len(low),
            "max_pred": float(y_pred.max()),
            "min_pred": float(y_pred.min()),
        },
    )


def _check_coverage(
    df: pd.DataFrame, model_name: str, task: str
) -> AuditCheck:
    if df is None or df.empty:
        return AuditCheck(
            name="coverage", passed=False, severity="error",
            detail="No predictions", model_name=model_name, task=task,
        )
    total_rows = len(df)
    if "target_day" in df.columns:
        n_days = df["target_day"].nunique()
        expected_rows = n_days * 24
        coverage = total_rows / expected_rows * 100 if expected_rows > 0 else 0
    else:
        n_days = 0
        coverage = 0

    if coverage >= 99.9:
        return AuditCheck(
            name="coverage", passed=True, severity="info",
            detail=f"Coverage: {coverage:.1f}% ({total_rows}/{n_days * 24})",
            model_name=model_name, task=task,
        )
    return AuditCheck(
        name="coverage", passed=False,
        severity="error" if coverage < 90 else "warning",
        detail=f"Coverage: {coverage:.1f}% ({total_rows}/{n_days * 24} expected)",
        model_name=model_name, task=task,
        evidence={"coverage_pct": coverage, "n_days": n_days},
    )
