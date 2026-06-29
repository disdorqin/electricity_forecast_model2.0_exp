#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_extreme_events.py - Extreme price event diagnostic tool (Agent 2)

Business context:
  Realtime high-price spike miss is severe (e.g. true=1000, pred=400).
  Focus months: 2025-11, 2025-12, and 9_16 day segment.
  This tool diagnoses first. Does NOT modify final pipeline or negative price logic.

Extreme event types:
  - high_spike:  realtime_price > high threshold
  - low_valley:   realtime_price < low threshold (diagnosis only)
  - spread_extreme: |realtime - dayahead| > spread threshold
  - residual_underestimate: y_true - y_pred > underestimate threshold
  - residual_overestimate:  y_pred - y_true > overestimate threshold

Threshold strategy (quantile-based, no fixed business thresholds):
  - Global quantile: global_high_quantile / global_low_quantile
  - Period-aware quantile: per 1_8 / 9_16 / 17_24

Data leakage note:
  This script uses actual values for "diagnosis" (looks back with D+1 truth).
  Reports clearly mark "diagnosis available" vs "forecast available".
  Offline diagnosis does NOT change any model training or prediction pipeline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("diagnose_extreme_events")

# Chinese to English column name mapping
CN_COLUMN_MAP = {
    "时刻": "ds",
    "日前电价": "dayahead_price",
    "实时电价": "realtime_price",
    "日前出清价": "dayahead_price",
    "实时出清价": "realtime_price",
    "地方电厂总加预测值": "local_plant_forecast",
    "联络线受电负荷预测值": "tie_line_forecast",
    "风电总加预测值": "wind_forecast",
    "光伏总加预测值": "solar_forecast",
    "核电总加预测值": "nuclear_forecast",
    "自备机组总加预测值": "self_unit_forecast",
    "试验机组总加预测值": "test_unit_forecast",
    "直调负荷预测值": "load_forecast",
    "竞价空间预测值": "bidding_space_forecast",
    "新能源总加预测值": "renewable_forecast",
    "地方电厂总加实际值": "local_plant_actual",
    "联络线受电负荷实际值": "tie_line_actual",
    "风电总加实际值": "wind_actual",
    "光伏总加实际值": "solar_actual",
    "核电总加实际值": "nuclear_actual",
    "自备机组总加实际值": "self_unit_actual",
    "试验机组总加实际值": "test_unit_actual",
    "直调负荷实际值": "load_actual",
    "竞价空间实际值": "bidding_space_actual",
    "新能源总加实际值": "renewable_actual",
}

EXOGENOUS_FEATURES = [
    "load_forecast", "load_actual",
    "wind_forecast", "wind_actual",
    "solar_forecast", "solar_actual",
    "renewable_forecast", "renewable_actual",
    "bidding_space_forecast", "bidding_space_actual",
    "tie_line_forecast", "tie_line_actual",
    "nuclear_forecast", "nuclear_actual",
]

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extreme electricity price event diagnostic tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.xlsx", help="Path to raw data (xlsx or csv)")
    parser.add_argument("--runs-root", default=None, help="Prediction run root (alternative way to locate predictions)")
    parser.add_argument("--prediction-pack", default=None, help="Pre-built prediction pack CSV")
    parser.add_argument("--pred-dir", default=None, help="Directory with prediction CSV(s)")
    parser.add_argument("--prediction-path", default=None, dest="pred_path", help="Single prediction CSV")
    parser.add_argument("--target", default="realtime", choices=["dayahead","realtime","both"], help="Market")
    parser.add_argument("--start-date", default="2025-11-01", help="Start date")
    parser.add_argument("--end-date", default="2025-12-31", help="End date")
    parser.add_argument("--out-dir", default="reports/local/p0_full_run/extreme_events", help="Output directory")
    parser.add_argument("--high-quantile", type=float, default=0.95, help="High quantile")
    parser.add_argument("--low-quantile", type=float, default=0.10, help="Low quantile")
    parser.add_argument("--spread-quantile", type=float, default=0.95, help="Spread quantile")
    parser.add_argument("--residual-quantile", type=float, default=0.90, help="Residual quantile")
    parser.add_argument("--period-aware", default=True, action=argparse.BooleanOptionalAction, help="Period-aware")
    parser.add_argument("--max-bad-samples", type=int, default=100, help="Max bad samples")
    return parser.parse_args(argv)

def load_data(data_path):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = None
    for enc in ("gbk", "gb18030", "utf-8", "utf-8-sig"):
        try:
            raw = pd.read_csv(path, encoding=enc)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw is None:
        raise ValueError(f"Cannot read {path}")
    raw.rename(columns={c: CN_COLUMN_MAP.get(c, c) for c in raw.columns}, inplace=True)
    return raw

def parse_timestamp(df):
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    dropped = df["ds"].isna().sum()
    if dropped > 0:
        logger.warning("Dropped %d rows", dropped)
        df = df.dropna(subset=["ds"]).copy()
    return df

def compute_business_columns(df):
    df = df.copy()
    mask_midnight = df["ds"].dt.hour == 0
    df["business_day"] = np.where(mask_midnight, (df["ds"] - pd.Timedelta(days=1)).dt.date, df["ds"].dt.date)
    df["hour_business"] = np.where(mask_midnight, 24, df["ds"].dt.hour).astype(int)
    def _period(h):
        if 1 <= h <= 8: return "1_8"
        elif 9 <= h <= 16: return "9_16"
        elif 17 <= h <= 24: return "17_24"
        return "unknown"
    df["period"] = df["hour_business"].apply(_period)
    return df

def load_predictions(pred_dir, pred_path, start_date, end_date):
    if pred_dir is None and pred_path is None:
        return None
    frames = []
    if pred_path:
        paths = [Path(pred_path)]
    else:
        base = Path(pred_dir)
        if not base.exists():
            logger.warning("Prediction dir not found: %s", base)
            return None
        tap_csv = base / "validation" / "validation_tap_long_table.csv"
        if tap_csv.exists():
            paths = [tap_csv]
            logger.info("Found validation tap: %s", tap_csv)
        else:
            paths = list(base.rglob("*.csv"))
            paths = [p for p in paths if "manifest" not in p.name.lower() and "summary" not in p.name.lower()]
    if not paths:
        logger.warning("No prediction files found")
        return None
    for p in paths:
        try:
            pdf = pd.read_csv(p, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                pdf = pd.read_csv(p, encoding="gbk")
            except Exception:
                continue
        pdf["_source_file"] = p.name
        pdf.rename(columns={
            "时刻": "ds",
            "prediction": "y_pred",
            "pred": "y_pred",
            "y_pred": "y_pred",
            "actual": "y_true",
            "y_true": "y_true",
            "model": "model_name",
        }, inplace=True)
        frames.append(pdf)
    if not frames:
        return None
    merged = pd.concat(frames, ignore_index=True)
    merged["ds"] = pd.to_datetime(merged["ds"], errors="coerce")
    merged = merged.dropna(subset=["ds"]).copy()
    if "y_pred" not in merged.columns:
        merged["y_pred"] = float("nan")
    else:
        merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
    if "model_name" not in merged.columns:
        merged["model_name"] = "unknown"
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    merged = merged[(merged["ds"] >= start_ts) & (merged["ds"] < end_ts)].copy()
    logger.info("Loaded predictions: %d rows, %d models, %s ~ %s",
                len(merged), merged["model_name"].nunique(),
                merged["ds"].min(), merged["ds"].max())
    if merged.empty: return None
    return merged


def load_predictions_from_pack(pack_path, start_date, end_date):
    """Load predictions from a pre-built prediction pack CSV."""
    path = Path(pack_path)
    if not path.exists():
        logger.warning("Prediction pack not found: %s", pack_path)
        return None
    try:
        pdf = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        try:
            pdf = pd.read_csv(path, encoding="gbk")
        except Exception:
            logger.error("Cannot read prediction pack: %s", path)
            return None
    pdf.rename(columns={
        "时刻": "ds", "prediction": "y_pred", "pred": "y_pred",
        "actual": "y_true", "model": "model_name",
    }, inplace=True)
    if "ds" not in pdf.columns:
        logger.warning("Prediction pack has no ds column; cols=%s", list(pdf.columns))
        return None
    pdf["ds"] = pd.to_datetime(pdf["ds"], errors="coerce")
    pdf = pdf.dropna(subset=["ds"]).copy()
    if "y_pred" not in pdf.columns:
        pdf["y_pred"] = float("nan")
    else:
        pdf["y_pred"] = pd.to_numeric(pdf["y_pred"], errors="coerce")
    if "model_name" not in pdf.columns:
        pdf["model_name"] = "unknown"
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    pdf = pdf[(pdf["ds"] >= start_ts) & (pdf["ds"] < end_ts)].copy()
    logger.info("Loaded prediction pack: %d rows, %d models, %s ~ %s",
                len(pdf), pdf["model_name"].nunique(),
                pdf["ds"].min(), pdf["ds"].max())
    return pdf if not pdf.empty else None


def compute_thresholds(df, args):
    price_col = "realtime_price"
    thresholds = {}
    high_q = df[price_col].quantile(args.high_quantile)
    if args.period_aware:
        period_q = df.groupby("period")[price_col].quantile(args.high_quantile).to_dict()
    else:
        period_q = {}
    thresholds["high_spike"] = {"global": float(high_q), "period": {k: float(v) for k, v in period_q.items()}}
    low_q = df[price_col].quantile(args.low_quantile)
    if args.period_aware:
        period_low_q = df.groupby("period")[price_col].quantile(args.low_quantile).to_dict()
    else:
        period_low_q = {}
    thresholds["low_valley"] = {"global": float(low_q), "period": {k: float(v) for k, v in period_low_q.items()}}
    if "dayahead_price" in df.columns:
        spread_q = (df[price_col] - df["dayahead_price"]).abs().quantile(args.spread_quantile)
        if args.period_aware:
            spread_period_q = df.groupby("period").apply(lambda g: (g[price_col] - g["dayahead_price"]).abs().quantile(args.spread_quantile)).to_dict()
        else:
            spread_period_q = {}
        thresholds["spread_extreme"] = {"global": float(spread_q), "period": {k: float(v) for k, v in spread_period_q.items()}}
    else:
        thresholds["spread_extreme"] = {"global": float("nan"), "period": {}}
    return thresholds

def detect_events(df, thresholds, pred_df=None, residual_quantile=0.90):
    price_col = "realtime_price"
    records = []
    for idx, row in df.iterrows():
        true_price = row[price_col]
        period = row["period"]
        # high_spike
        if thresholds["high_spike"]["period"]:
            th_h = thresholds["high_spike"]["period"].get(period, thresholds["high_spike"]["global"])
        else:
            th_h = thresholds["high_spike"]["global"]
        if pd.notna(true_price) and true_price > th_h:
            records.append(_build_event_record(row, "high_spike", th_h, true_price, f"rt>{th_h:.0f}"))
        # low_valley
        if thresholds["low_valley"]["period"]:
            th_l = thresholds["low_valley"]["period"].get(period, thresholds["low_valley"]["global"])
        else:
            th_l = thresholds["low_valley"]["global"]
        if pd.notna(true_price) and true_price < th_l:
            records.append(_build_event_record(row, "low_valley", th_l, true_price, f"rt<{th_l:.0f}"))
        # spread_extreme
        if "dayahead_price" in df.columns and pd.notna(row.get("dayahead_price")):
            spread = abs(true_price - row["dayahead_price"]) if pd.notna(true_price) else float("nan")
            if thresholds["spread_extreme"]["period"]:
                th_s = thresholds["spread_extreme"]["period"].get(period, thresholds["spread_extreme"]["global"])
            else:
                th_s = thresholds["spread_extreme"]["global"]
            if pd.notna(spread) and spread > th_s:
                rec = _build_event_record(row, "spread_extreme", th_s, spread, f"|rt-da|>{th_s:.0f}")
                rec["spread"] = spread
                records.append(rec)
    events_df = pd.DataFrame(records) if records else pd.DataFrame()
    if pred_df is not None and not pred_df.empty:
        res_events = _detect_residual_events(df, pred_df, residual_quantile)
        if not res_events.empty:
            events_df = pd.concat([events_df, res_events], ignore_index=True)
    if not events_df.empty:
        events_df = events_df.sort_values("ds").reset_index(drop=True)
    return events_df

def _build_event_record(row, event_type, threshold, value, trigger_expr):
    rec = {
        "ds": row["ds"],
        "business_day": row.get("business_day"),
        "hour_business": row.get("hour_business"),
        "period": row.get("period"),
        "event_type": event_type,
        "threshold": round(float(threshold), 2),
        "value": round(float(value), 2) if pd.notna(value) else None,
        "trigger": trigger_expr,
        "true_price": row.get("realtime_price"),
        "dayahead_price": row.get("dayahead_price"),
        "load_actual": row.get("load_actual"),
        "load_forecast": row.get("load_forecast"),
        "wind_actual": row.get("wind_actual"),
        "solar_actual": row.get("solar_actual"),
        "renewable_actual": row.get("renewable_actual"),
        "renewable_forecast": row.get("renewable_forecast"),
        "bidding_space_actual": row.get("bidding_space_actual"),
        "tie_line_actual": row.get("tie_line_actual"),
        "net_load_actual": (
            row["load_actual"] - row["renewable_actual"]
            if pd.notna(row.get("load_actual")) and pd.notna(row.get("renewable_actual"))
            else None),
        "renewable_penetration": (
            row["renewable_actual"] / (row["load_actual"] + 1e-5)
            if pd.notna(row.get("load_actual")) and pd.notna(row.get("renewable_actual"))
            else None),
    }
    return rec

def _detect_residual_events(df, pred_df, residual_quantile):
    df_align = df.copy()
    df_align["ds"] = pd.to_datetime(df_align["ds"])
    pred_agg = pred_df.copy()
    pred_agg["ds"] = pd.to_datetime(pred_agg["ds"])
    if "model_name" in pred_agg.columns and pred_agg["model_name"].nunique() > 1:
        logger.info("Averaging %d models", pred_agg["model_name"].nunique())
        pred_agg = pred_agg.groupby("ds", as_index=False).agg(y_pred=("y_pred", "mean"), n_models=("y_pred", "count"))
    elif "y_pred" in pred_agg.columns:
        pred_agg = pred_agg[["ds", "y_pred"]].drop_duplicates(subset=["ds"]).copy()
        pred_agg["n_models"] = 1
    aligned = df_align.merge(pred_agg[["ds", "y_pred", "n_models"]], on="ds", how="inner")
    if aligned.empty:
        logger.warning("No aligned data points")
        return pd.DataFrame()
    aligned["residual"] = aligned["realtime_price"] - aligned["y_pred"]
    aligned["underestimate"] = aligned["residual"]
    aligned["overestimate"] = -aligned["residual"]
    under_th = aligned["underestimate"].quantile(residual_quantile)
    over_th = aligned["overestimate"].quantile(residual_quantile)
    records = []
    for _, row in aligned.iterrows():
        if pd.notna(row["underestimate"]) and row["underestimate"] > under_th:
            rec = _build_event_record(row, "residual_underestimate", under_th, row["underestimate"], f"true-pred>{under_th:.0f}")
            rec["y_pred"] = row.get("y_pred")
            rec["residual"] = row.get("residual")
            records.append(rec)
        if pd.notna(row["overestimate"]) and row["overestimate"] > over_th:
            rec = _build_event_record(row, "residual_overestimate", over_th, row["overestimate"], f"pred-true>{over_th:.0f}")
            rec["y_pred"] = row.get("y_pred")
            rec["residual"] = row.get("residual")
            records.append(rec)
    logger.info("Residual events: %d under + %d over",
        sum(1 for r in records if r["event_type"] == "residual_underestimate"),
        sum(1 for r in records if r["event_type"] == "residual_overestimate"))
    return pd.DataFrame(records) if records else pd.DataFrame()

def compute_monthly_summary(events_df):
    if events_df.empty: return pd.DataFrame()
    spike = events_df[events_df["event_type"] == "high_spike"].copy()
    if spike.empty: return pd.DataFrame()
    spike["month"] = spike["ds"].dt.strftime("%Y-%m")
    def _agg(g):
        n_9 = (g["period"] == "9_16").sum()
        return pd.Series({
            "event_count": len(g),
            "avg_price": g["true_price"].mean(),
            "max_price": g["true_price"].max(),
            "p9_16_ratio": n_9 / max(len(g), 1),
            "n_9_16": n_9,
        })
    return spike.groupby("month", group_keys=False).apply(_agg).reset_index().sort_values("month")


def compute_period_summary(events_df):
    if events_df.empty: return pd.DataFrame()
    def _agg(g):
        return pd.Series({
            "total_events": len(g),
            "high_spike": int((g["event_type"] == "high_spike").sum()),
            "low_valley": int((g["event_type"] == "low_valley").sum()),
            "spread_extreme": int((g["event_type"] == "spread_extreme").sum()),
            "residual_underestimate": int((g["event_type"] == "residual_underestimate").sum()),
            "residual_overestimate": int((g["event_type"] == "residual_overestimate").sum()),
            "avg_true_price": g["true_price"].mean(),
        })
    return events_df.groupby("period", group_keys=False).apply(_agg).reset_index()


def compute_bad_cases(events_df, max_samples=100):
    under = events_df[events_df["event_type"] == "residual_underestimate"].copy()
    if under.empty: return pd.DataFrame()
    under = under.sort_values("value", ascending=False).head(max_samples)
    cols = ["ds","business_day","hour_business","period","true_price","y_pred","residual",
            "load_actual","solar_actual","wind_actual","renewable_actual",
            "tie_line_actual","bidding_space_actual","net_load_actual","renewable_penetration"]
    return under[[c for c in cols if c in under.columns]].reset_index(drop=True)

def write_event_csv(events_df, out_dir):
    path = out_dir / "extreme_event_report.csv"
    if events_df.empty:
        path.write_text("No events found" + chr(10), encoding="utf-8-sig")
        logger.info("Empty: %s", path); return
    events_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Written: %s (%d rows)", path, len(events_df))

def write_monthly_summary(monthly_df, out_dir):
    path = out_dir / "monthly_extreme_summary.csv"
    if monthly_df.empty:
        path.write_text("month,event_count,avg_price,max_price,p9_16_ratio,n_9_16" + chr(10), encoding="utf-8-sig")
        logger.info("Empty: %s", path); return
    monthly_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Written: %s", path)

def write_period_summary(period_df, out_dir):
    path = out_dir / "period_extreme_summary.csv"
    if period_df.empty:
        path.write_text("period,total_events,high_spike,low_valley,spread_extreme,residual_underestimate,residual_overestimate,avg_true_price" + chr(10), encoding="utf-8-sig")
        logger.info("Empty: %s", path); return
    period_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Written: %s", path)

def write_bad_cases(bad_df, out_dir):
    path = out_dir / "bad_case_samples.csv"
    if bad_df.empty:
        path.write_text("No residual_underestimate events found" + chr(10), encoding="utf-8-sig")
        logger.info("Empty: %s", path); return
    bad_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Written: %s (%d rows)", path, len(bad_df))

def write_markdown_report(events_df, monthly_df, period_df, bad_df, thresholds, args, pred_available, out_dir):
    md = _build_report_md(events_df, monthly_df, period_df, bad_df, thresholds, args, pred_available)
    path = out_dir / "extreme_event_report.md"
    path.write_text(md, encoding="utf-8")
    logger.info("Written: %s", path)


def _build_report_md(events_df, monthly_df, period_df, bad_df, thresholds, args, pred_available):
    md = []
    ap = md.append
    ap("# 极端电价事件诊断报告" + "\n")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    ap(f"**生成时间**: {now_str}" + "\n")
    ap(f"**诊断目标**: {args.target}" + "\n")
    ap(f"**时间范围**: {args.start_date} ~ {args.end_date}" + "\n")
    ap(f"**高价分位数阈值**: P{args.high_quantile*100:.0f}" + "\n")
    ap(f"**低价分位数阈值**: P{args.low_quantile*100:.0f}" + "\n")
    period_label = "是" if args.period_aware else "否"
    pred_label = "是" if pred_available else "否"
    ap(f"**Period 感知**: {period_label}" + "\n")
    ap(f"**预测文件**: {pred_label}" + "\n" + "\n")

    # Data availability
    ap("---" + "\n")
    ap("## 数据可用性说明" + "\n" + "\n")
    ap("| 数据类别 | 诊断可用 | 预测可用 | 说明 |" + "\n")
    ap("|----------|----------|----------|------|" + "\n")
    ap("| 实时电价实际值 | ✅ | ❌ | 真实价格仅用于历史回看诊断 |" + "\n")
    ap("| 日前电价实际值 | ✅ | ❌ | 同上 |" + "\n")
    ap("| 负荷/新能源实际值 | ✅ | ❌ | 实际值存在未来信息 |" + "\n")
    ap("| 负荷/新能源预测值 | ✅ | ✅ | 预测值可用作特征 |" + "\n")
    ap("| 模型预测值 | ✅ | ✅ | 来自 OOF 或推理输出 |" + "\n")
    ap("| rt-da 价差 | ✅ | ❌ | 依赖未来真实价格 |" + "\n" + "\n")
    ap("> **注意**: 本报告使用真实价格进行事后诊断，不改变任何模型训练或预测管线。" + "\n" + "\n")

    # Thresholds
    ap("---" + "\n")
    ap("## 阈值设定" + "\n" + "\n")
    _add_threshold_section(md, "high_spike (高价尖峰)", thresholds["high_spike"], args.high_quantile)
    _add_threshold_section(md, "low_valley (低价谷底)", thresholds["low_valley"], args.low_quantile)
    _add_threshold_section(md, "spread_extreme (价差异常)", thresholds["spread_extreme"], args.spread_quantile)

    _add_monthly_table(md, monthly_df)
    _add_focus_analysis(md, events_df)
    _add_period_table(md, period_df)
    _add_hour_distribution(md, events_df)
    _add_bad_cases_table(md, bad_df)
    _add_exogenous_analysis(md, events_df)
    _add_recommendations(md, events_df)

    return "".join(md)

def _add_threshold_section(md, label, th_dict, quantile):
    ap = md.append
    ap('### ' + label + '\n')
    ap('- Global threshold: P{:.0f} = {:.2f}\n'.format(quantile * 100, th_dict.get('global', float('nan'))))
    if th_dict.get('period'):
        ap('- Period-aware thresholds:\n')
        for p, v in sorted(th_dict['period'].items()):
            ap('  - {}: {:.2f}\n'.format(p, v))
    else:
        ap('- Period-aware: not used\n')
    ap('\n')


def _add_monthly_table(md, monthly_df):
    ap = md.append
    ap('---\n')
    ap('## 月度极端事件统计（high_spike）\n\n')
    if monthly_df.empty:
        ap('*本月度内未检测到高价尖峰事件。*\n\n')
        return
    ap('| 月份 | 事件数 | 平均价格 | 最高价格 | 9_16占比 | 9_16次数 |\n')
    ap('|------|--------|----------|----------|----------|----------|\n')
    for _, row in monthly_df.iterrows():
        ap('| {} | {} | {:.0f} | {:.0f} | {:.1%} | {} |\n'.format(
            row['month'], int(row['event_count']),
            row['avg_price'], row['max_price'],
            row['p9_16_ratio'], int(row['n_9_16'])))
    ap('\n')


def _add_focus_analysis(md, events_df):
    ap = md.append
    ap('---\n')
    ap('## 重点时段分析（9_16 段）\n\n')
    focus = events_df[events_df['period'] == '9_16'] if not events_df.empty else pd.DataFrame()
    if focus.empty:
        ap('*9_16 时段未检测到极端事件。*\n\n')
        return
    spike_focus = focus[focus['event_type'] == 'high_spike']
    under_focus = focus[focus['event_type'] == 'residual_underestimate']
    ap('- 9_16 高价尖峰事件数: {}\n'.format(len(spike_focus)))
    ap('- 9_16 低估残差事件数: {}\n'.format(len(under_focus)))
    if not spike_focus.empty:
        ap('- 9_16 尖峰平均价格: {:.0f}\n'.format(spike_focus['true_price'].mean()))
        ap('- 9_16 尖峰最高价格: {:.0f}\n'.format(spike_focus['true_price'].max()))
    ap('\n')
    # Top-10 worst underestimates in 9_16
    if not under_focus.empty:
        worst = under_focus.sort_values('value', ascending=False).head(10)
        ap('### Top-10 严重低估（9_16 段）\n\n')
        ap('| 时间 | 小时 | 真实价格 | 预测值 | 残差 | 负荷实际 | 光伏实际 | 风电实际 | 净负荷 |\n')
        ap('|------|------|----------|--------|------|----------|----------|----------|--------|\n')
        for _, row in worst.iterrows():
            ap('| {} | {} | {:.0f} | {:.0f} | {:.0f} | {} | {} | {} | {} |\n'.format(
                row.get('ds', ''), row.get('hour_business', ''),
                row.get('true_price', 0), row.get('y_pred', 0), row.get('residual', 0),
                _fmt(row.get('load_actual')), _fmt(row.get('solar_actual')),
                _fmt(row.get('wind_actual')), _fmt(row.get('net_load_actual'))))
        ap('\n')


def _fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '-'
    return '{:.0f}'.format(v)



def _add_period_table(md, period_df):
    ap = md.append
    ap('---\n')
    ap('## 各 Period 事件分布\n\n')
    if period_df.empty:
        ap('*无事件数据。*\n\n')
        return
    ap('| Period | 总事件 | 高价尖峰 | 低价谷底 | 价差异常 | 低估残差 | 高估残差 | 平均价格 |\n')
    ap('|--------|--------|----------|----------|----------|----------|----------|----------|\n')
    for _, row in period_df.iterrows():
        ap('| {} | {} | {} | {} | {} | {} | {} | {:.0f} |\n'.format(
            row['period'], int(row['total_events']),
            int(row['high_spike']), int(row['low_valley']),
            int(row['spread_extreme']), int(row['residual_underestimate']),
            int(row['residual_overestimate']), row['avg_true_price']))
    ap('\n')


def _add_hour_distribution(md, events_df):
    ap = md.append
    ap('---\n')
    ap('## 小时分布\n\n')
    if events_df.empty:
        ap('*无事件数据。*\n\n')
        return
    spike = events_df[events_df['event_type'] == 'high_spike'].copy()
    if spike.empty:
        ap('*无高价尖峰事件。*\n\n')
        return
    spike['hour'] = spike['hour_business']
    hourly = spike.groupby('hour').agg(
        count=('event_type', 'count'),
        avg_price=('true_price', 'mean'),
        max_price=('true_price', 'max'),
    ).reset_index()
    ap('| 小时 | 尖峰次数 | 平均价格 | 最高价格 |\n')
    ap('|------|----------|----------|----------|\n')
    for _, row in hourly.iterrows():
        ap('| {} | {} | {:.0f} | {:.0f} |\n'.format(
            int(row['hour']), int(row['count']),
            row['avg_price'], row['max_price']))
    ap('\n')


def _add_bad_cases_table(md, bad_df):
    ap = md.append
    ap('---\n')
    ap('## 严重低估案例（Top {}）\n\n'.format(len(bad_df) if not bad_df.empty else 0))
    if bad_df.empty:
        ap('*无低估残差事件。*\n\n')
        return
    ap('| 日期 | 小时 | Period | 真实价格 | 预测值 | 残差 | 负荷实际 | 光伏实际 | 风电实际 | 净负荷 | 可再生渗透率 |\n')
    ap('|------|------|--------|----------|--------|------|----------|----------|----------|--------|-------------|\n')
    for _, row in bad_df.iterrows():
        ds_str = str(row.get('ds', ''))[:10] if pd.notna(row.get('ds', None)) else ''
        ap('| {} | {} | {} | {:.0f} | {:.0f} | {:.0f} | {} | {} | {} | {} | {:.1%} |\n'.format(
            ds_str,
            _fmt_short(row.get('hour_business')),
            row.get('period', ''),
            row.get('true_price', 0), row.get('y_pred', 0), row.get('residual', 0),
            _fmt_short(row.get('load_actual')), _fmt_short(row.get('solar_actual')),
            _fmt_short(row.get('wind_actual')), _fmt_short(row.get('net_load_actual')),
            row.get('renewable_penetration', 0) if pd.notna(row.get('renewable_penetration', None)) else 0))
    ap('\n')


def _fmt_short(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return '-'
    return '{:.0f}'.format(v)



def _add_exogenous_analysis(md, events_df):
    ap = md.append
    ap('---\n')
    ap('## 外生变量关联分析\n\n')
    spike = events_df[events_df['event_type'] == 'high_spike'].copy()
    if spike.empty:
        ap('*无高价尖峰事件，无法分析外生变量关联。*\n\n')
        return
    renewables = ['wind_actual', 'solar_actual', 'renewable_actual']
    loads = ['load_actual', 'load_forecast', 'net_load_actual']
    for col in renewables + loads:
        if col in spike.columns:
            vals = spike[col].dropna()
            if not vals.empty:
                if col == 'renewable_penetration':
                    ap('- {}: 尖峰时段均值={:.1%}, 中位数={:.1%}, 范围=[{:.1%}, {:.1%}]\n'.format(
                        col, vals.mean(), vals.median(), vals.min(), vals.max()))
                else:
                    ap('- {}: 尖峰时段均值={:.0f}, 中位数={:.0f}, 范围=[{:.0f}, {:.0f}]\n'.format(
                        col, vals.mean(), vals.median(), vals.min(), vals.max()))
    ap('\n')
    # Correlation with net load
    if 'net_load_actual' in spike.columns and 'true_price' in spike.columns:
        valid = spike[['net_load_actual', 'true_price']].dropna()
        if len(valid) > 5:
            corr = valid['net_load_actual'].corr(valid['true_price'])
            ap('- 尖峰时段 net_load vs true_price 相关系数: {:.3f}\n'.format(corr))
    ap('\n')


def _add_recommendations(md, events_df):
    ap = md.append
    ap('---\n')
    ap('## 诊断建议\n\n')
    spike = events_df[events_df['event_type'] == 'high_spike'] if not events_df.empty else pd.DataFrame()
    under = events_df[events_df['event_type'] == 'residual_underestimate'] if not events_df.empty else pd.DataFrame()
    n_spike = len(spike)
    n_under = len(under)
    n_9_16_spike = len(spike[spike['period'] == '9_16']) if not spike.empty else 0
    ap('1. **高价尖峰漏报严重性**: {} 次高价尖峰事件，{} 次低估残差事件\n'.format(n_spike, n_under))
    if n_9_16_spike > 0:
        ap('   - 其中 9_16 段尖峰占比: {:.1%} ({}次)\n'.format(n_9_16_spike / max(n_spike, 1), n_9_16_spike))
    ap('2. **9_16 段专项优化**: 该时段为光伏大发转负荷爬坡期，建议增加 net_load 差分特征\n')
    ap('3. **极端值裁剪**: 若模型对训练集 price cap (e.g. 1500) 敏感，可考虑单独处理 cap 时段\n')
    ap('4. **多模型融合加权**: 低估时段可提高高灵敏度模型的融合权重\n')
    ap('5. **外生变量利用**: 光伏实际值在尖峰时段偏低，建议强化 solar_actual / solar_forecast 的使用\n')
    ap('\n')


def _print_summary(events_df, monthly_df, thresholds):
    logger.info('=' * 60)
    logger.info('极端事件诊断摘要')
    logger.info('=' * 60)
    logger.info('高价尖峰阈值(global): %.2f', thresholds['high_spike']['global'])
    logger.info('低价谷底阈值(global): %.2f', thresholds['low_valley']['global'])
    if 'spread_extreme' in thresholds:
        logger.info('价差异常阈值(global): %.2f', thresholds['spread_extreme']['global'])
    if not events_df.empty:
        for etype in ['high_spike', 'low_valley', 'spread_extreme', 'residual_underestimate', 'residual_overestimate']:
            cnt = len(events_df[events_df['event_type'] == etype])
            logger.info('  %s: %d events', etype, cnt)
    if not monthly_df.empty:
        logger.info('月度统计:')
        for _, row in monthly_df.iterrows():
            logger.info('  %s: %d events, avg_price=%.0f, max_price=%.0f, 9_16_ratio=%.1f%%',
                        row['month'], int(row['event_count']), row['avg_price'],
                        row['max_price'], row['p9_16_ratio'] * 100)
    logger.info('=' * 60)


def main():
    args = parse_args()
    logger.info('Args: %s', args)
    logger.info('Loading data from: %s', args.data_path)
    df = load_data(args.data_path)
    df = parse_timestamp(df)
    logger.info('Raw rows: %d, cols: %s', len(df), list(df.columns))
    df = compute_business_columns(df)

    # Filter date range
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    df = df[(df['ds'] >= start) & (df['ds'] < end)].copy()
    logger.info('Filtered rows: %d (range: %s ~ %s)', len(df), args.start_date, args.end_date)

    if df.empty:
        logger.error('No data after filtering. Exiting.')
        sys.exit(1)

    # Load predictions (optional) — try prediction-pack first, then legacy paths
    pred_df = None
    if args.prediction_pack:
        logger.info("Loading predictions from prediction pack: %s", args.prediction_pack)
        pred_df = load_predictions_from_pack(args.prediction_pack, args.start_date, args.end_date)
    if pred_df is None:
        pred_df = load_predictions(args.pred_dir, args.pred_path, args.start_date, args.end_date)
    pred_available = pred_df is not None and not pred_df.empty

    # Compute thresholds
    thresholds = compute_thresholds(df, args)
    logger.info('Thresholds computed:')
    for key, th in thresholds.items():
        logger.info('  %s: global=%.2f, period=%s', key, th['global'], th['period'])

    # Detect events
    events_df = detect_events(df, thresholds, pred_df, args.residual_quantile)
    logger.info('Total events: %d', len(events_df))

    # Compute summaries
    monthly_df = compute_monthly_summary(events_df)
    period_df = compute_period_summary(events_df)
    bad_df = compute_bad_cases(events_df, args.max_bad_samples)

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write outputs
    write_event_csv(events_df, out_dir)
    write_monthly_summary(monthly_df, out_dir)
    write_period_summary(period_df, out_dir)
    write_bad_cases(bad_df, out_dir)
    write_markdown_report(events_df, monthly_df, period_df, bad_df, thresholds, args, pred_available, out_dir)

    # Print summary
    _print_summary(events_df, monthly_df, thresholds)

    logger.info('All outputs written to: %s', out_dir)


if __name__ == '__main__':
    main()
