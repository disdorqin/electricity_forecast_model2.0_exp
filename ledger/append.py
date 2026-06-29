# -*- coding: utf-8 -*-
"""Ledger Append — 将每日 pipeline 产出物追加到账本。

从 production_pipeline 的标准输出目录读取：
  outputs/{run_date}/
    ├── final/
    │   ├── dayahead_final_predictions.csv
    │   └── realtime_final_predictions.csv   (或 corrected)
    ├── dayahead/
    │   ├── real/all_model_forecasts_long.csv
    │   └── fused/fused_predictions.csv
    └── realtime/
        ├── real/all_model_forecasts_long.csv
        ├── fused/fused_predictions.csv
        └── final/realtime_final_predictions_corrected.csv

核心逻辑：
  1. 定位 pipeline 产出文件
  2. 读取各模型预测、融合预测、分类器修正
  3. 按 (forecast_date, hour_business, target, model_name) 对齐
  4. 追加到 ledger CSV
  5. 去重（避免重复写入同一天）
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledger.schema import (
    LEDGER_COLUMNS,
    LEDGER_DTYPES,
    DEFAULT_LEDGER_DIR,
    ledger_path,
    make_empty_ledger,
    validate_ledger_schema,
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
    business_hour_from_timestamp,
)

logger = logging.getLogger(__name__)

_OUTPUT_ROOT = Path("outputs")



def ledger_append_from_pipeline_run(
    run_date: str,
    *,
    output_root: str | Path = "outputs",
    ledger_dir: str | Path | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """从某日 pipeline 产出物读取预测并追加到账本。

    Parameters
    ----------
    run_date : str
        运行日期（YYYY-MM-DD），对应 outputs/{run_date}/ 目录。
    output_root : str or Path
        pipeline 输出根目录，默认 "outputs"。
    ledger_dir : str or Path or None
        账本目录。默认 data/local_ledger/。
    force : bool
        如果当日已存在账本行，是否覆盖重写。
    dry_run : bool
        仅扫描并汇报，不实际写入。

    Returns
    -------
    dict
        包含 appended_rows、errors、warnings 的统计信息。
    """
    output_root = Path(output_root)
    run_dir = output_root / run_date
    ledger_dir = Path(ledger_dir) if ledger_dir else Path(f"{DEFAULT_LEDGER_DIR}")

    if not run_dir.exists():
        return {"status": "error", "message": f"Run dir not found: {run_dir}", "appended_rows": 0}

    # 1. 扫描 pipeline 产出
    try:
        assets = find_pipeline_outputs(run_date, output_root=output_root)
    except Exception as exc:
        return {"status": "error", "message": f"Failed to scan outputs: {exc}", "appended_rows": 0}

    if not assets or all(v is None for v in assets.values()):
        return {"status": "error", "message": f"No pipeline outputs found for {run_date}", "appended_rows": 0}

    logger.info(
        "Found pipeline outputs for %s: final_da=%s, final_rt=%s, "
        "fused_da=%s, fused_rt=%s, forecast_da=%s, forecast_rt=%s",
        run_date,
        assets.get("final_da"), assets.get("final_rt"),
        assets.get("fused_da"), assets.get("fused_rt"),
        assets.get("forecast_da"), assets.get("forecast_rt"),
    )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []

    pipeline_version = _read_pipeline_version(run_dir)
    final_rt_corrected = assets.get("final_rt_corrected")

    for target in ("dayahead", "realtime"):
        forecast_path = assets.get(f"forecast_{target[:2]}")
        fused_path = assets.get(f"fused_{target[:2]}")
        final_path = assets.get(f"final_{target[:2]}")

        if final_path is None:
            warnings.append(f"No final output for {target}, skipping")
            continue

        forecast_df = _safe_read_csv(forecast_path)
        fused_df = _safe_read_csv(fused_path)
        final_df = _safe_read_csv(final_path)

        if forecast_df is None and final_df is None:
            warnings.append(f"No data available for {target}, skipping")
            continue

        cutoff_desc = _determine_cutoff_desc(target, run_date)
        target_rows = _build_target_rows(
            target=target,
            run_date=run_date,
            forecast_df=forecast_df,
            fused_df=fused_df,
            final_df=final_df,
            corrected_path=final_rt_corrected if target == "realtime" else None,
            pipeline_version=pipeline_version,
            cutoff_desc=cutoff_desc,
            run_dir=run_dir,
            errors=errors,
            warnings=warnings,
        )
        rows.extend(target_rows)

    if not rows:
        return {
            "status": "error",
            "message": "Zero rows constructed from pipeline outputs",
            "appended_rows": 0,
            "errors": errors,
            "warnings": warnings,
        }

    logger.info("Constructed %d ledger rows for %s", len(rows), run_date)

    if dry_run:
        return {
            "status": "dry_run",
            "appended_rows": len(rows),
            "n_models": len(set(r[MODEL_NAME] for r in rows)),
            "forecast_dates": sorted(set(r[FORECAST_DATE] for r in rows)),
            "errors": errors,
            "warnings": warnings,
        }

    result = _append_to_ledger(rows, ledger_dir, force=force, errors=errors, warnings=warnings)
    result["errors"] = errors
    result["warnings"] = warnings
    return result



def find_pipeline_outputs(
    run_date: str,
    *,
    output_root: str | Path = "outputs",
) -> dict[str, Path | None]:
    """扫描 production_pipeline 的标准产出文件。

    Returns
    -------
    dict
        Keys: final_da, final_rt, final_rt_corrected, fused_da, fused_rt,
              forecast_da, forecast_rt
    """
    output_root = Path(output_root)
    run_dir = output_root / run_date

    return {
        "final_da": _find_file(run_dir / "final" / "dayahead_final_predictions.csv"),
        "final_rt": _find_file(run_dir / "final" / "realtime_final_predictions.csv"),
        "final_rt_corrected": _find_file(run_dir / "final" / "realtime_final_predictions_corrected.csv"),
        "fused_da": _find_file(run_dir / "dayahead" / "fused" / "fused_predictions.csv"),
        "fused_rt": _find_file(run_dir / "realtime" / "fused" / "fused_predictions.csv"),
        "forecast_da": _find_file(run_dir / "dayahead" / "real" / "all_model_forecasts_long.csv"),
        "forecast_rt": _find_file(run_dir / "realtime" / "real" / "all_model_forecasts_long.csv"),
    }


def _find_file(path: Path) -> Path | None:
    """如果文件存在且非空则返回，否则返回 None。"""
    if path.exists() and path.stat().st_size > 0:
        return path
    return None


def _safe_read_csv(path: Path | None) -> pd.DataFrame | None:
    """安全读取 CSV，失败时返回 None。"""
    if path is None:
        return None
    try:
        for enc in ("utf-8", "gbk", "gb18030"):
            try:
                df = pd.read_csv(path, encoding=enc)
                if not df.empty:
                    return df
            except (UnicodeDecodeError, LookupError):
                continue
        logger.warning("Could not decode CSV with any encoding: %s", path)
        return None
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _read_pipeline_version(run_dir: Path) -> str:
    """从 manifest 读取 pipeline 版本。"""
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            return manifest.get("pipeline_version", "r3d_tap_gef_v1")
        except Exception:
            pass
    return "r3d_tap_gef_v1"


def _determine_cutoff_desc(target: str, run_date: str) -> str:
    """返回数据截止描述。"""
    if target == "dayahead":
        return f"dayahead_{run_date}_08:00"
    else:
        return f"realtime_{run_date}_14:00"





def _calc_period(hour_business):
    if 1 <= hour_business <= 8:
        return "1_8"
    elif 9 <= hour_business <= 16:
        return "9_16"
    else:
        return "17_24"


def _ensure_business_cols(df):
    """Ensure business_day and hour_business columns exist."""
    df = df.copy()
    if FORECAST_DATE not in df.columns:
        for col in ("business_day", "target_day", "date"):
            if col in df.columns:
                df[FORECAST_DATE] = df[col].astype(str)
                break
    if FORECAST_DATE not in df.columns and "ds" in df.columns:
        ts = pd.to_datetime(df["ds"], errors="coerce")
        df[FORECAST_DATE] = ts.apply(
            lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.notna(t) and t.hour == 0
            else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
        )
    if HOUR_BUSINESS not in df.columns:
        for col in ("hour", "hour_bus", "h"):
            if col in df.columns:
                df[HOUR_BUSINESS] = pd.to_numeric(df[col], errors="coerce")
                break
    if HOUR_BUSINESS not in df.columns and "ds" in df.columns:
        ts = pd.to_datetime(df["ds"], errors="coerce")
        df[HOUR_BUSINESS] = ts.dt.hour.replace({0: 24})
    return df


def _append_to_ledger(new_rows, ledger_dir, *, force=False, errors, warnings):
    """Append rows to ledger CSV with dedup."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(new_rows)
    targets_in_batch = new_df[TARGET].unique().tolist()
    results = {"appended_rows": len(new_rows), "files": {}}
    for single_target in targets_in_batch:
        target_df = new_df[new_df[TARGET] == single_target].copy()
        tgt_path = ledger_dir / f"ledger_{single_target}.csv"
        n_written = _append_df(target_df, tgt_path, force=force, warnings=warnings)
        results["files"][str(tgt_path)] = n_written
    merged_path = ledger_dir / "ledger.csv"
    n_written_merged = _append_df(new_df, merged_path, force=force, warnings=warnings)
    results["files"][str(merged_path)] = n_written_merged
    return results


def _append_df(df, path, *, force=False, warnings):
    """Append DataFrame to CSV with dedup by key columns."""
    out_cols = [c for c in LEDGER_COLUMNS if c in df.columns]
    df_out = df[out_cols].copy()
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path)
            for c in out_cols:
                if c not in existing.columns:
                    existing[c] = float("nan")
            dedup_keys = [RUN_DATE, FORECAST_DATE, HOUR_BUSINESS, TARGET, MODEL_NAME]
            present_keys = [k for k in dedup_keys if k in existing.columns and k in df_out.columns]
            if present_keys:
                existing_keys = existing[present_keys].drop_duplicates()
                new_keys = df_out[present_keys]
                merge_keys = new_keys.merge(existing_keys, on=present_keys, how="left", indicator=True)
                dup_mask = merge_keys["_merge"] == "both"
                n_dup = dup_mask.sum()
                if n_dup > 0 and not force:
                    warnings.append(f"{path.name}: {n_dup} duplicate rows detected. Use --force to overwrite.")
                    df_out = df_out[~dup_mask.values].copy()
                elif n_dup > 0 and force:
                    warnings.append(f"{path.name}: force=True, removing {n_dup} existing rows")
                    for _, dup_row in df_out[dup_mask.values][present_keys].iterrows():
                        mask = True
                        for k in present_keys:
                            mask = mask & (existing[k] == dup_row[k])
                        existing = existing[~mask]
                    existing.to_csv(path, index=False)
                if not df_out.empty:
                    df_out.to_csv(path, mode="a", header=False, index=False)
                    n_new = len(df_out)
                    return n_new
                else:
                    return 0
            else:
                df_out.to_csv(path, mode="a", header=False, index=False)
                return len(df_out)
        except Exception as exc:
            logger.warning("Error reading ledger %s: %s. Overwriting.", path, exc)
            df_out.to_csv(path, index=False)
            return len(df_out)
    else:
        df_out.to_csv(path, index=False)
        return len(df_out)


def _build_target_rows(
    *,
    target: str,
    run_date: str,
    forecast_df,
    fused_df,
    final_df,
    corrected_path,
    pipeline_version: str,
    cutoff_desc: str,
    run_dir,
    errors: list,
    warnings: list,
):
    """Build ledger rows for one target."""
    if final_df is None:
        return []
    final_df = _ensure_business_cols(final_df)
    if FORECAST_DATE not in final_df.columns or HOUR_BUSINESS not in final_df.columns:
        errors.append(f"{target}/final: missing forecast_date or hour_business")
        return []
    corrected_df = None
    if corrected_path and corrected_path.exists():
        corrected_df = _safe_read_csv(corrected_path)
        if corrected_df is not None:
            corrected_df = _ensure_business_cols(corrected_df)
    final_df[HOUR_BUSINESS] = pd.to_numeric(final_df[HOUR_BUSINESS], errors="coerce").astype("Int64")
    key_cols = [FORECAST_DATE, HOUR_BUSINESS]
    if TIMESTAMP in final_df.columns:
        key_cols.append(TIMESTAMP)
    models = []
    if forecast_df is not None and MODEL_NAME in forecast_df.columns:
        forecast_df = _ensure_business_cols(forecast_df)
        forecast_df[HOUR_BUSINESS] = pd.to_numeric(forecast_df[HOUR_BUSINESS], errors="coerce").astype("Int64")
        models = sorted(forecast_df[MODEL_NAME].unique())
    elif MODEL_NAME in final_df.columns:
        models = sorted(final_df[MODEL_NAME].unique())
    else:
        models = ["unknown"]
    final_model_col = MODEL_NAME if MODEL_NAME in final_df.columns else None
    base_pairs = final_df[key_cols].drop_duplicates().dropna(subset=[FORECAST_DATE, HOUR_BUSINESS])
    rows = []
    now_iso = __import__("datetime").datetime.now().isoformat()
    run_dir_str = str(run_dir)
    for _, pair_row in base_pairs.iterrows():
        bd = str(pair_row[FORECAST_DATE])
        hb = int(pair_row[HOUR_BUSINESS])
        ts_val = pair_row.get(TIMESTAMP, None)
        import pandas as _pd
        if _pd.isna(ts_val) or ts_val is None:
            from ledger.schema import timestamp_from_business_hour
            ts_val = timestamp_from_business_hour(bd, hb)
        mask_final = ((final_df[FORECAST_DATE].astype(str) == bd) & (final_df[HOUR_BUSINESS] == hb))
        final_row = final_df[mask_final]
        final_pred_val = None
        if not final_row.empty:
            for col in ("y_pred", "y_fused", "final_pred"):
                if col in final_row.columns:
                    val = _pd.to_numeric(final_row[col].iloc[0], errors="coerce")
                    if _pd.notna(val):
                        final_pred_val = val
                        break
        base_fused_val = None
        if fused_df is not None:
            fused_df = _ensure_business_cols(fused_df)
            fused_df[HOUR_BUSINESS] = _pd.to_numeric(fused_df[HOUR_BUSINESS], errors="coerce").astype("Int64")
            mask_fused = ((fused_df[FORECAST_DATE].astype(str) == bd) & (fused_df[HOUR_BUSINESS] == hb))
            fused_row = fused_df[mask_fused]
            if not fused_row.empty:
                for col in ("y_pred", "y_fused"):
                    if col in fused_row.columns:
                        val = _pd.to_numeric(fused_row[col].iloc[0], errors="coerce")
                        if _pd.notna(val):
                            base_fused_val = val
                            break
        spike_corrected_val = None
        if corrected_df is not None:
            corrected_df = _ensure_business_cols(corrected_df)
            mask_corr = ((corrected_df[FORECAST_DATE].astype(str) == bd) & (corrected_df[HOUR_BUSINESS] == hb))
            corr_row = corrected_df[mask_corr]
            if not corr_row.empty:
                for col in ("y_fused_corrected", "y_pred", "final_pred"):
                    if col in corr_row.columns:
                        val = _pd.to_numeric(corr_row[col].iloc[0], errors="coerce")
                        if _pd.notna(val):
                            spike_corrected_val = val
                            break
        period = _calc_period(hb)
        for model in models:
            y_pred_val = None
            if forecast_df is not None and MODEL_NAME in forecast_df.columns:
                mask_model = ((forecast_df[FORECAST_DATE].astype(str) == bd) & (forecast_df[HOUR_BUSINESS] == hb) & (forecast_df[MODEL_NAME] == model))
                model_row = forecast_df[mask_model]
                if not model_row.empty:
                    for col in (Y_PRED, "pred", "prediction", "y"):
                        if col in model_row.columns:
                            val = _pd.to_numeric(model_row[col].iloc[0], errors="coerce")
                            if _pd.notna(val):
                                y_pred_val = val
                                break
            if y_pred_val is None and final_model_col is not None:
                mask_final_model = mask_final & (final_df[final_model_col] == model)
                final_model_row = final_df[mask_final_model]
                if not final_model_row.empty:
                    for col in (Y_PRED, "pred", "prediction"):
                        if col in final_model_row.columns:
                            val = _pd.to_numeric(final_model_row[col].iloc[0], errors="coerce")
                            if _pd.notna(val):
                                y_pred_val = val
                                break
            row = {
                RUN_DATE: run_date,
                FORECAST_DATE: bd,
                HOUR_BUSINESS: hb,
                TIMESTAMP: str(ts_val),
                TARGET: target,
                MODEL_NAME: model,
                Y_PRED: y_pred_val if y_pred_val is not None else float("nan"),
                BASE_FUSED_PRED: base_fused_val if base_fused_val is not None else float("nan"),
                SPIKE_CORRECTED_PRED: spike_corrected_val if spike_corrected_val is not None else float("nan"),
                FINAL_PRED: final_pred_val if final_pred_val is not None else float("nan"),
                Y_TRUE: float("nan"),
                PERIOD: period,
                AVAILABLE_DATA_CUTOFF: cutoff_desc,
                PIPELINE_VERSION: pipeline_version,
                SOURCE_FILE: run_dir_str,
                CREATED_AT: now_iso,
            }
            rows.append(row)
    return rows
