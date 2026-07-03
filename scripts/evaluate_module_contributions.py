"""Decompose module contributions in the forecasting pipeline — Phase 13.

Evaluates the incremental contribution of each stage in the pipeline:

- **A.** Per-model raw predictions (sgdfnet only, timesfm only)
- **B.** Fused baseline (y_fused before intraday correction)
- **C.** Fused + IntradayTracker shadow / low_weight simulation

For each stage computes: overall sMAPE_floor50, 9_16 sMAPE, negative bucket
sMAPE, spike bucket sMAPE, and delta vs previous stage.

Usage::

    python scripts/evaluate_module_contributions.py \
        --fused-predictions  reports/local/phase13/.../monthly_fused_predictions.csv \
        --ground-truth       path/to/ground_truth.csv \
        --intraday-pack      reports/local/phase13/.../aligned_pack.csv \
        --output-dir         reports/local/phase13/real_mainline_replay
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup — ensure the corrections package is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                      # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MAIN_REPO_CANDIDATE = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp"
if _MAIN_REPO_CANDIDATE.is_dir() and str(_MAIN_REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_MAIN_REPO_CANDIDATE))

from corrections.intraday_tracker.adapter import (
    load_intraday_pack,
    normalize_intraday_pack,
    validate_intraday_pack,
)
from corrections.intraday_tracker.apply import (
    apply_intraday_tracker_correction,
)
from corrections.intraday_tracker.policy import (
    IntradayTrackerMainlineConfig,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SMAPE_FLOOR = 50
DEFAULT_OUTPUT_DIR = "reports/local/phase13/real_mainline_replay"
DEFAULT_CONFIG = "config/intraday_tracker.yaml"

# Bucket thresholds
NEGATIVE_THRESHOLD = 0.0
SPIKE_PERCENTILE = 95

# Period segments
SEGMENT_9_16_HOURS = set(range(9, 17))  # hours 9..16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialiser fallback for numpy / pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with a floor of 50 on both predictions and actuals.

    sMAPE = mean( 2 * |y_true - y_pred| / (max(|y_true|, FLOOR) + max(|y_pred|, FLOOR)) )
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    denom = np.maximum(np.abs(y_true), SMAPE_FLOOR) + np.maximum(np.abs(y_pred), SMAPE_FLOOR)
    denom = np.where(denom == 0, 1.0, denom)
    pointwise = 2.0 * np.abs(y_true - y_pred) / denom
    return float(np.mean(pointwise))


def classify_buckets(y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Classify each sample into negative / spike / normal buckets."""
    spike_threshold = float(np.percentile(y_true, SPIKE_PERCENTILE))
    masks = {
        "negative": y_true <= NEGATIVE_THRESHOLD,
        "spike": (y_true > NEGATIVE_THRESHOLD) & (y_true >= spike_threshold),
        "normal": (y_true > NEGATIVE_THRESHOLD) & (y_true < spike_threshold),
    }
    return masks


def compute_stage_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
    label: str = "",
) -> dict:
    """Compute comprehensive metrics for a pipeline stage.

    Returns a dict with overall, 9_16 segment, and per-bucket sMAPE values.
    """
    result = {"label": label, "n_samples": len(y_true)}

    # Overall sMAPE
    if len(y_true) > 0:
        result["overall_smape"] = smape_floor50(y_true, y_pred)
    else:
        result["overall_smape"] = float("nan")

    # 9_16 segment sMAPE
    mask_9_16 = np.isin(hours, list(SEGMENT_9_16_HOURS))
    if mask_9_16.sum() > 0:
        result["segment_9_16_smape"] = smape_floor50(y_true[mask_9_16], y_pred[mask_9_16])
        result["segment_9_16_n"] = int(mask_9_16.sum())
    else:
        result["segment_9_16_smape"] = float("nan")
        result["segment_9_16_n"] = 0

    # Per-bucket sMAPE
    masks = classify_buckets(y_true)
    for bucket_name, mask in masks.items():
        if mask.sum() > 0:
            result[f"{bucket_name}_bucket_smape"] = smape_floor50(y_true[mask], y_pred[mask])
            result[f"{bucket_name}_bucket_n"] = int(mask.sum())
        else:
            result[f"{bucket_name}_bucket_smape"] = float("nan")
            result[f"{bucket_name}_bucket_n"] = 0

    return result


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fused_predictions(path: str) -> pd.DataFrame:
    """Load fused predictions CSV and normalise column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Fused predictions loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Normalise prediction column
    if "y_fused" not in df.columns:
        for alias in ("y_pred", "rt_pred", "rt_pred_final"):
            if alias in df.columns:
                df["y_fused"] = df[alias]
                logger.info("Mapped '%s' -> 'y_fused'", alias)
                break

    if "y_fused" not in df.columns:
        raise ValueError(f"Cannot find prediction column in {path}.")

    if "rt_pred" not in df.columns:
        df["rt_pred"] = df["y_fused"]

    for col in ("business_day", "hour_business"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype(int)
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth CSV with Chinese column names.

    Handles:
    - Chinese columns: 时刻 (timestamp), 实时电价 (rt_price)
    - Standard columns: business_day, hour_business, rt_actual

    Timestamp convention:
      - timestamp D 00:00  -> business_day D-1, hour_business 24
      - timestamp D HH:00 (HH>=1) -> business_day D, hour HH
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Ground truth loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Check if already in business_day format
    if "business_day" in df.columns and "hour_business" in df.columns:
        # Already normalised — just find the actual column
        if "rt_actual" not in df.columns:
            for alias in ("实时电价", "rt_price", "y_true", "actual"):
                if alias in df.columns:
                    df["rt_actual"] = df[alias].astype(float)
                    logger.info("Mapped '%s' -> 'rt_actual'", alias)
                    break
        if "rt_actual" not in df.columns:
            raise ValueError(f"Cannot find actual column in {path}.")

        df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
        df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype(int)
        return df

    # Need to convert from timestamp format
    ts_col = None
    for candidate in ("时刻", "timestamp", "ds", "datetime"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"Cannot find timestamp column in {path}. "
                         f"Expected one of: 时刻, timestamp, ds, datetime, business_day")

    price_col = None
    for candidate in ("实时电价", "rt_price", "rt_actual", "y_true", "actual"):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        raise ValueError(f"Cannot find price column in {path}. "
                         f"Expected one of: 实时电价, rt_price, rt_actual, y_true")

    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["_ts"])

    def _convert_ts(ts: pd.Timestamp) -> Tuple[pd.Timestamp, int]:
        if ts.hour == 0:
            return pd.Timestamp(ts.date()) - pd.Timedelta(days=1), 24
        else:
            return pd.Timestamp(ts.date()), ts.hour

    converted = df["_ts"].apply(_convert_ts)
    df["business_day"] = [c[0] for c in converted]
    df["hour_business"] = [c[1] for c in converted]
    df["rt_actual"] = df[price_col].astype(float)
    df = df.drop(columns=["_ts"])

    logger.info("Ground truth converted: %d rows, business_day range [%s, %s]",
                len(df), df["business_day"].min(), df["business_day"].max())
    return df


def load_per_model_predictions(fused_path: str) -> Dict[str, Optional[pd.DataFrame]]:
    """Try to load per-model raw predictions from the fused predictions file.

    Looks for columns like: sgdfnet_pred, timesfm_pred, lgbm_pred, etc.
    Also checks for separate per-model files in the same directory.
    """
    result: Dict[str, Optional[pd.DataFrame]] = {
        "sgdfnet": None,
        "timesfm": None,
    }

    df = pd.read_csv(fused_path, encoding="utf-8-sig")

    # Check for model-specific columns in the fused file
    model_col_map = {
        "sgdfnet": ["sgdfnet_pred", "sgdfnet_y_pred", "pred_sgdfnet"],
        "timesfm": ["timesfm_pred", "timesfm_y_pred", "pred_timesfm"],
    }

    for model_name, candidates in model_col_map.items():
        for col in candidates:
            if col in df.columns:
                model_df = df[["business_day", "hour_business"]].copy()
                model_df["business_day"] = pd.to_datetime(model_df["business_day"], errors="coerce")
                model_df["hour_business"] = pd.to_numeric(model_df["hour_business"], errors="coerce").astype(int)
                model_df["model_pred"] = df[col]
                result[model_name] = model_df
                logger.info("Found per-model predictions for '%s' in column '%s'", model_name, col)
                break

    # Also check for separate per-model files in the same directory
    fused_dir = Path(fused_path).parent
    for model_name in result:
        if result[model_name] is not None:
            continue
        # Look for files like sgdfnet_predictions.csv, timesfm_predictions.csv
        for pattern in [f"{model_name}_predictions.csv", f"{model_name}_pred.csv"]:
            candidate = fused_dir / pattern
            if candidate.is_file():
                try:
                    mdf = pd.read_csv(str(candidate), encoding="utf-8-sig")
                    if "business_day" in mdf.columns and "hour_business" in mdf.columns:
                        mdf["business_day"] = pd.to_datetime(mdf["business_day"], errors="coerce")
                        mdf["hour_business"] = pd.to_numeric(mdf["hour_business"], errors="coerce").astype(int)
                        # Find prediction column
                        pred_col = None
                        for pc in ("rt_pred", "y_pred", "y_fused", "prediction", "model_pred"):
                            if pc in mdf.columns:
                                pred_col = pc
                                break
                        if pred_col:
                            mdf["model_pred"] = mdf[pred_col]
                            result[model_name] = mdf[["business_day", "hour_business", "model_pred"]]
                            logger.info("Loaded per-model predictions for '%s' from %s", model_name, candidate)
                            break
                except Exception as exc:
                    logger.warning("Failed to load %s: %s", candidate, exc)

    return result


def load_and_prepare_pack(pack_path: str) -> pd.DataFrame:
    """Load, normalise, validate, and deduplicate the intraday pack."""
    raw = load_intraday_pack(pack_path)
    pack = normalize_intraday_pack(raw, source_pack_path=pack_path)
    validation = validate_intraday_pack(pack, mode="offline")
    if not validation.valid:
        logger.warning("Intraday pack validation issues: %s", validation.errors)

    # Deduplicate
    if "business_day" in pack.columns and "target_hour" in pack.columns:
        n_before = len(pack)
        dupes = pack.groupby(["business_day", "target_hour"]).size()
        n_dup_groups = int((dupes > 1).sum())
        if n_dup_groups > 0:
            logger.info("Deduplicating pack: %d duplicate groups", n_dup_groups)
            if "cutoff_hour" in pack.columns:
                pack = pack.sort_values("cutoff_hour", ascending=False)
            pack = pack.drop_duplicates(subset=["business_day", "target_hour"], keep="first")
            logger.info("After dedup: %d rows (removed %d)", len(pack), n_before - len(pack))

    logger.info("Intraday pack prepared: %d rows", len(pack))
    return pack


def load_config(config_path: str) -> IntradayTrackerMainlineConfig:
    """Load tracker config from YAML, falling back to defaults."""
    p = Path(config_path)
    if p.is_file():
        logger.info("Loading config from %s", p)
        return IntradayTrackerMainlineConfig.from_yaml(str(p))
    logger.warning("Config file not found at %s, using defaults", p)
    return IntradayTrackerMainlineConfig()


# ---------------------------------------------------------------------------
# Stage evaluation
# ---------------------------------------------------------------------------

def evaluate_stage_a_per_model(
    model_predictions: Dict[str, Optional[pd.DataFrame]],
    gt_df: pd.DataFrame,
) -> Dict[str, dict]:
    """Evaluate Stage A: per-model raw predictions."""
    results: Dict[str, dict] = {}

    for model_name, model_df in model_predictions.items():
        if model_df is None:
            results[model_name] = {"label": f"stage_a_{model_name}", "status": "NOT_AVAILABLE"}
            logger.info("Stage A [%s]: NOT_AVAILABLE", model_name)
            continue

        # Merge with ground truth
        merged = model_df.merge(
            gt_df[["business_day", "hour_business", "rt_actual"]],
            on=["business_day", "hour_business"],
            how="inner",
            suffixes=("", "_gt"),
        )

        if len(merged) == 0:
            results[model_name] = {"label": f"stage_a_{model_name}", "status": "NO_MATCH"}
            logger.info("Stage A [%s]: NO_MATCH (0 merged rows)", model_name)
            continue

        y_true = merged["rt_actual"].values
        y_pred = merged["model_pred"].values
        hours = merged["hour_business"].values

        metrics = compute_stage_metrics(y_true, y_pred, hours, label=f"stage_a_{model_name}")
        metrics["status"] = "OK"
        results[model_name] = metrics

        logger.info("Stage A [%s]: overall=%.4f, 9_16=%.4f, n=%d",
                     model_name, metrics["overall_smape"],
                     metrics.get("segment_9_16_smape", float("nan")),
                     metrics["n_samples"])

    return results


def evaluate_stage_b_fused(
    fused_df: pd.DataFrame,
    gt_df: pd.DataFrame,
) -> dict:
    """Evaluate Stage B: fused baseline (y_fused before intraday)."""
    merged = fused_df.merge(
        gt_df[["business_day", "hour_business", "rt_actual"]],
        on=["business_day", "hour_business"],
        how="inner",
        suffixes=("", "_gt"),
    )

    if len(merged) == 0:
        return {"label": "stage_b_fused", "status": "NO_MATCH"}

    y_true = merged["rt_actual"].values
    y_pred = merged["y_fused"].values
    hours = merged["hour_business"].values

    metrics = compute_stage_metrics(y_true, y_pred, hours, label="stage_b_fused")
    metrics["status"] = "OK"

    logger.info("Stage B [fused]: overall=%.4f, 9_16=%.4f, n=%d",
                metrics["overall_smape"],
                metrics.get("segment_9_16_smape", float("nan")),
                metrics["n_samples"])

    return metrics


def evaluate_stage_c_intraday(
    fused_df: pd.DataFrame,
    pack_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    config: IntradayTrackerMainlineConfig,
    mode: str = "low_weight",
) -> dict:
    """Evaluate Stage C: fused + IntradayTracker correction."""
    try:
        result_df, stats = apply_intraday_tracker_correction(
            base_forecast_df=fused_df,
            intraday_pack_df=pack_df,
            mode=mode,
            config=config,
            prediction_mode="INTRADAY",
        )
    except Exception as exc:
        logger.error("Stage C [%s] failed: %s", mode, exc)
        return {"label": f"stage_c_{mode}", "status": "ERROR", "error": str(exc)}

    # Determine prediction column
    pred_col = None
    for c in ("rt_pred_after_intraday", "y_fused_after_intraday", "rt_pred", "y_fused"):
        if c in result_df.columns:
            pred_col = c
            break
    if pred_col is None:
        return {"label": f"stage_c_{mode}", "status": "NO_PRED_COLUMN"}

    merged = result_df.merge(
        gt_df[["business_day", "hour_business", "rt_actual"]],
        on=["business_day", "hour_business"],
        how="inner",
        suffixes=("", "_gt"),
    )

    if len(merged) == 0:
        return {"label": f"stage_c_{mode}", "status": "NO_MATCH"}

    y_true = merged["rt_actual"].values
    y_pred = merged[pred_col].values
    hours = merged["hour_business"].values

    metrics = compute_stage_metrics(y_true, y_pred, hours, label=f"stage_c_{mode}")
    metrics["status"] = "OK"
    metrics["correction_stats"] = {
        "matched_rows": stats.get("matched_rows", 0),
        "applied_rows": stats.get("applied_rows", 0),
        "avg_fusion_weight": stats.get("avg_fusion_weight", 0.0),
    }

    logger.info("Stage C [%s]: overall=%.4f, 9_16=%.4f, n=%d, applied=%d",
                mode, metrics["overall_smape"],
                metrics.get("segment_9_16_smape", float("nan")),
                metrics["n_samples"],
                stats.get("applied_rows", 0))

    return metrics


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(stages: List[dict]) -> List[dict]:
    """Compute delta vs previous stage for each stage."""
    for i, stage in enumerate(stages):
        if stage.get("status") != "OK":
            stage["delta_overall"] = "NOT_AVAILABLE"
            stage["delta_9_16"] = "NOT_AVAILABLE"
            continue

        if i == 0:
            stage["delta_overall"] = 0.0
            stage["delta_9_16"] = 0.0
            continue

        prev = stages[i - 1]
        if prev.get("status") != "OK":
            stage["delta_overall"] = "NOT_AVAILABLE"
            stage["delta_9_16"] = "NOT_AVAILABLE"
            continue

        # Delta in percentage points (positive = improvement)
        prev_overall = prev.get("overall_smape", float("nan"))
        curr_overall = stage.get("overall_smape", float("nan"))
        if not (np.isnan(prev_overall) or np.isnan(curr_overall)):
            stage["delta_overall"] = (prev_overall - curr_overall) * 100
        else:
            stage["delta_overall"] = "NOT_AVAILABLE"

        prev_916 = prev.get("segment_9_16_smape", float("nan"))
        curr_916 = stage.get("segment_9_16_smape", float("nan"))
        if not (np.isnan(prev_916) or np.isnan(curr_916)):
            stage["delta_9_16"] = (prev_916 - curr_916) * 100
        else:
            stage["delta_9_16"] = "NOT_AVAILABLE"

    return stages


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_module_contribution_summary(stages: List[dict], out_dir: Path) -> None:
    """Write module_contribution_summary.csv."""
    rows = []
    for stage in stages:
        row = {
            "stage": stage.get("label", ""),
            "status": stage.get("status", "UNKNOWN"),
            "n_samples": stage.get("n_samples", 0),
            "overall_smape": stage.get("overall_smape", float("nan")),
            "segment_9_16_smape": stage.get("segment_9_16_smape", float("nan")),
            "segment_9_16_n": stage.get("segment_9_16_n", 0),
            "negative_bucket_smape": stage.get("negative_bucket_smape", float("nan")),
            "negative_bucket_n": stage.get("negative_bucket_n", 0),
            "spike_bucket_smape": stage.get("spike_bucket_smape", float("nan")),
            "spike_bucket_n": stage.get("spike_bucket_n", 0),
            "delta_overall_pp": stage.get("delta_overall", "NOT_AVAILABLE"),
            "delta_9_16_pp": stage.get("delta_9_16", "NOT_AVAILABLE"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    path = out_dir / "module_contribution_summary.csv"
    df.to_csv(str(path), index=False, encoding="utf-8-sig")
    logger.info("Module contribution summary written to %s (%d rows)", path, len(df))


def write_module_contribution_report(stages: List[dict], out_dir: Path) -> None:
    """Write module_contribution_report.md."""

    def _fmt(val, unit: str = "") -> str:
        if val is None or val == "NOT_AVAILABLE" or val == "NO_MATCH" or val == "ERROR":
            return str(val)
        if isinstance(val, float) and np.isnan(val):
            return "N/A"
        if isinstance(val, float):
            return f"{val:.4f}{unit}"
        return str(val)

    def _fmt_delta(val) -> str:
        if val == "NOT_AVAILABLE":
            return "N/A"
        if isinstance(val, (int, float)):
            sign = "+" if val > 0 else ""
            return f"{sign}{val:.4f}pp"
        return str(val)

    lines: List[str] = [
        "# Phase 13 — Module Contribution Analysis",
        "",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Pipeline Stages",
        "",
        "- **Stage A:** Per-model raw predictions (sgdfnet, timesfm)",
        "- **Stage B:** Fused baseline (y_fused before intraday correction)",
        "- **Stage C:** Fused + IntradayTracker correction (shadow / low_weight)",
        "",
        "## Overall sMAPE (floor=50)",
        "",
        "| Stage | Status | Samples | Overall sMAPE | Delta vs Prev | 9_16 sMAPE | Delta 9_16 |",
        "|-------|--------|---------|---------------|---------------|------------|------------|",
    ]

    for stage in stages:
        label = stage.get("label", "")
        status = stage.get("status", "UNKNOWN")
        n = stage.get("n_samples", 0)
        overall = _fmt(stage.get("overall_smape"))
        delta = _fmt_delta(stage.get("delta_overall"))
        seg916 = _fmt(stage.get("segment_9_16_smape"))
        delta916 = _fmt_delta(stage.get("delta_9_16"))

        lines.append(f"| {label} | {status} | {n} | {overall} | {delta} | {seg916} | {delta916} |")

    lines.append("")

    # Bucket analysis
    lines += [
        "## Per-Bucket sMAPE",
        "",
        "| Stage | Negative sMAPE | Neg N | Spike sMAPE | Spike N | Normal sMAPE | Normal N |",
        "|-------|---------------|-------|-------------|---------|--------------|----------|",
    ]

    for stage in stages:
        label = stage.get("label", "")
        if stage.get("status") != "OK":
            lines.append(f"| {label} | N/A | 0 | N/A | 0 | N/A | 0 |")
            continue

        neg_smape = _fmt(stage.get("negative_bucket_smape"))
        neg_n = stage.get("negative_bucket_n", 0)
        spike_smape = _fmt(stage.get("spike_bucket_smape"))
        spike_n = stage.get("spike_bucket_n", 0)
        normal_smape = _fmt(stage.get("normal_bucket_smape"))
        normal_n = stage.get("normal_bucket_n", 0)

        lines.append(
            f"| {label} | {neg_smape} | {neg_n} | {spike_smape} | {spike_n} "
            f"| {normal_smape} | {normal_n} |"
        )

    lines.append("")

    # Interpretation
    lines += [
        "## Interpretation",
        "",
    ]

    # Find the best performing stage
    ok_stages = [s for s in stages if s.get("status") == "OK"]
    if ok_stages:
        best = min(ok_stages, key=lambda s: s.get("overall_smape", float("inf")))
        lines.append(
            f"- **Best overall stage:** {best['label']} "
            f"(sMAPE = {best['overall_smape']:.4f})"
        )

        # Check fusion contribution (A -> B)
        stage_a_stages = [s for s in ok_stages if s["label"].startswith("stage_a_")]
        stage_b = [s for s in ok_stages if s["label"] == "stage_b_fused"]
        if stage_a_stages and stage_b:
            best_a = min(stage_a_stages, key=lambda s: s.get("overall_smape", float("inf")))
            fusion_delta = (best_a["overall_smape"] - stage_b[0]["overall_smape"]) * 100
            lines.append(
                f"- **Fusion contribution** ({best_a['label']} -> fused): "
                f"{fusion_delta:+.4f}pp"
            )

        # Check intraday contribution (B -> C)
        stage_c = [s for s in ok_stages if s["label"].startswith("stage_c_")]
        if stage_b and stage_c:
            intraday_delta = (stage_b[0]["overall_smape"] - stage_c[0]["overall_smape"]) * 100
            lines.append(
                f"- **Intraday contribution** (fused -> {stage_c[0]['label']}): "
                f"{intraday_delta:+.4f}pp"
            )
    else:
        lines.append("- No stages completed successfully. Check data availability.")

    lines += [
        "",
        "## Output Files",
        "",
        "- `module_contribution_summary.csv` — Machine-readable stage metrics",
        "- `module_contribution_report.md` — This report",
        "",
        "---",
        f"*Generated by scripts/evaluate_module_contributions.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path = out_dir / "module_contribution_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Module contribution report written to %s", path)


# ---------------------------------------------------------------------------
# Main evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate_module_contributions(
    fused_predictions_path: str,
    ground_truth_path: str,
    intraday_pack_path: Optional[str],
    output_dir: str,
    config_path: str = DEFAULT_CONFIG,
) -> List[dict]:
    """Run the full module contribution decomposition.

    Returns a list of stage metric dicts.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading fused predictions from %s", fused_predictions_path)
    fused_df = load_fused_predictions(fused_predictions_path)

    logger.info("Loading ground truth from %s", ground_truth_path)
    gt_df = load_ground_truth(ground_truth_path)

    # ---- Stage A: Per-model raw predictions ----
    logger.info("=" * 60)
    logger.info("Stage A: Per-model raw predictions")
    logger.info("=" * 60)
    model_predictions = load_per_model_predictions(fused_predictions_path)
    stage_a_results = evaluate_stage_a_per_model(model_predictions, gt_df)

    # Build stage list for delta computation
    stages: List[dict] = []

    # Add each available model as a stage
    for model_name in ("sgdfnet", "timesfm"):
        if model_name in stage_a_results:
            stages.append(stage_a_results[model_name])

    # ---- Stage B: Fused baseline ----
    logger.info("=" * 60)
    logger.info("Stage B: Fused baseline")
    logger.info("=" * 60)
    stage_b = evaluate_stage_b_fused(fused_df, gt_df)
    stages.append(stage_b)

    # ---- Stage C: Fused + IntradayTracker ----
    if intraday_pack_path and Path(intraday_pack_path).is_file():
        logger.info("=" * 60)
        logger.info("Stage C: Fused + IntradayTracker")
        logger.info("=" * 60)

        logger.info("Loading intraday pack from %s", intraday_pack_path)
        pack_df = load_and_prepare_pack(intraday_pack_path)

        config = load_config(config_path)

        # Shadow mode
        stage_c_shadow = evaluate_stage_c_intraday(
            fused_df, pack_df, gt_df, config, mode="shadow",
        )
        stages.append(stage_c_shadow)

        # Low-weight mode
        stage_c_lw = evaluate_stage_c_intraday(
            fused_df, pack_df, gt_df, config, mode="low_weight",
        )
        stages.append(stage_c_lw)
    else:
        logger.warning("No intraday pack provided or file not found. Stage C will be NOT_AVAILABLE.")
        stages.append({"label": "stage_c_shadow", "status": "NOT_AVAILABLE"})
        stages.append({"label": "stage_c_low_weight", "status": "NOT_AVAILABLE"})

    # ---- Compute deltas ----
    stages = compute_deltas(stages)

    # ---- Write outputs ----
    write_module_contribution_summary(stages, out)
    write_module_contribution_report(stages, out)

    # Write full JSON
    json_path = out / "module_contribution_detail.json"
    # Strip non-serialisable items
    clean_stages = []
    for s in stages:
        cs = {k: v for k, v in s.items() if k != "correction_stats"}
        if "correction_stats" in s:
            cs["correction_stats"] = s["correction_stats"]
        clean_stages.append(cs)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
            "fused_predictions_path": fused_predictions_path,
            "ground_truth_path": ground_truth_path,
            "intraday_pack_path": intraday_pack_path or "",
            "stages": clean_stages,
        }, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Module contribution detail written to %s", json_path)

    # ---- Summary log ----
    logger.info("=" * 60)
    logger.info("Module Contribution Summary:")
    for stage in stages:
        label = stage.get("label", "")
        status = stage.get("status", "")
        overall = stage.get("overall_smape", float("nan"))
        delta = stage.get("delta_overall", "N/A")
        if status == "OK":
            logger.info("  %-25s  sMAPE=%.4f  delta=%s", label, overall, delta)
        else:
            logger.info("  %-25s  %s", label, status)
    logger.info("=" * 60)

    return stages


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 — Evaluate Module Contributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fused-predictions", required=True,
        help="Path to monthly fused predictions CSV.",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth CSV (supports Chinese column names: 时刻, 实时电价).",
    )
    parser.add_argument(
        "--intraday-pack", default=None,
        help="Path to intraday correction pack CSV (aligned pack from Phase 13).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to intraday tracker YAML config (default: {DEFAULT_CONFIG}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("Phase 13 — Evaluate Module Contributions")
    logger.info("=" * 60)
    logger.info("Fused predictions : %s", args.fused_predictions)
    logger.info("Ground truth      : %s", args.ground_truth)
    logger.info("Intraday pack     : %s", args.intraday_pack or "(not provided)")
    logger.info("Output dir        : %s", args.output_dir)
    logger.info("Config            : %s", args.config)

    # Validate required inputs
    for label, path_str in [
        ("fused-predictions", args.fused_predictions),
        ("ground-truth", args.ground_truth),
    ]:
        if not Path(path_str).is_file():
            logger.error("Input file not found: %s (%s)", label, path_str)
            return 1

    if args.intraday_pack and not Path(args.intraday_pack).is_file():
        logger.warning("Intraday pack not found: %s — Stage C will be NOT_AVAILABLE",
                        args.intraday_pack)

    evaluate_module_contributions(
        fused_predictions_path=args.fused_predictions,
        ground_truth_path=args.ground_truth,
        intraday_pack_path=args.intraday_pack,
        output_dir=args.output_dir,
        config_path=args.config,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
