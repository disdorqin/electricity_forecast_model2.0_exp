"""Shadow replay for electricity forecast Phase 12 — intraday tracker.

End-to-end shadow replay that re-simulates the intraday tracker correction
across multiple modes (shadow, low_weight, high_weight) using:
  - A fused predictions CSV (main pipeline fused output)
  - An intraday correction pack CSV (deep branch Phase 10)
  - A ground truth CSV (raw data with actual prices)

Produces per-mode sMAPE (floor=50), per-bucket, per-cutoff, and per-policy
metrics, along with a consolidated markdown report.

Usage::

    python scripts/run_intraday_shadow_replay.py \
        --fused-predictions  path/to/fused_predictions.csv \
        --intraday-pack      path/to/intraday_pack.csv \
        --ground-truth       path/to/ground_truth.csv \
        --mode               shadow \
        --prediction-mode    INTRADAY \
        --config             config/intraday_tracker.yaml \
        --out-dir            reports/local/phase12/intraday_shadow_replay \
        --simulate-modes     shadow,low_weight,high_weight
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

# Attempt to add the main experiment repo if it exists alongside this workspace
_MAIN_REPO_CANDIDATE = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp"
if _MAIN_REPO_CANDIDATE.is_dir() and str(_MAIN_REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_MAIN_REPO_CANDIDATE))

# ---------------------------------------------------------------------------
# Imports from corrections.intraday_tracker
# ---------------------------------------------------------------------------
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
SIMULATE_MODES = ("shadow", "low_weight", "high_weight")
DEFAULT_OUT_DIR = "reports/local/phase12/intraday_shadow_replay"
DEFAULT_CONFIG = "config/intraday_tracker.yaml"

# Bucket thresholds
NEGATIVE_THRESHOLD = 0.0        # actual <= 0
SPIKE_PERCENTILE = 95           # actual >= 95th percentile


# ===================================================================
# Metric helpers
# ===================================================================

def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute sMAPE with a floor of 50 on both predictions and actuals.

    sMAPE = mean( 2 * |y_true - y_pred| / (max(|y_true|, FLOOR) + max(|y_pred|, FLOOR)) )

    The floor prevents division-by-zero and dampens the metric for near-zero values.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    denom = np.maximum(np.abs(y_true), SMAPE_FLOOR) + np.maximum(np.abs(y_pred), SMAPE_FLOOR)
    denom = np.where(denom == 0, 1.0, denom)
    pointwise = 2.0 * np.abs(y_true - y_pred) / denom
    return float(np.mean(pointwise))


def classify_buckets(y_true: np.ndarray) -> Dict[str, np.ndarray]:
    """Classify each sample into negative / spike / normal buckets.

    Returns a dict mapping bucket name to boolean mask.
    """
    spike_threshold = float(np.percentile(y_true, SPIKE_PERCENTILE))
    masks = {
        "negative": y_true <= NEGATIVE_THRESHOLD,
        "spike": (y_true > NEGATIVE_THRESHOLD) & (y_true >= spike_threshold),
        "normal": (y_true > NEGATIVE_THRESHOLD) & (y_true < spike_threshold),
    }
    return masks


# ===================================================================
# Data loading
# ===================================================================

def load_fused_predictions(path: str) -> pd.DataFrame:
    """Load fused predictions CSV and normalise column names.

    Expected columns: business_day, hour_business, ds, y_fused (or y_pred),
    period, target_day.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Fused predictions loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Normalise prediction column — ensure y_fused exists
    if "y_fused" not in df.columns:
        for alias in ("y_pred", "rt_pred", "rt_pred_final"):
            if alias in df.columns:
                df["y_fused"] = df[alias]
                logger.info("Mapped '%s' -> 'y_fused'", alias)
                break

    if "y_fused" not in df.columns:
        raise ValueError(
            f"Cannot find prediction column in {path}. "
            f"Expected one of: y_fused, y_pred, rt_pred"
        )

    # Ensure rt_pred alias for apply_intraday_tracker_correction
    if "rt_pred" not in df.columns:
        df["rt_pred"] = df["y_fused"]

    # Ensure required columns
    for col in ("business_day", "hour_business"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth CSV with Chinese column names and convert timestamps.

    Expected columns: 时刻 (timestamp), 实时电价 (rt_price).

    Timestamp convention:
      - timestamp D 00:00  -> business_day D-1, hour_business 24
      - timestamp D HH:00 (HH>=1) -> business_day D, hour HH
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Ground truth loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Detect timestamp column
    ts_col = None
    for candidate in ("时刻", "timestamp", "ds", "datetime"):
        if candidate in df.columns:
            ts_col = candidate
            break
    if ts_col is None:
        raise ValueError(f"Cannot find timestamp column in {path}. "
                         f"Expected one of: 时刻, timestamp, ds, datetime")

    # Detect actual price column
    price_col = None
    for candidate in ("实时电价", "rt_price", "rt_actual", "y_true", "actual"):
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        raise ValueError(f"Cannot find price column in {path}. "
                         f"Expected one of: 实时电价, rt_price, rt_actual, y_true")

    # Parse timestamps
    df["_ts"] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=["_ts"])

    # Convert timestamp to business_day + hour_business
    # Convention:
    #   D 00:00 -> business_day = D-1, hour_business = 24
    #   D HH:00 (HH>=1) -> business_day = D, hour_business = HH
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


def load_and_prepare_pack(pack_path: str, *, deduplicate: bool = True) -> pd.DataFrame:
    """Load, normalise, validate, and optionally deduplicate the intraday pack.

    When *deduplicate* is True (default) and the pack contains multiple rows for
    the same ``(business_day, target_hour)`` pair — typically because the deep
    branch exports predictions at several cutoff hours — we keep only the row
    with the **highest** cutoff_hour (i.e. the most observed intraday data).
    """
    raw = load_intraday_pack(pack_path)
    pack = normalize_intraday_pack(raw, source_pack_path=pack_path)
    validation = validate_intraday_pack(pack, mode="offline")
    if not validation.valid:
        logger.warning("Intraday pack validation issues: %s", validation.errors)

    if deduplicate and "business_day" in pack.columns and "target_hour" in pack.columns:
        n_before = len(pack)
        dupes = pack.groupby(["business_day", "target_hour"]).size()
        n_dup_groups = int((dupes > 1).sum())

        if n_dup_groups > 0:
            logger.info(
                "Pack has %d duplicate (business_day, target_hour) groups "
                "(%d rows). Deduplicating by keeping highest cutoff_hour.",
                n_dup_groups, n_before,
            )
            if "cutoff_hour" in pack.columns:
                pack = pack.sort_values("cutoff_hour", ascending=False)
            pack = pack.drop_duplicates(
                subset=["business_day", "target_hour"], keep="first",
            )
            logger.info(
                "After deduplication: %d rows (removed %d)",
                len(pack), n_before - len(pack),
            )
        else:
            logger.info("Pack has no duplicate (business_day, target_hour) groups.")

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


# ===================================================================
# Shadow replay core
# ===================================================================

def run_correction(
    base_df: pd.DataFrame,
    pack_df: pd.DataFrame,
    mode: str,
    config: IntradayTrackerMainlineConfig,
    prediction_mode: str = "INTRADAY",
) -> Tuple[pd.DataFrame, dict]:
    """Run intraday correction for a single mode and return (result_df, stats)."""
    result_df, stats = apply_intraday_tracker_correction(
        base_forecast_df=base_df,
        intraday_pack_df=pack_df,
        mode=mode,
        config=config,
        prediction_mode=prediction_mode,
    )
    return result_df, stats


def merge_with_ground_truth(
    result_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pred_col: str = "rt_pred_after_intraday",
) -> pd.DataFrame:
    """Merge corrected predictions with ground truth on (business_day, hour_business)."""
    merged = result_df.merge(
        gt_df[["business_day", "hour_business", "rt_actual"]],
        on=["business_day", "hour_business"],
        how="inner",
        suffixes=("", "_gt"),
    )
    # Use the appropriate prediction column
    if pred_col in merged.columns:
        merged["prediction"] = merged[pred_col]
    elif "y_fused_after_intraday" in merged.columns:
        merged["prediction"] = merged["y_fused_after_intraday"]
    elif "rt_pred" in merged.columns:
        merged["prediction"] = merged["rt_pred"]
    else:
        raise ValueError(f"Cannot find prediction column '{pred_col}' in merged df")
    return merged


def compute_overall_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Compute overall sMAPE_floor50 metric."""
    return {
        "label": label,
        "overall_smape": smape_floor50(y_true, y_pred),
        "n_samples": len(y_true),
    }


def compute_bucket_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> List[dict]:
    """Compute per-bucket (normal/spike/negative) sMAPE metrics."""
    masks = classify_buckets(y_true)
    rows = []
    for bucket_name, mask in masks.items():
        if mask.sum() > 0:
            rows.append({
                "label": label,
                "bucket": bucket_name,
                "smape_floor50": smape_floor50(y_true[mask], y_pred[mask]),
                "n_samples": int(mask.sum()),
            })
        else:
            rows.append({
                "label": label,
                "bucket": bucket_name,
                "smape_floor50": float("nan"),
                "n_samples": 0,
            })
    return rows


def compute_hourly_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hours: np.ndarray,
    label: str = "",
) -> pd.DataFrame:
    """Compute per-hour sMAPE metrics."""
    unique_hours = sorted(set(hours))
    rows = []
    for h in unique_hours:
        mask = hours == h
        if mask.sum() > 0:
            rows.append({
                "label": label,
                "hour": int(h),
                "smape_floor50": smape_floor50(y_true[mask], y_pred[mask]),
                "n_samples": int(mask.sum()),
            })
    return pd.DataFrame(rows)


def compute_cutoff_metrics(
    result_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    pack_df: pd.DataFrame,
    label: str = "",
) -> List[dict]:
    """Compute per-cutoff-hour metrics.

    Groups predictions by the cutoff_hour from the intraday pack and computes
    sMAPE for each cutoff group.
    """
    if "cutoff_hour" not in pack_df.columns:
        return []

    # Build a lookup: (business_day, target_hour) -> cutoff_hour
    cutoff_lookup = {}
    for _, row in pack_df.iterrows():
        key = (pd.Timestamp(row["business_day"]), int(row["target_hour"]))
        cutoff_lookup[key] = row.get("cutoff_hour", None)

    # Merge result with ground truth to get actuals
    merged = result_df.merge(
        gt_df[["business_day", "hour_business", "rt_actual"]],
        on=["business_day", "hour_business"],
        how="inner",
        suffixes=("", "_gt"),
    )

    # Determine prediction column
    pred_col = None
    for c in ("rt_pred_after_intraday", "y_fused_after_intraday", "rt_pred"):
        if c in merged.columns:
            pred_col = c
            break
    if pred_col is None:
        return []

    # Assign cutoff_hour to each merged row
    def _get_cutoff(row):
        key = (row["business_day"], int(row["hour_business"]))
        return cutoff_lookup.get(key, None)

    merged["cutoff_hour"] = merged.apply(_get_cutoff, axis=1)
    merged = merged.dropna(subset=["cutoff_hour"])

    if len(merged) == 0:
        return []

    rows = []
    for cutoff, grp in merged.groupby("cutoff_hour"):
        y_t = grp["rt_actual"].values
        y_p = grp[pred_col].values
        rows.append({
            "label": label,
            "cutoff_hour": int(cutoff),
            "smape_floor50": smape_floor50(y_t, y_p),
            "n_samples": len(y_t),
        })
    return rows


def compute_policy_metrics(
    stats_per_mode: Dict[str, dict],
) -> pd.DataFrame:
    """Build per-policy metrics from per-mode stats dicts."""
    rows = []
    for mode_name, stats in stats_per_mode.items():
        policy_counts = stats.get("policy_counts", {})
        for decision, count in policy_counts.items():
            rows.append({
                "mode": mode_name,
                "policy_decision": decision,
                "count": count,
            })
        guardrail_counts = stats.get("guardrail_counts", {})
        for reason, count in guardrail_counts.items():
            rows.append({
                "mode": mode_name,
                "policy_decision": f"GUARDRAIL:{reason}",
                "count": count,
            })
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["mode", "policy_decision", "count"])


# ===================================================================
# JSON serialisation helper
# ===================================================================

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


# ===================================================================
# Output writers
# ===================================================================

def write_predictions_csv(
    result_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Write replay_predictions.csv — the full prediction table from shadow mode."""
    path = out_dir / "replay_predictions.csv"
    result_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Predictions written to %s (%d rows)", path, len(result_df))


def write_manifest(
    modes: List[str],
    stats_per_mode: Dict[str, dict],
    metrics_per_mode: Dict[str, dict],
    fused_path: str,
    pack_path: str,
    gt_path: Optional[str],
    prediction_mode: str,
    config_path: str,
    out_dir: Path,
    baseline_metrics: Optional[dict] = None,
    gain_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Write replay_manifest.json."""
    manifest = {
        "replay_timestamp": datetime.now().isoformat(timespec="seconds"),
        "fused_predictions_path": fused_path,
        "intraday_pack_path": pack_path,
        "ground_truth_path": gt_path or "",
        "prediction_mode": prediction_mode,
        "config_path": config_path,
        "simulated_modes": modes,
        "smape_floor": SMAPE_FLOOR,
        "baseline": baseline_metrics or {},
        "gain_vs_baseline": gain_metrics or {},
        "per_mode_summary": {},
    }
    for mode_name in modes:
        stats = stats_per_mode.get(mode_name, {})
        metrics = metrics_per_mode.get(mode_name, {})
        manifest["per_mode_summary"][mode_name] = {
            "overall_smape": metrics.get("overall_smape", float("nan")),
            "n_samples": metrics.get("n_samples", 0),
            "matched_rows": stats.get("matched_rows", 0),
            "applied_rows": stats.get("applied_rows", 0),
            "shadow_rows": stats.get("shadow_rows", 0),
            "disabled_rows": stats.get("disabled_rows", 0),
            "avg_fusion_weight": stats.get("avg_fusion_weight", 0.0),
            "avg_confidence": stats.get("avg_confidence", 0.0),
            "fallback_reason": stats.get("fallback_reason"),
            "safe_fallback": stats.get("safe_fallback", True),
        }

    path = out_dir / "replay_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Manifest written to %s", path)


def write_metrics_summary(
    metrics_per_mode: Dict[str, dict],
    bucket_rows: List[dict],
    out_dir: Path,
) -> None:
    """Write replay_metrics_summary.json."""
    summary = {
        "replay_timestamp": datetime.now().isoformat(timespec="seconds"),
        "smape_floor": SMAPE_FLOOR,
        "per_mode_metrics": {},
        "bucket_summary": {},
    }
    for mode_name, metrics in metrics_per_mode.items():
        summary["per_mode_metrics"][mode_name] = {
            "overall_smape": metrics.get("overall_smape", float("nan")),
            "n_samples": metrics.get("n_samples", 0),
        }

    # Aggregate bucket info per mode
    for row in bucket_rows:
        mode = row["label"]
        if mode not in summary["bucket_summary"]:
            summary["bucket_summary"][mode] = {}
        summary["bucket_summary"][mode][row["bucket"]] = {
            "smape_floor50": row["smape_floor50"],
            "n_samples": row["n_samples"],
        }

    path = out_dir / "replay_metrics_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Metrics summary written to %s", path)


def write_hourly_metrics(all_hourly: List[pd.DataFrame], out_dir: Path) -> None:
    """Write replay_hourly_metrics.csv."""
    df = pd.concat(all_hourly, ignore_index=True) if all_hourly else pd.DataFrame()
    path = out_dir / "replay_hourly_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Hourly metrics written to %s (%d rows)", path, len(df))


def write_bucket_metrics(bucket_rows: List[dict], out_dir: Path) -> None:
    """Write replay_bucket_metrics.csv."""
    df = pd.DataFrame(bucket_rows) if bucket_rows else pd.DataFrame(
        columns=["label", "bucket", "smape_floor50", "n_samples"]
    )
    path = out_dir / "replay_bucket_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Bucket metrics written to %s", path)


def write_cutoff_metrics(cutoff_rows: List[dict], out_dir: Path) -> None:
    """Write replay_cutoff_metrics.csv."""
    df = pd.DataFrame(cutoff_rows) if cutoff_rows else pd.DataFrame(
        columns=["label", "cutoff_hour", "smape_floor50", "n_samples"]
    )
    path = out_dir / "replay_cutoff_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Cutoff metrics written to %s (%d rows)", path, len(df))


def write_policy_metrics(policy_df: pd.DataFrame, out_dir: Path) -> None:
    """Write replay_policy_metrics.csv."""
    path = out_dir / "replay_policy_metrics.csv"
    policy_df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Policy metrics written to %s (%d rows)", path, len(policy_df))


def write_shadow_replay_report(
    metrics_per_mode: Dict[str, dict],
    stats_per_mode: Dict[str, dict],
    bucket_rows: List[dict],
    modes: List[str],
    fused_path: str,
    pack_path: str,
    gt_path: Optional[str],
    prediction_mode: str,
    out_dir: Path,
    baseline_metrics: Optional[dict] = None,
    gain_metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Write replay_report.md — consolidated markdown report."""
    lines: List[str] = [
        "# Phase 12 — Intraday Shadow Replay Report",
        "",
        f"**Replay timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- **Fused predictions:** `{fused_path}`",
        f"- **Intraday pack:** `{pack_path}`",
        f"- **Ground truth:** `{gt_path or '(not provided)'}`",
        f"- **Prediction mode:** {prediction_mode}",
        f"- **Simulated modes:** {', '.join(modes)}",
        "",
    ]

    # Baseline metrics
    if baseline_metrics is not None:
        lines += [
            "## Baseline (No Correction)",
            "",
            f"- **sMAPE (floor=50):** {baseline_metrics['overall_smape']:.6f}",
            f"- **Samples:** {baseline_metrics['n_samples']}",
            "",
        ]

    # Overall metrics table
    lines += [
        "## Overall sMAPE (floor=50)",
        "",
        "| Mode | sMAPE | Samples | Gain vs Baseline (pp) |",
        "|------|-------|---------|-----------------------|",
    ]
    for mode_name in modes:
        m = metrics_per_mode.get(mode_name, {})
        smape = m.get("overall_smape", float("nan"))
        n = m.get("n_samples", 0)
        gain_key = f"{mode_name}_gain_pp"
        gain = gain_metrics.get(gain_key, float("nan")) if gain_metrics else float("nan")
        gain_str = f"{gain:+.4f}" if not np.isnan(gain) else "N/A"
        lines.append(f"| {mode_name} | {smape:.6f} | {n} | {gain_str} |")
    lines.append("")

    # Bucket metrics
    lines += [
        "## Per-Bucket sMAPE",
        "",
        "Buckets: negative (actual <= 0), spike (actual >= 95th percentile), normal (rest).",
        "",
        "| Mode | Bucket | sMAPE | Samples |",
        "|------|--------|-------|---------|",
    ]
    for row in bucket_rows:
        smape_val = row["smape_floor50"]
        smape_str = f"{smape_val:.6f}" if not (isinstance(smape_val, float) and np.isnan(smape_val)) else "N/A"
        lines.append(f"| {row['label']} | {row['bucket']} | {smape_str} | {row['n_samples']} |")
    lines.append("")

    # Per-mode operational stats
    lines += [
        "## Per-Mode Operational Statistics",
        "",
    ]
    for mode_name in modes:
        stats = stats_per_mode.get(mode_name, {})
        lines += [
            f"### {mode_name}",
            "",
            f"- Pack rows: {stats.get('pack_rows', 0)}",
            f"- Matched rows: {stats.get('matched_rows', 0)}",
            f"- Applied rows: {stats.get('applied_rows', 0)}",
            f"- Shadow rows: {stats.get('shadow_rows', 0)}",
            f"- Disabled rows: {stats.get('disabled_rows', 0)}",
            f"- Avg fusion weight: {stats.get('avg_fusion_weight', 0.0):.4f}",
            f"- Avg confidence: {stats.get('avg_confidence', 0.0):.4f}",
            f"- Fallback reason: {stats.get('fallback_reason', 'None')}",
            f"- Safe fallback: {stats.get('safe_fallback', True)}",
            "",
        ]

    # Policy decisions
    lines += [
        "## Policy Decision Distribution",
        "",
    ]
    has_policy = False
    for mode_name in modes:
        stats = stats_per_mode.get(mode_name, {})
        pc = stats.get("policy_counts", {})
        if pc:
            has_policy = True
            lines.append(f"**{mode_name}:**")
            lines.append("")
            lines.append("| Decision | Count |")
            lines.append("|----------|-------|")
            for decision, count in sorted(pc.items()):
                lines.append(f"| {decision} | {count} |")
            lines.append("")
    if not has_policy:
        lines.append("No policy decisions recorded.")
        lines.append("")

    # Guardrail summary
    has_guardrails = False
    for mode_name in modes:
        stats = stats_per_mode.get(mode_name, {})
        gc = stats.get("guardrail_counts", {})
        if gc:
            has_guardrails = True
    if has_guardrails:
        lines += [
            "## Guardrail Summary",
            "",
        ]
        for mode_name in modes:
            stats = stats_per_mode.get(mode_name, {})
            gc = stats.get("guardrail_counts", {})
            if gc:
                lines.append(f"**{mode_name}:**")
                lines.append("")
                lines.append("| Reason | Count |")
                lines.append("|--------|-------|")
                for reason, count in sorted(gc.items()):
                    lines.append(f"| {reason} | {count} |")
                lines.append("")

    # Configuration
    lines += [
        "## Configuration",
        "",
        f"- sMAPE floor: {SMAPE_FLOOR}",
        f"- Spike percentile: {SPIKE_PERCENTILE}",
        f"- Negative threshold: {NEGATIVE_THRESHOLD}",
        "",
        "## Output Files",
        "",
        "- `replay_predictions.csv` — Full prediction table (shadow mode)",
        "- `replay_manifest.json` — Run manifest with per-mode summary",
        "- `replay_metrics_summary.json` — Metrics summary JSON",
        "- `replay_hourly_metrics.csv` — Per-hour sMAPE per mode",
        "- `replay_bucket_metrics.csv` — Per-bucket sMAPE per mode",
        "- `replay_cutoff_metrics.csv` — Per-cutoff-hour sMAPE per mode",
        "- `replay_policy_metrics.csv` — Policy decision distribution per mode",
        "- `replay_report.md` — This report",
        "",
        "---",
        f"*Generated by scripts/run_intraday_shadow_replay.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path = out_dir / "replay_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Shadow replay report written to %s", path)


# ===================================================================
# Main shadow replay pipeline
# ===================================================================

def shadow_replay(
    fused_predictions_path: str,
    intraday_pack_path: str,
    ground_truth_path: Optional[str],
    modes: List[str],
    prediction_mode: str,
    out_dir: str,
    config_path: str,
    *,
    deduplicate: bool = True,
) -> Dict[str, dict]:
    """Run the full shadow replay pipeline.

    Returns a dict mapping mode name to its metrics dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading fused predictions from %s", fused_predictions_path)
    base_df = load_fused_predictions(fused_predictions_path)

    logger.info("Loading intraday pack from %s", intraday_pack_path)
    pack_df = load_and_prepare_pack(intraday_pack_path, deduplicate=deduplicate)

    gt_df = None
    if ground_truth_path:
        logger.info("Loading ground truth from %s", ground_truth_path)
        gt_df = load_ground_truth(ground_truth_path)

    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    # ---- Compute baseline (no correction) metrics ----
    baseline_metrics: Optional[dict] = None
    if gt_df is not None:
        logger.info("Computing baseline (no correction) metrics...")
        baseline_result, baseline_stats = run_correction(
            base_df, pack_df, "off", config, prediction_mode,
        )
        baseline_merged = merge_with_ground_truth(
            baseline_result, gt_df, pred_col="rt_pred_before_intraday",
        )
        y_true_b = baseline_merged["rt_actual"].values
        y_pred_b = baseline_merged["prediction"].values
        baseline_metrics = compute_overall_metrics(y_true_b, y_pred_b, label="baseline")
        logger.info(
            "  baseline sMAPE: %.6f (%d samples)",
            baseline_metrics["overall_smape"], baseline_metrics["n_samples"],
        )

    # ---- Run each simulated mode ----
    metrics_per_mode: Dict[str, dict] = {}
    stats_per_mode: Dict[str, dict] = {}
    all_hourly: List[pd.DataFrame] = []
    all_bucket_rows: List[dict] = []
    all_cutoff_rows: List[dict] = []
    shadow_result_df: Optional[pd.DataFrame] = None

    for mode in modes:
        logger.info("Running shadow replay in '%s' mode...", mode)
        result_df, stats = run_correction(base_df, pack_df, mode, config, prediction_mode)
        stats_per_mode[mode] = stats

        # Record shadow-mode predictions for output
        if mode == modes[0] or mode == "shadow":
            shadow_result_df = result_df

        # Compute metrics only if ground truth is available
        if gt_df is not None:
            pred_col = "rt_pred_after_intraday"
            merged = merge_with_ground_truth(result_df, gt_df, pred_col=pred_col)
            y_true = merged["rt_actual"].values
            y_pred = merged["prediction"].values

            # Overall metrics
            mode_metrics = compute_overall_metrics(y_true, y_pred, label=mode)
            metrics_per_mode[mode] = mode_metrics
            logger.info("  %s overall sMAPE: %.6f (%d samples)",
                        mode, mode_metrics["overall_smape"], mode_metrics["n_samples"])

            # Per-bucket metrics
            bucket_rows = compute_bucket_metrics(y_true, y_pred, label=mode)
            all_bucket_rows.extend(bucket_rows)

            # Per-hour metrics
            hourly_df = compute_hourly_metrics(
                y_true, y_pred, merged["hour_business"].values, label=mode,
            )
            all_hourly.append(hourly_df)

            # Per-cutoff metrics
            cutoff_rows = compute_cutoff_metrics(result_df, gt_df, pack_df, label=mode)
            all_cutoff_rows.extend(cutoff_rows)
        else:
            metrics_per_mode[mode] = {
                "overall_smape": float("nan"),
                "n_samples": 0,
            }
            logger.info("  %s — no ground truth, metrics skipped", mode)

    # ---- Compute gain metrics ----
    gain_metrics: Dict[str, float] = {}
    if baseline_metrics is not None:
        baseline_smape = baseline_metrics["overall_smape"]
        for mode_name, m in metrics_per_mode.items():
            mode_smape = m.get("overall_smape", float("nan"))
            if not np.isnan(mode_smape):
                gain_metrics[f"{mode_name}_gain_pp"] = (baseline_smape - mode_smape) * 100
        logger.info("Gain vs baseline (percentage points): %s", gain_metrics)

    # ---- Per-policy metrics ----
    policy_df = compute_policy_metrics(stats_per_mode)

    # ---- Write all outputs ----
    # Predictions CSV (use shadow mode result, or first mode)
    if shadow_result_df is not None:
        write_predictions_csv(shadow_result_df, out)

    # Manifest
    write_manifest(
        modes=modes,
        stats_per_mode=stats_per_mode,
        metrics_per_mode=metrics_per_mode,
        fused_path=fused_predictions_path,
        pack_path=intraday_pack_path,
        gt_path=ground_truth_path,
        prediction_mode=prediction_mode,
        config_path=config_path,
        out_dir=out,
        baseline_metrics=baseline_metrics,
        gain_metrics=gain_metrics,
    )

    # Metrics summary
    write_metrics_summary(metrics_per_mode, all_bucket_rows, out)

    # Hourly metrics
    write_hourly_metrics(all_hourly, out)

    # Bucket metrics
    write_bucket_metrics(all_bucket_rows, out)

    # Cutoff metrics
    write_cutoff_metrics(all_cutoff_rows, out)

    # Policy metrics
    write_policy_metrics(policy_df, out)

    # Markdown report
    write_shadow_replay_report(
        metrics_per_mode=metrics_per_mode,
        stats_per_mode=stats_per_mode,
        bucket_rows=all_bucket_rows,
        modes=modes,
        fused_path=fused_predictions_path,
        pack_path=intraday_pack_path,
        gt_path=ground_truth_path,
        prediction_mode=prediction_mode,
        out_dir=out,
        baseline_metrics=baseline_metrics,
        gain_metrics=gain_metrics,
    )

    logger.info("Shadow replay complete. All outputs written to %s", out)
    return metrics_per_mode


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 12 — Intraday Shadow Replay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fused-predictions", required=True,
        help="Path to fused predictions CSV (main pipeline fused output)",
    )
    parser.add_argument(
        "--intraday-pack", required=True,
        help="Path to intraday correction pack CSV (Phase 10 handoff)",
    )
    parser.add_argument(
        "--ground-truth", default=None,
        help="Path to ground truth CSV (raw data with actual prices). "
             "If missing, only shadow operational report is produced.",
    )
    parser.add_argument(
        "--mode", default="shadow", choices=["shadow", "low_weight", "high_weight"],
        help="Primary simulation mode (default: shadow).",
    )
    parser.add_argument(
        "--prediction-mode", default="INTRADAY", choices=["FULL_DAY", "INTRADAY"],
        help="Prediction mode (default: INTRADAY).",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to intraday tracker YAML config (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--simulate-modes", default=None,
        help="Comma-separated list of modes to simulate "
             "(default: shadow,low_weight,high_weight).",
    )
    parser.add_argument(
        "--no-deduplicate", action="store_true", default=False,
        help="Disable automatic pack deduplication (keep all cutoff rows).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)

    # Determine modes to simulate
    if args.simulate_modes:
        modes = [m.strip() for m in args.simulate_modes.split(",") if m.strip()]
    else:
        modes = list(SIMULATE_MODES)

    logger.info("=" * 72)
    logger.info("Phase 12 — Intraday Shadow Replay")
    logger.info("=" * 72)
    logger.info("Fused predictions : %s", args.fused_predictions)
    logger.info("Intraday pack     : %s", args.intraday_pack)
    logger.info("Ground truth      : %s", args.ground_truth or "(not provided)")
    logger.info("Primary mode      : %s", args.mode)
    logger.info("Prediction mode   : %s", args.prediction_mode)
    logger.info("Simulate modes    : %s", modes)
    logger.info("Output dir        : %s", args.out_dir)
    logger.info("Config            : %s", args.config)

    # Validate required inputs exist
    for label, path_str in [
        ("fused-predictions", args.fused_predictions),
        ("intraday-pack", args.intraday_pack),
    ]:
        if not Path(path_str).is_file():
            logger.error("Input file not found: %s (%s)", label, path_str)
            return 1

    if args.ground_truth and not Path(args.ground_truth).is_file():
        logger.error("Input file not found: ground-truth (%s)", args.ground_truth)
        return 1

    metrics_per_mode = shadow_replay(
        fused_predictions_path=args.fused_predictions,
        intraday_pack_path=args.intraday_pack,
        ground_truth_path=args.ground_truth,
        modes=modes,
        prediction_mode=args.prediction_mode,
        out_dir=args.out_dir,
        config_path=args.config,
        deduplicate=not args.no_deduplicate,
    )

    # Print summary
    logger.info("=" * 72)
    logger.info("Shadow Replay Summary")
    logger.info("=" * 72)
    for mode_name, metrics in metrics_per_mode.items():
        smape = metrics.get("overall_smape", float("nan"))
        n = metrics.get("n_samples", 0)
        logger.info("  %-12s  sMAPE=%.6f  n=%d", mode_name, smape, n)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
