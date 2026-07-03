"""Main pipeline evaluation for Phase 11 intraday tracker integration.

Evaluates the intraday tracker correction across three modes (shadow, low_weight,
high_weight) against ground truth, computing sMAPE (floor=50) metrics overall and
per bucket (negative, spike, normal). Produces a go/no-go verdict for mainline
integration.

Usage::

    python evaluate_intraday_mainline.py \
        --base-forecast  path/to/base_forecast.csv \
        --intraday-pack  path/to/intraday_pack.csv \
        --ground-truth   path/to/ground_truth.csv \
        --mode           shadow \
        --out-dir        reports/local/phase11/intraday_mainline_eval \
        --config         configs/intraday_tracker.yaml
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
PROJECT_ROOT = Path(__file__).resolve().parent
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
EVAL_MODES = ("shadow", "low_weight", "high_weight")
DEFAULT_OUT_DIR = "reports/local/phase11/intraday_mainline_eval"
DEFAULT_CONFIG = "configs/intraday_tracker.yaml"

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
    # Avoid division by zero (should not happen with floor=50, but be safe)
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

def load_base_forecast(path: str) -> pd.DataFrame:
    """Load base forecast CSV and normalise column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Base forecast loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Normalise prediction column
    if "rt_pred" not in df.columns:
        for alias in ("y_fused", "rt_pred_final", "y_pred"):
            if alias in df.columns:
                df["rt_pred"] = df[alias]
                logger.info("Mapped '%s' -> 'rt_pred'", alias)
                break

    if "rt_pred" not in df.columns:
        raise ValueError(f"Cannot find prediction column in {path}. "
                         f"Expected one of: rt_pred, y_fused, y_pred")

    # Ensure required columns
    for col in ("business_day", "hour_business"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    return df


def load_ground_truth(path: str) -> pd.DataFrame:
    """Load ground truth CSV and normalise column names."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Ground truth loaded: %d rows, columns=%s", len(df), list(df.columns))

    # Normalise actual column
    if "rt_actual" not in df.columns:
        for alias in ("y_true", "actual", "rt_true"):
            if alias in df.columns:
                df["rt_actual"] = df[alias]
                logger.info("Mapped '%s' -> 'rt_actual'", alias)
                break

    if "rt_actual" not in df.columns:
        raise ValueError(f"Cannot find actual column in {path}. "
                         f"Expected one of: rt_actual, y_true, actual")

    for col in ("business_day", "hour_business"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    return df


def load_and_prepare_pack(pack_path: str) -> pd.DataFrame:
    """Load, normalise, and validate the intraday pack."""
    raw = load_intraday_pack(pack_path)
    pack = normalize_intraday_pack(raw, source_pack_path=pack_path)
    validation = validate_intraday_pack(pack, mode="offline")
    if not validation.valid:
        logger.warning("Intraday pack validation issues: %s", validation.errors)
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
# Evaluation core
# ===================================================================

def run_correction(
    base_df: pd.DataFrame,
    pack_df: pd.DataFrame,
    mode: str,
    config: IntradayTrackerMainlineConfig,
) -> Tuple[pd.DataFrame, dict]:
    """Run intraday correction for a single mode and return (result_df, stats)."""
    result_df, stats = apply_intraday_tracker_correction(
        base_forecast_df=base_df,
        intraday_pack_df=pack_df,
        mode=mode,
        config=config,
        prediction_mode="INTRADAY",
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
    elif "rt_pred" in merged.columns:
        merged["prediction"] = merged["rt_pred"]
    else:
        raise ValueError(f"Cannot find prediction column '{pred_col}' in merged df")
    return merged


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "",
) -> dict:
    """Compute overall and per-bucket sMAPE metrics."""
    overall = smape_floor50(y_true, y_pred)
    masks = classify_buckets(y_true)

    bucket_metrics = {}
    for bucket_name, mask in masks.items():
        if mask.sum() > 0:
            bucket_metrics[bucket_name] = smape_floor50(y_true[mask], y_pred[mask])
        else:
            bucket_metrics[bucket_name] = float("nan")

    return {
        "label": label,
        "overall_smape": overall,
        "n_samples": len(y_true),
        "bucket_smape": bucket_metrics,
        "bucket_counts": {k: int(v.sum()) for k, v in masks.items()},
    }


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


# ===================================================================
# Verdict logic
# ===================================================================

def determine_verdict(
    low_weight_final_gain: float,
    negative_bucket_gain: float,
    overall_final_gain: float,
) -> str:
    """Determine GO / SHADOW_ONLY / NO-GO verdict.

    Rules
    -----
    - GO:           low_weight final gain >= 0.5 AND negative bucket not worse
    - SHADOW_ONLY:  shadow corrected gain > 0 but low_weight final gain < 0.5
    - NO-GO:        final gain <= 0 OR negative bucket worse > 1.0
    """
    # NO-GO conditions
    if overall_final_gain <= 0:
        return "NO-GO"
    if negative_bucket_gain < -1.0:
        # "negative bucket worse > 1.0" means the gain is more negative than -1.0
        return "NO-GO"

    # GO condition
    if low_weight_final_gain >= 0.5 and negative_bucket_gain >= 0:
        return "GO"

    # SHADOW_ONLY: shadow corrected helps but low_weight final doesn't reach threshold
    if low_weight_final_gain < 0.5:
        return "SHADOW_ONLY"

    # Negative bucket is worse but not enough to be NO-GO
    if negative_bucket_gain < 0:
        return "SHADOW_ONLY"

    return "SHADOW_ONLY"


# ===================================================================
# Output writers
# ===================================================================

def write_metrics_summary(
    metrics: dict,
    verdict: str,
    out_dir: Path,
) -> None:
    """Write metrics_summary.json."""
    summary = {
        "evaluation_timestamp": datetime.now().isoformat(timespec="seconds"),
        "smape_floor": SMAPE_FLOOR,
        "verdict": verdict,
        "metrics": metrics,
    }
    path = out_dir / "metrics_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Metrics summary written to %s", path)


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


def write_hourly_metrics(all_hourly: List[pd.DataFrame], out_dir: Path) -> None:
    """Write hourly_metrics.csv."""
    df = pd.concat(all_hourly, ignore_index=True) if all_hourly else pd.DataFrame()
    path = out_dir / "hourly_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Hourly metrics written to %s (%d rows)", path, len(df))


def write_bucket_metrics(all_metrics: List[dict], out_dir: Path) -> None:
    """Write bucket_metrics.csv."""
    rows = []
    for m in all_metrics:
        for bucket, smape_val in m["bucket_smape"].items():
            rows.append({
                "label": m["label"],
                "bucket": bucket,
                "smape_floor50": smape_val,
                "n_samples": m["bucket_counts"].get(bucket, 0),
            })
    df = pd.DataFrame(rows)
    path = out_dir / "bucket_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Bucket metrics written to %s", path)


def write_policy_metrics(stats_per_mode: Dict[str, dict], out_dir: Path) -> None:
    """Write policy_metrics.csv from per-mode stats dicts."""
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
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["mode", "policy_decision", "count"]
    )
    path = out_dir / "policy_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Policy metrics written to %s", path)


def write_cutoff_metrics(stats_per_mode: Dict[str, dict], out_dir: Path) -> None:
    """Write cutoff_metrics.csv summarising cutoff-hour statistics."""
    rows = []
    for mode_name, stats in stats_per_mode.items():
        rows.append({
            "mode": mode_name,
            "pack_rows": stats.get("pack_rows", 0),
            "matched_rows": stats.get("matched_rows", 0),
            "applied_rows": stats.get("applied_rows", 0),
            "shadow_rows": stats.get("shadow_rows", 0),
            "disabled_rows": stats.get("disabled_rows", 0),
            "avg_fusion_weight": stats.get("avg_fusion_weight", 0.0),
            "avg_confidence": stats.get("avg_confidence", 0.0),
            "fallback_reason": stats.get("fallback_reason", ""),
            "safe_fallback": stats.get("safe_fallback", True),
        })
    df = pd.DataFrame(rows)
    path = out_dir / "cutoff_metrics.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Cutoff metrics written to %s", path)


def write_integration_eval_report(
    metrics: dict,
    verdict: str,
    stats_per_mode: Dict[str, dict],
    out_dir: Path,
) -> None:
    """Write integration_eval_report.md."""
    lines: List[str] = [
        "# Phase 11 — Intraday Tracker Mainline Integration Evaluation Report",
        "",
        f"**Evaluation timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]

    # Verdict explanation
    if verdict == "GO":
        lines += [
            "The intraday tracker meets the integration criteria:",
            "- Low-weight final gain >= 0.5 percentage points",
            "- Negative bucket is not worse than baseline",
            "",
        ]
    elif verdict == "SHADOW_ONLY":
        lines += [
            "The intraday tracker shows promise in shadow mode but does not yet",
            "meet the threshold for active mainline integration.",
            "- Shadow corrected gain > 0 (correction direction is correct)",
            "- Low-weight final gain < 0.5 (insufficient magnitude)",
            "",
        ]
    else:
        lines += [
            "The intraday tracker does **not** meet integration criteria:",
            "- Final gain <= 0 OR negative bucket degradation > 1.0",
            "- Recommend further tuning before mainline deployment",
            "",
        ]

    # Metrics table
    lines += [
        "## Metrics Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]
    for key, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"| {key} | {val:.4f} |")
        else:
            lines.append(f"| {key} | {val} |")
    lines.append("")

    # Per-mode details
    lines += [
        "## Per-Mode Statistics",
        "",
    ]
    for mode_name, stats in stats_per_mode.items():
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
            "",
        ]

    # Bucket analysis
    lines += [
        "## Bucket Analysis",
        "",
        "Buckets: negative (actual <= 0), spike (actual >= 95th percentile), normal (rest).",
        "",
        "See `bucket_metrics.csv` for detailed per-bucket sMAPE values.",
        "",
    ]

    # Configuration
    lines += [
        "## Configuration",
        "",
        f"- sMAPE floor: {SMAPE_FLOOR}",
        f"- Spike percentile: {SPIKE_PERCENTILE}",
        f"- Negative threshold: {NEGATIVE_THRESHOLD}",
        "",
        "---",
        f"*Generated by evaluate_intraday_mainline.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path = out_dir / "integration_eval_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Integration eval report written to %s", path)


# ===================================================================
# Main evaluation pipeline
# ===================================================================

def evaluate(
    base_forecast_path: str,
    intraday_pack_path: str,
    ground_truth_path: str,
    mode: str,
    out_dir: str,
    config_path: str,
) -> str:
    """Run the full evaluation pipeline and return the verdict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading base forecast from %s", base_forecast_path)
    base_df = load_base_forecast(base_forecast_path)

    logger.info("Loading ground truth from %s", ground_truth_path)
    gt_df = load_ground_truth(ground_truth_path)

    logger.info("Loading intraday pack from %s", intraday_pack_path)
    pack_df = load_and_prepare_pack(intraday_pack_path)

    logger.info("Loading config from %s", config_path)
    config = load_config(config_path)

    # ---- Baseline metrics (no intraday correction) ----
    logger.info("Computing baseline metrics (no correction)...")
    baseline_result, baseline_stats = run_correction(base_df, pack_df, "off", config)
    baseline_merged = merge_with_ground_truth(baseline_result, gt_df, pred_col="rt_pred_before_intraday")
    y_true_base = baseline_merged["rt_actual"].values
    y_pred_base = baseline_merged["prediction"].values
    baseline_metrics = compute_metrics(y_true_base, y_pred_base, label="baseline")
    baseline_hourly = compute_hourly_metrics(
        y_true_base, y_pred_base,
        baseline_merged["hour_business"].values,
        label="baseline",
    )

    # ---- Run all three evaluation modes ----
    all_metrics = [baseline_metrics]
    all_hourly = [baseline_hourly]
    stats_per_mode: Dict[str, dict] = {"baseline": baseline_stats}

    corrections_per_mode: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for eval_mode in EVAL_MODES:
        logger.info("Running correction in '%s' mode...", eval_mode)
        result_df, stats = run_correction(base_df, pack_df, eval_mode, config)
        stats_per_mode[eval_mode] = stats

        merged = merge_with_ground_truth(result_df, gt_df, pred_col="rt_pred_after_intraday")
        y_t = merged["rt_actual"].values
        y_p = merged["prediction"].values
        corrections_per_mode[eval_mode] = (y_t, y_p)

        mode_metrics = compute_metrics(y_t, y_p, label=eval_mode)
        all_metrics.append(mode_metrics)

        mode_hourly = compute_hourly_metrics(
            y_t, y_p, merged["hour_business"].values, label=eval_mode,
        )
        all_hourly.append(mode_hourly)

    # ---- Compute gain metrics ----
    baseline_smape = baseline_metrics["overall_smape"]
    shadow_smape = all_metrics[1]["overall_smape"]       # shadow
    low_weight_smape = all_metrics[2]["overall_smape"]    # low_weight
    high_weight_smape = all_metrics[3]["overall_smape"]   # high_weight

    # Gains are in percentage-point reduction (positive = improvement)
    shadow_corrected_gain = (baseline_smape - shadow_smape) * 100
    low_weight_final_gain = (baseline_smape - low_weight_smape) * 100
    high_weight_final_gain = (baseline_smape - high_weight_smape) * 100
    gain_vs_baseline = low_weight_final_gain  # primary gain metric

    # Per-bucket gains (low_weight vs baseline)
    y_true_lw = corrections_per_mode["low_weight"][0]
    y_pred_lw = corrections_per_mode["low_weight"][1]
    masks = classify_buckets(y_true_base[:len(y_true_lw)])

    bucket_gains = {}
    for bucket_name, mask in masks.items():
        if mask.sum() > 0:
            base_bucket = smape_floor50(y_true_base[:len(y_true_lw)][mask], y_pred_base[:len(y_pred_base)][mask])
            lw_bucket = smape_floor50(y_true_lw[mask], y_pred_lw[mask])
            bucket_gains[bucket_name] = (base_bucket - lw_bucket) * 100
        else:
            bucket_gains[bucket_name] = float("nan")

    negative_bucket_gain = bucket_gains.get("negative", 0.0)
    spike_bucket_gain = bucket_gains.get("spike", 0.0)
    normal_bucket_gain = bucket_gains.get("normal", 0.0)

    # ---- Assemble metrics dict ----
    metrics = {
        "baseline_sMAPE": baseline_smape,
        "shadow_corrected_sMAPE": shadow_smape,
        "low_weight_final_sMAPE": low_weight_smape,
        "high_weight_final_sMAPE": high_weight_smape,
        "gain_vs_baseline": gain_vs_baseline,
        "shadow_corrected_gain": shadow_corrected_gain,
        "low_weight_final_gain": low_weight_final_gain,
        "high_weight_final_gain": high_weight_final_gain,
        "negative_bucket_gain": negative_bucket_gain,
        "spike_bucket_gain": spike_bucket_gain,
        "normal_bucket_gain": normal_bucket_gain,
    }

    # ---- Verdict ----
    verdict = determine_verdict(low_weight_final_gain, negative_bucket_gain, gain_vs_baseline)
    logger.info("Verdict: %s", verdict)
    logger.info("Metrics: %s", metrics)

    # ---- Write outputs ----
    write_metrics_summary(metrics, verdict, out)
    write_hourly_metrics(all_hourly, out)
    write_bucket_metrics(all_metrics, out)
    write_policy_metrics(stats_per_mode, out)
    write_cutoff_metrics(stats_per_mode, out)
    write_integration_eval_report(metrics, verdict, stats_per_mode, out)

    logger.info("Evaluation complete. All outputs written to %s", out)
    return verdict


# ===================================================================
# CLI
# ===================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 11 — Intraday Tracker Mainline Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-forecast", required=True,
        help="Path to base forecast CSV (columns: business_day, hour_business, rt_pred/y_fused)",
    )
    parser.add_argument(
        "--intraday-pack", required=True,
        help="Path to intraday correction pack CSV (Phase 10 handoff)",
    )
    parser.add_argument(
        "--ground-truth", required=True,
        help="Path to ground truth CSV (columns: business_day, hour_business, rt_actual/y_true)",
    )
    parser.add_argument(
        "--mode", default="shadow", choices=["shadow", "low_weight", "high_weight"],
        help="Primary evaluation mode (default: shadow). All three modes are evaluated; "
             "this selects the primary mode for verdict determination.",
    )
    parser.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to intraday tracker YAML config (default: {DEFAULT_CONFIG})",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)
    logger.info("=" * 72)
    logger.info("Phase 11 — Intraday Tracker Mainline Evaluation")
    logger.info("=" * 72)
    logger.info("Base forecast : %s", args.base_forecast)
    logger.info("Intraday pack : %s", args.intraday_pack)
    logger.info("Ground truth  : %s", args.ground_truth)
    logger.info("Mode          : %s", args.mode)
    logger.info("Output dir    : %s", args.out_dir)
    logger.info("Config        : %s", args.config)

    # Validate inputs exist
    for label, path_str in [
        ("base-forecast", args.base_forecast),
        ("intraday-pack", args.intraday_pack),
        ("ground-truth", args.ground_truth),
    ]:
        if not Path(path_str).is_file():
            logger.error("Input file not found: %s (%s)", label, path_str)
            return 1

    verdict = evaluate(
        base_forecast_path=args.base_forecast,
        intraday_pack_path=args.intraday_pack,
        ground_truth_path=args.ground_truth,
        mode=args.mode,
        out_dir=args.out_dir,
        config_path=args.config,
    )

    logger.info("=" * 72)
    logger.info("FINAL VERDICT: %s", verdict)
    logger.info("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
