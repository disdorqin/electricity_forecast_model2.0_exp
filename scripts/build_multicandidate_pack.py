#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_multicandidate_pack.py — Multi-candidate prediction pack for P0.

Merges LightGBM (Level 1) + baseline models (naive_lag1, naive_lag7, dayahead_proxy)
into a single prediction pack. Supports multiple fusion modes for base_fused_pred.

Fusion modes:
  - mean:                        Simple average of all available models
  - lightgbm_anchor_90:          0.9 * LightGBM + 0.1 * mean(other)
  - lightgbm_anchor_80:          0.8 * LightGBM + 0.2 * mean(other)
  - candidate_reference_only:    No fusion; pack provides candidates only
  - custom:                      Explicit per-model weights via --weights

Timestamp-level dedup:
  All manifest metrics (sMAPE, severe_underestimate) are computed on deduplicated
  timestamps (1 row per business_day + hour_business) to avoid row-level inflation
  from multiple model rows per timestamp.

Usage:
    python scripts/build_multicandidate_pack.py \\
        --level0-pack <path> --level1-pack <path> \\
        --risk-predictions <path> --out-dir <path> \\
        --fusion-mode mean --weights lightgbm=0.85,dayahead_proxy=0.08,...

Output:
    <out-dir>/
      - prediction_pack_realtime_multicandidate_{start}_{end}.csv
      - risk_predictions_multicandidate.csv
      - build_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

BASELINE_MODELS = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
ALL_MODELS = BASELINE_MODELS + ["lightgbm"]

FUSION_MODES = [
    "mean",
    "lightgbm_anchor_90",
    "lightgbm_anchor_80",
    "candidate_reference_only",
    "custom",
]


# ── Helpers ────────────────────────────────────────────────────────────

def compute_smape_floor50(
    y_true: pd.Series, y_pred: pd.Series
) -> np.ndarray:
    """Compute SMAPE with 50 floor on both values (vectorised)."""
    yt = np.maximum(np.abs(y_true.values), 50.0)
    yp = np.maximum(np.abs(y_pred.values), 50.0)
    denom = (yt + yp) / 2.0
    smape = np.where(denom > 1e-10, np.abs(yt - yp) / denom * 100, 0.0)
    return np.minimum(smape, 50.0)


def get_period(hour_business: int) -> str:
    """Map hour_business to period label."""
    if 9 <= hour_business <= 16:
        return "9_16"
    elif 1 <= hour_business <= 8:
        return "night"
    elif 17 <= hour_business <= 24:
        return "evening"
    return "night"


def parse_weights(weights_str: Optional[str]) -> dict[str, float]:
    """Parse --weights string into dict: 'lightgbm=0.85,dayahead_proxy=0.08,...'."""
    if not weights_str:
        return {}
    weights: dict[str, float] = {}
    for token in weights_str.split(","):
        token = token.strip()
        if "=" not in token:
            raise ValueError(f"Invalid weight token '{token}'; expected 'model=value'")
        model, val = token.split("=", 1)
        weights[model.strip()] = float(val.strip())
    return weights


# ── Fusion implementations ─────────────────────────────────────────────

def fuse_mean(
    df: pd.DataFrame,
    weight_config: dict[str, float],
) -> pd.Series:
    """Simple average of all available model predictions per timestamp."""
    return df.groupby(["business_day", "hour_business"])["y_pred"].mean()


def fuse_lightgbm_anchor(
    df: pd.DataFrame,
    anchor_weight: float,
    weight_config: dict[str, float],
) -> pd.Series:
    """Weighted blend: anchor_weight * LightGBM + (1-anchor_weight) * mean(other)."""
    lgbm = df[df["model_name"] == "lightgbm"].set_index(
        ["business_day", "hour_business"]
    )["y_pred"]
    others = df[df["model_name"] != "lightgbm"]
    other_mean = others.groupby(["business_day", "hour_business"])["y_pred"].mean()
    # Align indices
    idx = other_mean.index.union(lgbm.index)
    lgbm = lgbm.reindex(idx)
    other_mean = other_mean.reindex(idx)
    blended = anchor_weight * lgbm.fillna(other_mean) + (1 - anchor_weight) * other_mean.fillna(lgbm)
    return blended


def fuse_candidate_reference_only(
    df: pd.DataFrame,
    weight_config: dict[str, float],
) -> pd.Series:
    """No fusion: base_fused_pred = LightGBM, other models retained as candidates."""
    lgbm = df[df["model_name"] == "lightgbm"].set_index(
        ["business_day", "hour_business"]
    )["y_pred"]
    return lgbm


def fuse_custom(
    df: pd.DataFrame,
    weight_config: dict[str, float],
) -> pd.Series:
    """Weighted average using explicit per-model weights."""
    if not weight_config:
        raise ValueError("--weights required for fusion-mode=custom")
    # Normalise weights to sum to 1
    total = sum(weight_config.values())
    if total <= 0:
        raise ValueError("Weights must sum to > 0")
    norm = {k: v / total for k, v in weight_config.items()}

    # For each timestamp, compute weighted sum of available model predictions
    result: list[dict] = []
    groups = df.groupby(["business_day", "hour_business"])
    for key, grp in groups:
        available = grp.set_index("model_name")["y_pred"]
        weighted_sum = 0.0
        weight_sum_used = 0.0
        for model, w in norm.items():
            if model in available.index and not pd.isna(available[model]):
                weighted_sum += w * available[model]
                weight_sum_used += w
        fused_val = weighted_sum / weight_sum_used if weight_sum_used > 0 else float("nan")
        result.append({"business_day": key[0], "hour_business": key[1], "fused": fused_val})

    result_df = pd.DataFrame(result)
    return result_df.set_index(["business_day", "hour_business"])["fused"]


FUSION_FUNCTIONS = {
    "mean": fuse_mean,
    "lightgbm_anchor_90": lambda df, w: fuse_lightgbm_anchor(df, 0.9, w),
    "lightgbm_anchor_80": lambda df, w: fuse_lightgbm_anchor(df, 0.8, w),
    "candidate_reference_only": fuse_candidate_reference_only,
    "custom": fuse_custom,
}


def get_weights_description(fusion_mode: str, weight_config: dict[str, float]) -> dict:
    """Return human-readable weight description for manifest."""
    if fusion_mode == "mean":
        return {"method": "equal_weight_mean", "weights": {m: round(1 / len(ALL_MODELS), 4) for m in ALL_MODELS}}
    elif fusion_mode == "lightgbm_anchor_90":
        return {"method": "lightgbm_anchor_90", "anchor_weight": 0.9, "other_weight": 0.1}
    elif fusion_mode == "lightgbm_anchor_80":
        return {"method": "lightgbm_anchor_80", "anchor_weight": 0.8, "other_weight": 0.2}
    elif fusion_mode == "candidate_reference_only":
        return {"method": "base_fused_pred=LightGBM", "weights": {"lightgbm": 1.0}, "candidate_models": BASELINE_MODELS}
    elif fusion_mode == "custom":
        total = sum(weight_config.values()) if weight_config else 1.0
        return {"method": "explicit_weights", "weights": {k: round(v / total, 4) for k, v in weight_config.items()}}
    return {"method": fusion_mode, "weights": weight_config}


# ── Build ──────────────────────────────────────────────────────────────

def load_level0_models(path: Path) -> pd.DataFrame:
    """Load baseline model rows (exclude baseline_fusion)."""
    df = pd.read_csv(path)
    df = df[df["model_name"].isin(BASELINE_MODELS)].copy()
    df["model_name"] = df["model_name"].astype(str)
    return df


def load_lightgbm(path: Path) -> pd.DataFrame:
    """Load LightGBM predictions."""
    df = pd.read_csv(path)
    df = df[df["model_name"] == "lightgbm"].copy()
    return df


def build_fused_pred(
    combined: pd.DataFrame,
    fusion_mode: str,
    weight_config: dict[str, float],
) -> pd.DataFrame:
    """Compute base_fused_pred using the selected fusion mode.

    Returns DataFrame with one row per (business_day, hour_business)
    containing the fused value.
    """
    fn = FUSION_FUNCTIONS.get(fusion_mode)
    if fn is None:
        valid = list(FUSION_FUNCTIONS.keys())
        raise ValueError(f"Unknown fusion_mode '{fusion_mode}'. Valid: {valid}")

    fused_series = fn(combined, weight_config)
    fused = fused_series.reset_index()
    fused.columns = ["business_day", "hour_business", "base_fused_pred"]
    return fused


def build_pack(
    baseline: pd.DataFrame,
    lightgbm: pd.DataFrame,
    fusion_mode: str = "mean",
    weight_config: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Build multi-candidate pack with fused base_fused_pred.

    1. Combine all model rows per timestamp
    2. Compute base_fused_pred using the selected fusion mode
    3. Derive metric columns (residual, abs_error, smape_floor50)

    All metrics in the returned pack are per-row. For evaluation, use
    dedup_timestamp() on the result before computing aggregate metrics.
    """
    weight_config = weight_config or {}

    # Drop existing fused/derived columns (they'll be recomputed)
    drop_cols = [
        "base_fused_pred", "final_pred", "residual", "abs_error",
        "smape_floor50", "high_spike_flag", "severe_underestimate_flag",
    ]
    for df in (baseline, lightgbm):
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

    # Combine
    combined = pd.concat([baseline, lightgbm], ignore_index=True)

    # Report per-timestamp model count
    counts = combined.groupby(["business_day", "hour_business"]).size()
    incomplete = counts[counts < 4]
    if len(incomplete) > 0:
        print(f"  Incomplete timestamps ({len(incomplete)}):")
        for idx in incomplete.index:
            n = int(incomplete[idx])
            rows = combined[
                (combined["business_day"] == idx[0])
                & (combined["hour_business"] == idx[1])
            ]
            models_present = rows["model_name"].unique().tolist()
            print(f"    {idx[0]} hb={idx[1]}: {n} models ({models_present})")

    # Compute fused prediction
    fused = build_fused_pred(combined, fusion_mode, weight_config)

    # Merge fused back — left join keeps all rows
    pack = combined.merge(fused, on=["business_day", "hour_business"], how="left")

    # Set final_pred = base_fused_pred (pre-correction)
    pack["final_pred"] = pack["base_fused_pred"]

    # Derive metrics (per-row)
    pack["residual"] = pack["y_true"] - pack["base_fused_pred"]
    pack["abs_error"] = pack["residual"].abs()
    pack["smape_floor50"] = compute_smape_floor50(
        pack["y_true"], pack["base_fused_pred"]
    )

    # High spike flag
    pack["high_spike_flag"] = (
        (pack["y_true"] - pack["base_fused_pred"]).abs() > 200
    ).astype(int)

    # Severe underestimate flag
    pack["severe_underestimate_flag"] = (
        pack["y_true"] - pack["base_fused_pred"] > 200
    ).astype(int)

    # Meta columns
    pack["source_file"] = f"multicandidate_{fusion_mode}"
    pack["coverage_status"] = "available"
    pack["target"] = "realtime"

    # Ensure period column
    if "period" not in pack.columns:
        pack["period"] = pack["hour_business"].apply(get_period)
    else:
        pack["period"] = pack["period"].fillna(
            pack["hour_business"].apply(get_period)
        )

    # Sort
    pack = pack.sort_values(
        ["business_day", "hour_business", "model_name"]
    ).reset_index(drop=True)

    # Select output columns
    out_cols = [
        "business_day", "hour_business", "timestamp", "period",
        "target", "model_name", "y_pred", "base_fused_pred",
        "final_pred", "y_true", "residual", "abs_error",
        "smape_floor50", "high_spike_flag",
        "severe_underestimate_flag", "source_file", "coverage_status",
    ]
    for col in out_cols:
        if col not in pack.columns:
            pack[col] = None

    return pack[out_cols]


def dedup_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate to 1 row per (business_day, hour_business).

    Since all model rows at the same timestamp share the same base_fused_pred,
    final_pred, and y_true, dropping duplicates is safe for aggregate metrics.
    """
    return df.drop_duplicates(subset=["business_day", "hour_business"]).copy()


def compute_timestamp_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Compute base metrics on timestamp-level data."""
    ts = dedup_timestamp(df)
    smape = float(np.nanmean(compute_smape_floor50(ts["y_true"], ts["base_fused_pred"])))
    severe = int((ts["y_true"] - ts["base_fused_pred"] > 200).sum())
    return {
        "timestamp_level_base_smape": round(smape, 4),
        "timestamp_level_severe_underestimate": severe,
        "n_timestamps": len(ts),
    }


def build_risk_predictions(
    pack: pd.DataFrame,
    source_risk_path: Path,
) -> pd.DataFrame:
    """Build risk predictions (1 row per timestamp)."""
    source = pd.read_csv(source_risk_path)

    # Aggregate to unique (business_day, hour_business) — take MAX score
    risk_map = (
        source.groupby(["business_day", "hour_business"])["spike_risk_score"]
        .max()
        .reset_index()
    )

    timestamps = (
        pack[["business_day", "hour_business", "timestamp", "period"]]
        .drop_duplicates()
        .copy()
    )
    risk = timestamps.merge(risk_map, on=["business_day", "hour_business"], how="left")
    risk["spike_risk_score"] = risk["spike_risk_score"].fillna(0.0)
    risk["high_spike_prob"] = risk["spike_risk_score"]
    risk["spike_risk_flag"] = (risk["spike_risk_score"] > 0.8).astype(int)

    return risk


# ── CLI ────────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build multi-candidate prediction pack for P0 spike correction.",
    )
    parser.add_argument("--level0-pack", default=None,
                        help="Path to Level 0 baseline pack CSV")
    parser.add_argument("--level1-pack", default=None,
                        help="Path to Level 1 LightGBM pack CSV")
    parser.add_argument("--risk-predictions", default=None,
                        help="Path to source risk predictions CSV (with spike_risk_score)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for pack CSVs + manifest")
    parser.add_argument("--start-date", default="2025-11-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", default="2026-02-28",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--fusion-mode", default="mean",
        choices=FUSION_MODES,
        help="Fusion strategy for base_fused_pred (default: mean)",
    )
    parser.add_argument(
        "--weights", default=None,
        help=(
            "Explicit per-model weights for custom fusion mode. "
            "Format: 'lightgbm=0.85,dayahead_proxy=0.08,naive_lag7=0.05,naive_lag1=0.02'"
        ),
    )
    return parser.parse_args(argv)


def resolve_default_path(path_key: str) -> Path:
    """Resolve default paths relative to project root."""
    defaults = {
        "level0": (
            _PROJECT_ROOT
            / "reports/local/p0_full_run/prediction_pack_level0"
            / "prediction_pack_realtime_level0_2025_11_01_2026_02_28.csv"
        ),
        "level1": (
            _PROJECT_ROOT
            / "reports/local/p0_full_run/prediction_pack_level1"
            / "prediction_pack_realtime_level1_2025_11_01_2026_02_28.csv"
        ),
        "risk": (
            _PROJECT_ROOT
            / "reports/local/p0_full_run/level0/risk_model"
            / "spike_risk_predictions.csv"
        ),
        "out": (
            _PROJECT_ROOT
            / "reports/local/p0_full_run/prediction_pack_multicandidate"
        ),
    }
    return defaults[path_key]


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Resolve paths
    level0_path = Path(args.level0_pack) if args.level0_pack else resolve_default_path("level0")
    level1_path = Path(args.level1_pack) if args.level1_pack else resolve_default_path("level1")
    risk_path = Path(args.risk_predictions) if args.risk_predictions else resolve_default_path("risk")
    out_dir = Path(args.out_dir) if args.out_dir else resolve_default_path("out")

    start_date = args.start_date
    end_date = args.end_date
    fusion_mode = args.fusion_mode

    # Parse weights for custom mode
    weight_config: dict[str, float] = {}
    if args.weights:
        weight_config = parse_weights(args.weights)
        print(f"  Weights parsed: {weight_config}")

    if fusion_mode == "custom" and not weight_config:
        print("  [ERR] --fusion-mode=custom requires --weights")
        sys.exit(1)

    print("=" * 60)
    print(f"  Build Multi-Candidate Pack  |  fusion={fusion_mode}")
    print("=" * 60)

    # Validate inputs
    for path, label in [
        (level0_path, "Level 0 pack"),
        (level1_path, "Level 1 pack"),
        (risk_path, "Risk predictions"),
    ]:
        if not path.exists():
            print(f"  [ERR] {label} not found: {path}")
            sys.exit(1)
        print(f"  [OK] {label}: {path.name}")

    # Load
    print("\n  Loading baseline models...")
    baseline = load_level0_models(level0_path)
    print(f"    -> {len(baseline)} rows, models: {baseline['model_name'].unique().tolist()}")

    print("  Loading LightGBM...")
    lightgbm = load_lightgbm(level1_path)
    print(f"    -> {len(lightgbm)} rows")

    # Build multi-candidate pack
    print(f"\n  Building pack (fusion={fusion_mode})...")
    pack = build_pack(baseline, lightgbm, fusion_mode=fusion_mode, weight_config=weight_config)

    # Stats (row-level)
    timestamps = pack[["business_day", "hour_business"]].drop_duplicates()
    coverage_dates = pack["business_day"].nunique()
    n_rows = len(pack)
    n_ts = len(timestamps)
    print(f"    -> {n_rows} rows ({n_ts} timestamps x ~{n_rows // max(n_ts, 1)} models)")
    print(f"    -> {coverage_dates} unique business_days")

    # Timestamp-level metrics (deduplicated)
    ts_metrics = compute_timestamp_metrics(pack)
    print(f"\n  Timestamp-level base metrics (deduplicated):")
    print(f"    sMAPE_floor50:           {ts_metrics['timestamp_level_base_smape']}")
    print(f"    Severe underestimates:   {ts_metrics['timestamp_level_severe_underestimate']}")
    print(f"    Unique timestamps:       {ts_metrics['n_timestamps']}")

    # Output
    out_dir.mkdir(parents=True, exist_ok=True)

    start_compact = start_date.replace("-", "_")
    end_compact = end_date.replace("-", "_")
    pack_filename = f"prediction_pack_realtime_multicandidate_{start_compact}_{end_compact}.csv"
    pack_path = out_dir / pack_filename
    pack.to_csv(pack_path, index=False, encoding="utf-8")
    print(f"\n  [OK] Prediction pack: {pack_path}")

    # Build and write risk predictions
    risk = build_risk_predictions(pack, risk_path)
    risk_path_out = out_dir / "risk_predictions_multicandidate.csv"
    risk.to_csv(risk_path_out, index=False, encoding="utf-8")
    print(f"  [OK] Risk predictions: {risk_path_out}")

    # Weights description for manifest
    weights_desc = get_weights_description(fusion_mode, weight_config)

    # Known limitations (auto-generated)
    known_limitations = []
    if fusion_mode == "candidate_reference_only":
        known_limitations.append(
            "candidate_reference_only mode sets base_fused_pred = LightGBM; "
            "candidate rows from other models retained as reference only"
        )
    if n_ts < 2800:
        known_limitations.append(
            f"Only {n_ts} timestamps (expected ~2880 for Nov-Feb); "
            "some dates may have incomplete coverage"
        )
    known_limitations.append(
        "Row-level sMAPE inflates counts by ~4x due to multi-model rows; "
        "use timestamp-level metrics for final decisions"
    )

    # Manifest
    manifest = {
        "script": "scripts/build_multicandidate_pack.py",
        "fusion_mode": fusion_mode,
        "weights_used": weights_desc,
        "date_range": {"start": start_date, "end": end_date},
        "n_rows": n_rows,
        "n_timestamps": n_ts,
        "n_business_days": int(coverage_dates),
        "n_models_per_timestamp": 4,
        "timestamp_level_base_smape": ts_metrics["timestamp_level_base_smape"],
        "timestamp_level_severe_underestimate": ts_metrics["timestamp_level_severe_underestimate"],
        "input_pack_paths": {
            "level0": str(level0_path),
            "level1": str(level1_path),
        },
        "risk_prediction_path": str(risk_path),
        "known_limitations": known_limitations,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": (
            f"Multi-candidate prediction pack using fusion_mode='{fusion_mode}'. "
            "Intended for Phase 2 anchored fusion + correction activation evaluation."
        ),
    }
    manifest_path = out_dir / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  [OK] Manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
