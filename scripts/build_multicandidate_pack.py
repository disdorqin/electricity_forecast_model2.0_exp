#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_multicandidate_pack.py — Build multi-candidate prediction pack for P0.

Merges LightGBM (Level 1) + baseline models (naive_lag1, naive_lag7, dayahead_proxy)
into a single prediction pack where base_fused_pred = mean of all 4 models.

This gives the correction pipeline a meaningful fused prediction to compare
individual candidates against — enabling lift activation that single-model
packs cannot achieve.

Usage:
    python scripts/build_multicandidate_pack.py

Output:
    reports/local/p0_full_run/prediction_pack_multicandidate/
      - prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv
      - risk_predictions_multicandidate.csv
      - build_manifest.json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Paths ──────────────────────────────────────────────────────────────

P0_START = "2025-11-01"
P0_END = "2026-02-28"

LEVEL0_PACK = (
    _PROJECT_ROOT
    / "reports/local/p0_full_run/prediction_pack_level0"
    / "prediction_pack_realtime_level0_2025_11_01_2026_02_28.csv"
)
LEVEL1_PACK = (
    _PROJECT_ROOT
    / "reports/local/p0_full_run/prediction_pack_level1"
    / "prediction_pack_realtime_level1_2025_11_01_2026_02_28.csv"
)
OUT_DIR = _PROJECT_ROOT / "reports/local/p0_full_run/prediction_pack_multicandidate"

RISK_PREDICTIONS_SOURCE = (
    _PROJECT_ROOT
    / "reports/local/p0_full_run/level0/risk_model"
    / "spike_risk_predictions.csv"
)

BASELINE_MODELS = ["naive_lag1", "naive_lag7", "dayahead_proxy"]
ALL_MODELS = BASELINE_MODELS + ["lightgbm"]


# ── Helpers ────────────────────────────────────────────────────────────

def compute_smape_floor50(y_true: pd.Series, y_pred: pd.Series) -> np.ndarray:
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


# ── Build ──────────────────────────────────────────────────────────────

def load_level0_models(path: Path) -> pd.DataFrame:
    """Load baseline model rows (exclude baseline_fusion)."""
    df = pd.read_csv(path)
    df = df[df["model_name"].isin(BASELINE_MODELS)].copy()
    # Normalise model_name for clarity
    df["model_name"] = df["model_name"].astype(str)
    return df


def load_lightgbm(path: Path) -> pd.DataFrame:
    """Load LightGBM predictions."""
    df = pd.read_csv(path)
    df = df[df["model_name"] == "lightgbm"].copy()
    return df


def build_pack(
    baseline: pd.DataFrame,
    lightgbm: pd.DataFrame,
) -> pd.DataFrame:
    """Build multi-candidate pack with fused base_fused_pred.

    1. Combine all 4 model rows per timestamp
    2. Compute base_fused_pred = mean of all 4 y_pred values per timestamp
    3. Derive metric columns (residual, abs_error, smape_floor50)
    """
    # Drop existing fused/derived columns (they'll be recomputed)
    drop_cols = ["base_fused_pred", "final_pred", "residual", "abs_error",
                 "smape_floor50", "high_spike_flag", "severe_underestimate_flag"]
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
        print(f"  ⚠ {len(incomplete)} timestamps have < 4 models:")
        for idx in incomplete.index:
            n = int(incomplete[idx])
            # First non-complete
            if n < 4:
                rows = combined[
                    (combined["business_day"] == idx[0])
                    & (combined["hour_business"] == idx[1])
                ]
                models_present = rows["model_name"].unique().tolist()
                print(f"    {idx[0]} hb={idx[1]}: {n} models ({models_present})")

    # Compute fused prediction = mean of available models per timestamp
    # (gracefully handles timestamps with < 4 models)
    fused = (
        combined.groupby(["business_day", "hour_business"])["y_pred"]
        .mean()
        .reset_index()
        .rename(columns={"y_pred": "base_fused_pred"})
    )

    # Merge fused back — left join keeps all rows
    pack = combined.merge(fused, on=["business_day", "hour_business"], how="left")

    # Set final_pred = base_fused_pred (pre-correction)
    pack["final_pred"] = pack["base_fused_pred"]

    # Derive metrics
    pack["residual"] = pack["y_true"] - pack["base_fused_pred"]
    pack["abs_error"] = pack["residual"].abs()
    pack["smape_floor50"] = compute_smape_floor50(
        pack["y_true"], pack["base_fused_pred"]
    )

    # High spike flag (residual > 200 threshold)
    pack["high_spike_flag"] = (
        (pack["y_true"] - pack["base_fused_pred"]).abs() > 200
    ).astype(int)

    # Severe underestimate flag
    pack["severe_underestimate_flag"] = (
        pack["y_true"] - pack["base_fused_pred"] > 200
    ).astype(int)

    # Meta columns
    pack["source_file"] = "multicandidate_v1"
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


def build_risk_predictions(
    pack: pd.DataFrame,
    source_risk_path: Path,
) -> pd.DataFrame:
    """Build risk predictions for the multi-candidate pack.

    Takes spike_risk_score from the source risk predictions (merged on
    business_day + hour_business) and propagates it to all model rows.
    Also creates high_spike_prob alias for the correction pipeline.
    """
    source = pd.read_csv(source_risk_path)

    # Aggregate to unique (business_day, hour_business) — take MAX score
    # (different models have different spike risk scores per timestamp;
    #  using max ensures the fused correction is triggered when ANY model
    #  detects a spike risk — conservative for safety)
    risk_map = (
        source.groupby(["business_day", "hour_business"])["spike_risk_score"]
        .max()
        .reset_index()
    )

    # Build risk predictions from PACK rows, not from source directly
    # Use pack's unique timestamps as the frame, attach spike scores
    timestamps = pack[["business_day", "hour_business", "timestamp", "period"]].drop_duplicates().copy()
    risk = timestamps.merge(risk_map, on=["business_day", "hour_business"], how="left")
    risk["spike_risk_score"] = risk["spike_risk_score"].fillna(0.0)

    # Create high_spike_prob alias (used by run_correction)
    risk["high_spike_prob"] = risk["spike_risk_score"]

    # Spike flag (threshold 0.8 matches predict_realtime_spike_risk.py)
    risk["spike_risk_flag"] = (risk["spike_risk_score"] > 0.8).astype(int)

    return risk


# ── Main ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Build Multi-Candidate Prediction Pack")
    print("=" * 60)

    # Validate inputs
    for path, label in [
        (LEVEL0_PACK, "Level 0 pack"),
        (LEVEL1_PACK, "Level 1 pack"),
        (RISK_PREDICTIONS_SOURCE, "Risk predictions"),
    ]:
        if not path.exists():
            print(f"  ❌ {label} not found: {path}")
            sys.exit(1)
        print(f"  ✅ {label}: {path.name}")

    # Load
    print("\n  Loading baseline models...")
    baseline = load_level0_models(LEVEL0_PACK)
    print(f"    → {len(baseline)} rows, models: {baseline['model_name'].unique().tolist()}")

    print("  Loading LightGBM...")
    lightgbm = load_lightgbm(LEVEL1_PACK)
    print(f"    → {len(lightgbm)} rows")

    # Build multi-candidate pack
    print("\n  Building multi-candidate pack...")
    pack = build_pack(baseline, lightgbm)

    # Stats
    timestamps = pack[["business_day", "hour_business"]].drop_duplicates()
    coverage_dates = pack["business_day"].nunique()
    print(f"    → {len(pack)} rows ({len(timestamps)} timestamps × 4 models)")
    print(f"    → {coverage_dates} unique business_days")

    # Fusion statistics
    residuals = pack["residual"]
    print(f"\n  Fusion stats (base_fused_pred):")
    print(f"    Mean residual:    {residuals.mean():.2f}")
    print(f"    MAE:             {residuals.abs().mean():.2f}")
    print(f"    RMSE:            {np.sqrt((residuals ** 2).mean()):.2f}")
    print(
        f"    sMAPE_floor50:   {np.mean(compute_smape_floor50(pack['y_true'], pack['base_fused_pred'])):.4f}"
    )

    # Severe underestimates
    sev = (pack["y_true"] - pack["base_fused_pred"] > 200).sum()
    sev_by_model = (
        pack[pack["y_true"] - pack["base_fused_pred"] > 200]
        .groupby("model_name")
        .size()
        .to_dict()
    )
    print(f"\n  Severe underestimates (y_true - fused > 200):")
    print(f"    Total: {sev}")
    for m, c in sorted(sev_by_model.items()):
        print(f"      {m}: {c}")

    # Output
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pack_path = (
        OUT_DIR
        / "prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv"
    )
    pack.to_csv(pack_path, index=False, encoding="utf-8-sig")
    print(f"\n  ✅ Prediction pack: {pack_path}")

    # Build and write risk predictions
    risk = build_risk_predictions(pack, RISK_PREDICTIONS_SOURCE)
    risk_path = OUT_DIR / "risk_predictions_multicandidate.csv"
    risk.to_csv(risk_path, index=False, encoding="utf-8-sig")
    print(f"  ✅ Risk predictions: {risk_path}")

    # Manifest
    manifest = {
        "script": "scripts/build_multicandidate_pack.py",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": ALL_MODELS,
        "fusion_method": "mean of all 4 model predictions per timestamp",
        "date_range": {"start": P0_START, "end": P0_END},
        "total_rows": len(pack),
        "total_timestamps": len(timestamps),
        "coverage_dates": coverage_dates,
        "inputs": {
            "level0_pack": str(LEVEL0_PACK),
            "level1_pack": str(LEVEL1_PACK),
            "risk_predictions_source": str(RISK_PREDICTIONS_SOURCE),
        },
        "fusion_metrics": {
            "mean_residual": round(float(residuals.mean()), 4),
            "mae": round(float(residuals.abs().mean()), 4),
            "rmse": round(float(np.sqrt((residuals ** 2).mean())), 4),
            "smape_floor50": round(
                float(np.mean(compute_smape_floor50(pack["y_true"], pack["base_fused_pred"]))), 4
            ),
            "severe_underestimates": int(sev),
        },
        "severe_underestimates_by_model": sev_by_model,
        "purpose": (
            "Multi-candidate pack with fused base_fused_pred for correction pipeline. "
            "Enables lift activation by providing a meaningful ensemble reference "
            "across LightGBM + 3 baseline models."
        ),
    }
    manifest_path = OUT_DIR / "build_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  ✅ Manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
