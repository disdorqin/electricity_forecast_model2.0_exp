#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_p5_deep_ts_model_zoo.py — P5 Deep + Time-Series Model Zoo runner.

Scans available deep/TS models, runs inference for each, converts predictions
to W1 schema (business_day, hour_business, timestamp, y_pred, model_name,
source_file, prediction_mode, leakage_safe), evaluates metrics, and reports
diversity vs LightGBM.

CLI:
    python scripts/run_p5_deep_ts_model_zoo.py \
        --dataset-root reports/local/p5_model_zoo \
        --models timesfm,patchtst,itransformer,tsmixer,nhits,timemixer_plus,rt916_selective \
        --max-runtime-minutes 120 \
        --out-dir reports/local/p5_deep_ts_model_zoo

Usage:
    # Run only TimesFM (fast, preferred)
    python scripts/run_p5_deep_ts_model_zoo.py --models timesfm

    # Run all feasible models
    python scripts/run_p5_deep_ts_model_zoo.py --models all
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("p5_model_zoo")


# ── Constants ──────────────────────────────────────────────────────────
DATA_PATH = str(_PROJECT_ROOT / "data" / "shandong_pmos_hourly.xlsx")
FORECAST_START = "2025-11-01"
FORECAST_END = "2026-02-28"
W1_COLS = ["model_name", "business_day", "hour_business", "timestamp",
           "y_pred", "source_file", "prediction_mode", "leakage_safe"]

# LightGBM baseline for comparison
LGBM_PRED_PATH = str(_PROJECT_ROOT / "reports/local/p0_phase2_anchored/packs/lightgbm_anchor_90"
                      "/prediction_pack_realtime_multicandidate_2025_11_01_2026_02_28.csv")


@dataclass
class ModelInfo:
    """Inventory entry for one model."""
    name: str
    available: bool = False
    has_checkpoint: bool = False
    can_infer: bool = False
    can_train_small: bool = False
    estimated_runtime_min: float = 999.0
    reason: str = ""


@dataclass
class ModelResult:
    """Output from one model run."""
    name: str
    predictions: Optional[pd.DataFrame] = None
    runtime_seconds: float = 0.0
    success: bool = False
    error: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


# ── Data loading ──────────────────────────────────────────────────────

def load_raw_data() -> pd.DataFrame:
    """Load hourly electricity price data."""
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    df["时刻"] = pd.to_datetime(df["时刻"])
    df = df.sort_values("时刻").reset_index(drop=True)
    return df


def load_lightgbm_baseline() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load LightGBM predictions for comparison.

    Uses the single-model (lightgbm-only) prediction pack to get
    clean y_pred per timestamp for diversity analysis.
    """
    pp = pd.read_csv(LGBM_PRED_PATH)
    lgbm = pp[pp["model_name"] == "lightgbm"].copy()
    lgbm["timestamp"] = pd.to_datetime(lgbm["timestamp"])
    lgbm = lgbm.drop_duplicates(subset=["business_day", "hour_business"])
    lgbm = lgbm.sort_values("timestamp").reset_index(drop=True)

    # y_true for evaluation
    y_true = lgbm["y_true"].values
    y_pred_lgbm = lgbm["y_pred"].values
    return lgbm, y_true, y_pred_lgbm


# ── Schema converter ──────────────────────────────────────────────────

def to_w1_schema(
    model_name: str,
    timestamps: pd.DatetimeIndex | list,
    y_pred: np.ndarray | list,
    source_file: str = "run_p5_deep_ts_model_zoo.py",
    prediction_mode: str = "direct",
    leakage_safe: bool = True,
) -> pd.DataFrame:
    """Convert raw predictions to W1 schema DataFrame."""
    ts = pd.to_datetime(timestamps)
    df = pd.DataFrame({
        "timestamp": ts,
        "y_pred": np.asarray(y_pred, dtype=float),
    })
    df["business_day"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour_business"] = ((df["timestamp"].dt.hour + 1) % 24).replace(0, 24)
    df["model_name"] = model_name
    df["source_file"] = source_file
    df["prediction_mode"] = prediction_mode
    df["leakage_safe"] = leakage_safe
    return df[W1_COLS].copy()


# ── Metrics ───────────────────────────────────────────────────────────

def compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """sMAPE floor50."""
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    s = np.where(denom > 1e-10, np.abs(y_true - y_pred) / denom * 100, 0.0)
    return float(np.minimum(s, 50.0).mean())


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    hour_business: np.ndarray,
    high_spike_flag: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute all evaluation metrics."""
    m: dict[str, Any] = {}
    m["smape_floor50"] = round(compute_smape(y_true, y_pred), 4)

    # 9_16 period
    mask_9_16 = (hour_business >= 9) & (hour_business <= 16)
    if mask_9_16.sum() > 0:
        m["9_16_smape_floor50"] = round(compute_smape(y_true[mask_9_16], y_pred[mask_9_16]), 4)
    else:
        m["9_16_smape_floor50"] = None

    # Severe underestimates
    m["severe_underestimate_count"] = int(np.sum((y_true - y_pred) > 200))

    # High spike metrics
    if high_spike_flag is not None and high_spike_flag.sum() > 0:
        spike = high_spike_flag == 1
        m["high_spike_mae"] = round(float(np.mean(np.abs(y_true[spike] - y_pred[spike]))), 4)
    else:
        m["high_spike_mae"] = None

    # Coverage
    m["n_hours"] = len(y_true)
    m["n_days"] = int(np.ceil(len(y_true) / 24))

    return m


def compute_diversity(
    y_pred_model: np.ndarray,
    y_pred_lgbm: np.ndarray,
) -> dict[str, float]:
    """Compute diversity metrics vs LightGBM."""
    # Correlation
    mask = ~(np.isnan(y_pred_model) | np.isnan(y_pred_lgbm))
    if mask.sum() < 2:
        return {"correlation_vs_lgbm": None, "mae_diff_vs_lgbm": None}

    corr = float(np.corrcoef(y_pred_model[mask], y_pred_lgbm[mask])[0, 1])
    mae_diff = float(np.mean(np.abs(y_pred_model[mask] - y_pred_lgbm[mask])))

    return {
        "correlation_vs_lgbm": round(corr, 4),
        "mae_diff_vs_lgbm": round(mae_diff, 4),
    }


# ── Model-specific runners ───────────────────────────────────────────

def run_timesfm_inference(data_df: pd.DataFrame) -> ModelResult:
    """Run TimesFM full-window inference for the forecast period.

    Uses the native timesfm package (JAX backend).
    Strategy: for each forecast day, build context from preceding 512 hours,
    forecast 24 hours ahead, collect all daily predictions.
    """
    model_name = "timesfm"
    logger.info("  [TimesFM] Loading model...")

    try:
        import timesfm
        from timesfm import ForecastConfig
    except ImportError:
        return ModelResult(name=model_name, success=False, error="timesfm package not installed")

    t_start = time.time()

    # Load model
    try:
        model = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
        config = ForecastConfig(
            max_context=512,
            max_horizon=128,
            per_core_batch_size=32,
        )
        model.compile(config)
    except Exception as e:
        return ModelResult(name=model_name, success=False, error=f"model load failed: {e}")

    logger.info(f"  [TimesFM] Model loaded in {time.time() - t_start:.1f}s")

    # Build forecast schedule
    start_dt = pd.Timestamp(FORECAST_START)
    end_dt = pd.Timestamp(FORECAST_END)
    all_dates = pd.date_range(start_dt, end_dt, freq="D")

    # Ensure data sorted by time
    data = data_df.copy().sort_values("时刻")
    all_times = data["时刻"].values
    rt_prices = data["实时电价"].values.astype(np.float32)

    all_preds: list[np.ndarray] = []
    all_timestamps: list[pd.Timestamp] = []

    logger.info(f"  [TimesFM] Forecasting {len(all_dates)} days...")

    for i, date in enumerate(all_dates):
        if i > 0 and i % 20 == 0:
            elapsed = time.time() - t_start
            logger.info(f"  [TimesFM] {i}/{len(all_dates)} days ({elapsed:.0f}s elapsed)")

        # Context: 512 hours ending at 00:00 of the forecast day
        context_end = pd.Timestamp(date)  # 00:00 of forecast day
        context_start = context_end - pd.Timedelta(hours=512)

        mask = (all_times >= context_start) & (all_times < context_end)
        context_vals = rt_prices[mask]

        if len(context_vals) < 128:
            # Not enough context — use all available
            mask2 = all_times < context_end
            context_vals = rt_prices[mask2]

        # Pad or truncate to 512
        if len(context_vals) < 512:
            context_vals = np.pad(context_vals, (512 - len(context_vals), 0),
                                  "constant", constant_values=np.nan)
        elif len(context_vals) > 512:
            context_vals = context_vals[-512:]

        # Forecast 24 hours (next day 01:00 to 24:00)
        try:
            output_points, _ = model.forecast(horizon=24, inputs=[context_vals])
            day_preds = output_points[0]  # shape (24,)

            # Generate timestamps: forecast_day 01:00 to forecast_day 24:00
            day_start = pd.Timestamp(date) + pd.Timedelta(hours=1)
            day_ts = [day_start + pd.Timedelta(hours=h) for h in range(24)]
            all_preds.append(day_preds)
            all_timestamps.extend(day_ts)

        except Exception as e:
            logger.warning(f"  [TimesFM] Day {date.date()} failed: {e}")
            # Fill with NaN
            day_start = pd.Timestamp(date) + pd.Timedelta(hours=1)
            day_ts = [day_start + pd.Timedelta(hours=h) for h in range(24)]
            all_preds.append(np.full(24, np.nan))
            all_timestamps.extend(day_ts)

    runtime = time.time() - t_start
    logger.info(f"  [TimesFM] Done in {runtime:.0f}s")

    if not all_preds:
        return ModelResult(name=model_name, success=False, error="no predictions generated",
                           runtime_seconds=runtime)

    all_preds_arr = np.concatenate(all_preds)
    df = to_w1_schema(
        model_name=model_name,
        timestamps=all_timestamps,
        y_pred=all_preds_arr,
        prediction_mode="direct_24h",
        leakage_safe=True,
    )

    return ModelResult(name=model_name, predictions=df, success=True, runtime_seconds=runtime)


def run_timemixer_cached(data_df: pd.DataFrame) -> list[ModelResult]:
    """Check TimeMixer output directories for cached predictions.

    Scans TimeMixer/outputs_* for predictions_raw.csv files, converts
    to W1 schema, and checks if they cover the forecast period.
    """
    results: list[ModelResult] = []
    tm_dir = _PROJECT_ROOT / "TimeMixer"

    for subdir in sorted(tm_dir.glob("outputs_*")):
        csv_path = subdir / "predictions_raw.csv"
        if not csv_path.exists():
            continue

        try:
            df = pd.read_csv(csv_path)
            if "ds" not in df.columns:
                continue

            ds = pd.to_datetime(df["ds"])
            if ds.min() > pd.Timestamp(FORECAST_START) or ds.max() < pd.Timestamp(FORECAST_END):
                continue

            # Find realtime price column
            rt_col = None
            for col in ["realtime_price", "实时电价", "rt_price"]:
                if col in df.columns:
                    rt_col = col
                    break
            if rt_col is None:
                continue

            y_pred = df[rt_col].values.astype(float)
            ts = ds.values

            w1 = to_w1_schema(
                model_name=f"timemixer_{subdir.name.replace('outputs_', '')}",
                timestamps=ts,
                y_pred=y_pred,
                source_file=str(csv_path.relative_to(_PROJECT_ROOT)),
                prediction_mode="seq2seq",
                leakage_safe=False,  # TimeMixer outputs may use future data
            )

            results.append(ModelResult(
                name=f"timemixer_{subdir.name.replace('outputs_', '')}",
                predictions=w1,
                success=True,
                runtime_seconds=0,
            ))
            logger.info(f"  [TimeMixer] Found cached: {subdir.name} ({len(df)} rows)")

        except Exception as e:
            logger.debug(f"  [TimeMixer] Skip {subdir.name}: {e}")

    return results


def check_rt916_selective() -> ModelResult:
    """Check RT916 feasibility and available predictions.

    RT916 requires training. Only performs feasibility assessment
    and checks if any cached predictions exist.
    """
    model_name = "rt916_selective"
    rt_dir = _PROJECT_ROOT / "RT916_SpikeFusionNet"

    if not rt_dir.exists():
        return ModelResult(name=model_name, success=False, error="RT916 directory not found")

    # Check for any cached predictions
    cached = list(rt_dir.glob("outputs_*/pred_*.csv"))
    if cached:
        logger.info(f"  [RT916] Found {len(cached)} cached prediction files")
    else:
        logger.info(f"  [RT916] No cached predictions found")

    # Check if pipeline can run
    try:
        sys.path.insert(0, str(rt_dir))
        import pipeline as rt_pipeline
        has_pipeline = True
    except Exception:
        has_pipeline = False

    return ModelResult(
        name=model_name,
        success=False,  # No inference run, feasibility only
        error=f"RT916 needs training. Pipeline module: {'available' if has_pipeline else 'not found'}. "
              f"Cached files: {len(cached)}.",
        runtime_seconds=0,
    )


def check_sgdfnet_feasibility() -> ModelResult:
    """Check SGDFNet feasibility (no inference run)."""
    model_name = "sgdfnet"
    sg_dir = _PROJECT_ROOT / "SGDFNet"

    if not sg_dir.exists():
        return ModelResult(name=model_name, success=False, error="SGDFNet directory not found")

    try:
        sys.path.insert(0, str(sg_dir))
        import model as sg_model
        import dataprocess as sg_data
        has_code = True
    except Exception:
        has_code = False

    return ModelResult(
        name=model_name,
        success=False,
        error=f"SGDFNet needs training. Model code: {'available' if has_code else 'not found'}. "
              f"Not run.",
        runtime_seconds=0,
    )


# ── Main runner ───────────────────────────────────────────────────────

def inventory_models() -> dict[str, ModelInfo]:
    """Scan repo and environment for all candidate models."""
    info: dict[str, ModelInfo] = {}

    # TimesFM
    tf_avail = True
    try:
        import timesfm
    except ImportError:
        tf_avail = False
    tf_ckpt = list(_PROJECT_ROOT.glob("models/timesFM/model.safetensors"))
    info["timesfm"] = ModelInfo(
        name="timesfm",
        available=tf_avail,
        has_checkpoint=len(tf_ckpt) > 0,
        can_infer=tf_avail and len(tf_ckpt) > 0,
        can_train_small=False,
        estimated_runtime_min=5,  # ~2-5 min for 120 days
        reason="Pretrained foundation model, full-window inference"
    )

    # PatchTST — check if available via any package
    patch_avail = False
    try:
        from gluonts.model import patchtst  # noqa
        patch_avail = True
    except ImportError:
        pass
    info["patchtst"] = ModelInfo(
        name="patchtst",
        available=patch_avail,
        has_checkpoint=False,
        can_infer=False,
        can_train_small=patch_avail,
        estimated_runtime_min=120 if patch_avail else 999,
        reason="GluonTS PatchTST: not installed" if not patch_avail else "Available via gluon-ts"
    )

    # iTransformer — check if available
    it_avail = False
    try:
        import itransformer  # noqa
        it_avail = True
    except ImportError:
        pass
    info["itransformer"] = ModelInfo(
        name="itransformer",
        available=it_avail,
        has_checkpoint=False,
        can_infer=False,
        can_train_small=False,
        estimated_runtime_min=999,
        reason="Not installed (no package found)"
    )

    # TSMixer — check
    ts_avail = False
    try:
        import tsmixer  # noqa
        ts_avail = True
    except ImportError:
        pass
    info["tsmixer"] = ModelInfo(
        name="tsmixer",
        available=ts_avail,
        has_checkpoint=False,
        can_infer=False,
        can_train_small=False,
        estimated_runtime_min=999,
        reason="Not installed"
    )

    # N-HiTS — check
    nh_avail = False
    try:
        from gluonts.model import n_hits  # noqa
        nh_avail = True
    except ImportError:
        pass
    info["nhits"] = ModelInfo(
        name="nhits",
        available=nh_avail,
        has_checkpoint=False,
        can_infer=False,
        can_train_small=nh_avail,
        estimated_runtime_min=120 if nh_avail else 999,
        reason="Not installed" if not nh_avail else "Available via gluon-ts"
    )

    # TimeMixer++ (existing code)
    tm_dir = _PROJECT_ROOT / "TimeMixer"
    has_tm_code = tm_dir.exists() and (tm_dir / "model.py").exists()
    info["timemixer_plus"] = ModelInfo(
        name="timemixer_plus",
        available=has_tm_code,
        has_checkpoint=len(list(tm_dir.glob("outputs_*/predictions_raw.csv"))) > 0,
        can_infer=has_tm_code,
        can_train_small=has_tm_code,
        estimated_runtime_min=60 if has_tm_code else 999,
        reason="Existing repo code. Has cached predictions." if has_tm_code else "No code found"
    )

    # RT916 selective
    rt_dir = _PROJECT_ROOT / "RT916_SpikeFusionNet"
    has_rt = rt_dir.exists()
    info["rt916_selective"] = ModelInfo(
        name="rt916_selective",
        available=has_rt,
        has_checkpoint=False,
        can_infer=False,  # Needs training
        can_train_small=has_rt,
        estimated_runtime_min=120 if has_rt else 999,
        reason="Code exists but no checkpoints. Needs training for inference."
    )

    # SGDFNet feasibility
    sg_dir = _PROJECT_ROOT / "SGDFNet"
    has_sg = sg_dir.exists()
    info["sgdfnet"] = ModelInfo(
        name="sgdfnet",
        available=has_sg,
        has_checkpoint=False,
        can_infer=False,
        can_train_small=has_sg,
        estimated_runtime_min=999,
        reason="Code exists but needs training. Feasibility assessment only."
    )

    return info


def run_models(
    model_names: list[str],
    max_runtime_seconds: int,
    out_dir: Path,
) -> dict[str, ModelResult]:
    """Run requested models within time budget."""
    inventory = inventory_models()
    results: dict[str, ModelResult] = {}
    total_runtime = 0
    data_df = load_raw_data()

    for name in model_names:
        if name not in inventory:
            logger.warning(f"  Unknown model: {name}, skipping")
            continue

        info = inventory[name]
        logger.info(f"\n  {'='*50}")
        logger.info(f"  Model: {name}")
        logger.info(f"  Available: {info.available}")
        logger.info(f"  Checkpoint: {info.has_checkpoint}")
        logger.info(f"  Can infer: {info.can_infer}")
        logger.info(f"  Estimated runtime: {info.estimated_runtime_min} min")
        logger.info(f"  {'='*50}")

        if not info.available:
            results[name] = ModelResult(name=name, success=False,
                                        error=f"Not available: {info.reason}")
            continue

        if info.estimated_runtime_min * 60 > max_runtime_seconds - total_runtime:
            results[name] = ModelResult(name=name, success=False,
                                        error=f"Estimated runtime {info.estimated_runtime_min}min "
                                              f"exceeds remaining budget "
                                              f"{(max_runtime_seconds - total_runtime)//60}min")
            continue

        # Run model
        if name == "timesfm":
            result = run_timesfm_inference(data_df)

        elif name == "timemixer_plus":
            tm_results = run_timemixer_cached(data_df)
            if tm_results:
                for r in tm_results:
                    results[r.name] = r
            results[name] = ModelResult(name=name, success=False,
                                        error="No cached predictions for forecast period")
            continue

        elif name == "rt916_selective":
            result = check_rt916_selective()

        elif name == "sgdfnet":
            result = check_sgdfnet_feasibility()

        else:
            results[name] = ModelResult(name=name, success=False,
                                        error=f"Runner not implemented")
            continue

        total_runtime += result.runtime_seconds
        results[name] = result

        # Compute metrics if successful
        if result.success and result.predictions is not None:
            preds = result.predictions
            # Merge with y_true
            ts = pd.to_datetime(preds["timestamp"])
            hour_business = preds["hour_business"].values

            # Find matching y_true from data_df
            merged = pd.merge(
                preds,
                data_df[["时刻", "实时电价"]].rename(
                    columns={"时刻": "timestamp", "实时电价": "y_true"}),
                on="timestamp",
                how="left"
            )
            valid = merged.dropna(subset=["y_true", "y_pred"])
            if len(valid) == 0:
                result.metrics["error"] = "No matching y_true values"
                result.metrics = compute_metrics(np.array([0]), np.array([0]), np.array([0]))
            else:
                y_t = valid["y_true"].values
                y_p = valid["y_pred"].values
                hb = valid["hour_business"].values
                result.metrics = compute_metrics(y_t, y_p, hb)

                # Diversity vs LightGBM — align by (business_day, hour_business)
                lgbm_df_baseline, _, _ = load_lightgbm_baseline()
                aligned = pd.merge(
                    valid[["business_day", "hour_business", "y_pred"]],
                    lgbm_df_baseline[["business_day", "hour_business", "y_pred"]].rename(
                        columns={"y_pred": "y_pred_lgbm"}),
                    on=["business_day", "hour_business"],
                    how="inner"
                )
                if len(aligned) > 10:
                    div = compute_diversity(
                        aligned["y_pred"].values,
                        aligned["y_pred_lgbm"].values,
                    )
                    result.metrics.update(div)

            # Save W1 predictions
            pred_out = out_dir / "predictions" / f"{name}_w1.csv"
            pred_out.parent.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(pred_out, index=False, encoding="utf-8-sig")
            result.metrics["predictions_csv"] = str(pred_out)

    return results


# ── Report ────────────────────────────────────────────────────────────

def generate_summary(results: dict[str, ModelResult], model_names: list[str],
                     total_runtime: float, out_dir: Path) -> dict[str, Any]:
    """Generate summary report as dict."""
    summary: dict[str, Any] = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_period": f"{FORECAST_START} ~ {FORECAST_END}",
        "total_runtime_seconds": round(total_runtime, 1),
        "models_requested": model_names,
        "models_run": [],
        "results": {},
    }

    for name, result in results.items():
        entry = {
            "success": result.success,
            "runtime_seconds": round(result.runtime_seconds, 1),
        }
        if result.success:
            entry["metrics"] = result.metrics
        else:
            entry["error"] = result.error

        if result.success or name in ("rt916_selective", "sgdfnet"):
            summary["models_run"].append(name)

        summary["results"][name] = entry

    return summary


# ── CLI ───────────────────────────────────────────────────────────────

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5 Deep Time-Series Model Zoo evaluation.",
    )
    parser.add_argument("--dataset-root", default="reports/local/p5_model_zoo",
                        help="Dataset root (for future cached datasets)")
    parser.add_argument("--models", nargs="+", default=["timesfm"],
                        help="Models to run: timesfm, patchtst, itransformer, tsmixer, "
                             "nhits, timemixer_plus, rt916_selective, sgdfnet, or 'all'")
    parser.add_argument("--max-runtime-minutes", type=int, default=120,
                        help="Max total runtime in minutes")
    parser.add_argument("--out-dir", default="reports/local/p5_deep_ts_model_zoo",
                        help="Output directory for results and predictions")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  P5 Deep Time-Series Model Zoo")
    print("=" * 60)

    # Step 1: Inventory
    print("\n[1/4] Inventorying available models...")
    inventory = inventory_models()
    print(f"  {'Model':<22} {'Available':<12} {'Checkpoint':<12} {'Can Infer':<12} {'Est. Runtime':<12}")
    print(f"  {'-'*70}")
    for name, info in sorted(inventory.items()):
        print(f"  {name:<22} {str(info.available):<12} {str(info.has_checkpoint):<12} "
              f"{str(info.can_infer):<12} {info.estimated_runtime_min:<8.0f} min")

    # Resolve model list
    model_names = args.models
    if "all" in model_names:
        model_names = [n for n, info in sorted(inventory.items())
                       if info.available or n in ("timesfm", "timemixer_plus",
                                                  "rt916_selective", "sgdfnet")]

    # Step 2: Run models
    print(f"\n[2/4] Running {len(model_names)} models (max {args.max_runtime_minutes} min)...")
    max_sec = args.max_runtime_minutes * 60
    t_start = time.time()
    results = run_models(model_names, max_sec, out_dir)
    total_runtime = time.time() - t_start

    # Step 3: Evaluate
    print(f"\n[3/4] Evaluating results...")
    summary = generate_summary(results, model_names, total_runtime, out_dir)

    # Step 4: Report
    print(f"\n[4/4] Writing outputs...")

    # Summary JSON
    summary_path = out_dir / "p5_model_zoo_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print results table
    print(f"\n{'='*70}")
    print(f"  P5 Model Zoo Results")
    print(f"{'='*70}")
    print(f"  {'Model':<24} {'sMAPE':<10} {'Severe':<8} {'9-16 SMAPE':<12} {'Spike MAE':<12} {'Corr':<8}")
    print(f"  {'-'*70}")
    for name, result in sorted(results.items()):
        if result.success:
            m = result.metrics
            print(f"  {name:<24} {m.get('smape_floor50', '?'):<10} "
                  f"{m.get('severe_underestimate_count', '?'):<8} "
                  f"{str(m.get('9_16_smape_floor50', '?')):<12} "
                  f"{str(m.get('high_spike_mae', '?')):<12} "
                  f"{str(m.get('correlation_vs_lgbm', '?')):<8}")
        else:
            err = result.error[:50] if result.error else "not run"
            print(f"  {name:<24} {'—':<10} {'—':<8} {'—':<12} {'—':<12} —       ({err})")

    # LightGBM baseline
    lgbm_df, y_true_lgbm, y_pred_lgbm_arr = load_lightgbm_baseline()
    lgbm_metrics = compute_metrics(
        y_true_lgbm, y_pred_lgbm_arr,
        lgbm_df["hour_business"].values,
    )
    print(f"  {'─'*70}")
    print(f"  {'lightgbm (baseline)':<24} {lgbm_metrics.get('smape_floor50', '?'):<10} "
          f"{lgbm_metrics.get('severe_underestimate_count', '?'):<8} "
          f"{str(lgbm_metrics.get('9_16_smape_floor50', '?')):<12} "
          f"{str(lgbm_metrics.get('high_spike_mae', '?')):<12} —       ")

    print(f"\n  Summary: {summary_path}")
    print(f"  Predictions: {out_dir / 'predictions/'}")
    print(f"\n  Total runtime: {total_runtime:.0f}s ({total_runtime/60:.1f} min)")
    print(f"  Done.")


if __name__ == "__main__":
    main()
