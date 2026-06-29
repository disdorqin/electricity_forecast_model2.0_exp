"""OOF long-table loading and normalization.

Reads rolling-origin OOF prediction pools and standardizes them for learner training.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fusion.contracts import VALID_PERIODS, VALID_TASKS, infer_period, normalize_task

logger = logging.getLogger(__name__)

# Minimum required columns for OOF learner (before auto-fill)
OOF_MINIMAL_COLUMNS = [
    "task", "model_name", "target_day", "ds", "y_true", "y_pred",
]

# Full OOF long-table columns (matches rolling_oof/contracts.py)
OOF_FULL_COLUMNS = [
    "task", "model_name", "fold_id",
    "train_start", "train_end", "test_start", "test_end",
    "target_day", "business_day", "ds", "period", "hour_business",
    "y_true", "y_pred", "source", "run_mode", "created_at",
]


def load_and_normalize_oof_table(oof_path: str | Path) -> pd.DataFrame:
    """Load OOF long-table CSV and normalize to standard schema.

    Automatically fills missing business_day, hour_business, period.

    Parameters
    ----------
    oof_path : str or Path
        Path to oof_long_table.csv

    Returns
    -------
    pd.DataFrame
        Normalized OOF long-table with all OOF_FULL_COLUMNS present.
    """
    oof_path = Path(oof_path)
    if not oof_path.exists():
        raise FileNotFoundError(f"OOF long-table not found: {oof_path}")

    df = pd.read_csv(oof_path)
    logger.info(
        "Loaded OOF table: %d rows, %d models",
        len(df),
        df["model_name"].nunique() if "model_name" in df.columns else 0,
    )

    # Check minimum required columns
    missing = [c for c in OOF_MINIMAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OOF table missing required columns: {missing}")

    out = df.copy()

    # Normalize task labels
    out["task"] = out["task"].map(normalize_task)
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["target_day"] = pd.to_datetime(out["target_day"]).dt.strftime("%Y-%m-%d")
    out["ds"] = pd.to_datetime(out["ds"])

    # Auto-fill hour_business
    if "hour_business" not in out.columns or out["hour_business"].isna().all():
        out["hour_business"] = out["ds"].apply(
            lambda t: 24 if pd.notna(t) and t.hour == 0 else (t.hour if pd.notna(t) else pd.NA)
        )
        logger.info("Auto-filled hour_business from ds")
    out["hour_business"] = out["hour_business"].astype("Int64")

    # Auto-fill business_day
    if "business_day" not in out.columns or out["business_day"].isna().all():
        out["business_day"] = out["ds"].apply(
            lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.notna(t) and t.hour == 0
            else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
        )
        logger.info("Auto-filled business_day from ds")

    # Auto-fill period
    if "period" not in out.columns or out["period"].isna().all():
        out["period"] = out["hour_business"].apply(
            lambda h: infer_period(h) if pd.notna(h) else "unknown"
        )
        logger.info("Auto-filled period from hour_business")
    else:
        # Fill only NaN cells
        mask = out["period"].isna() | (out["period"].astype(str).str.strip() == "")
        if mask.any():
            out.loc[mask, "period"] = out.loc[mask, "hour_business"].apply(
                lambda h: infer_period(h) if pd.notna(h) else "unknown"
            )
    out["period"] = out["period"].astype(str).str.strip()

    # Numeric coercion
    out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")

    # Validate
    bad_tasks = sorted(set(out["task"].dropna()) - VALID_TASKS)
    bad_periods = sorted(set(out["period"].dropna()) - VALID_PERIODS)
    if bad_tasks:
        raise ValueError(f"Unsupported task labels: {bad_tasks}")
    if bad_periods:
        raise ValueError(f"Unsupported period labels: {bad_periods}")

    # Drop rows with NaN in critical columns
    critical = ["task", "model_name", "target_day", "ds", "period", "hour_business", "y_true", "y_pred"]
    before = len(out)
    out = out.dropna(subset=critical)
    dropped = before - len(out)
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN in critical columns", dropped)

    # Ensure all OOF_FULL_COLUMNS present
    for col in OOF_FULL_COLUMNS:
        if col not in out.columns:
            out[col] = None

    logger.info(
        "Normalized OOF: %d rows, %d tasks, %d models, periods=%s",
        len(out), out["task"].nunique(), out["model_name"].nunique(),
        sorted(out["period"].unique()),
    )
    return out


def load_forecast_long(forecast_path: str | Path) -> pd.DataFrame:
    """Load final forecast long-table (escort or daily_run output).

    y_true is optional for forecast tables.
    """
    forecast_path = Path(forecast_path)
    if not forecast_path.exists():
        raise FileNotFoundError(f"Forecast long-table not found: {forecast_path}")

    df = pd.read_csv(forecast_path)
    logger.info("Loaded forecast table: %d rows", len(df))

    required = ["task", "model_name", "target_day", "ds", "period", "hour_business", "y_pred"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Forecast table missing columns: {missing}")

    out = df.copy()
    out["task"] = out["task"].map(normalize_task)
    out["model_name"] = out["model_name"].astype(str).str.strip()
    out["target_day"] = pd.to_datetime(out["target_day"]).dt.strftime("%Y-%m-%d")
    out["ds"] = pd.to_datetime(out["ds"])

    if "hour_business" not in out.columns or out["hour_business"].isna().all():
        out["hour_business"] = out["ds"].apply(
            lambda t: 24 if pd.notna(t) and t.hour == 0 else (t.hour if pd.notna(t) else pd.NA)
        )
    out["hour_business"] = out["hour_business"].astype("Int64")

    if "business_day" not in out.columns or out["business_day"].isna().all():
        out["business_day"] = out["ds"].apply(
            lambda t: (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if pd.notna(t) and t.hour == 0
            else (t.strftime("%Y-%m-%d") if pd.notna(t) else None)
        )

    if "period" not in out.columns or out["period"].isna().all():
        out["period"] = out["hour_business"].apply(
            lambda h: infer_period(h) if pd.notna(h) else "unknown"
        )
    out["period"] = out["period"].astype(str).str.strip()

    out["y_pred"] = pd.to_numeric(out["y_pred"], errors="coerce")
    if "y_true" in out.columns:
        out["y_true"] = pd.to_numeric(out["y_true"], errors="coerce")

    logger.info("Normalized forecast: %d rows, %d models", len(out), out["model_name"].nunique())
    return out
