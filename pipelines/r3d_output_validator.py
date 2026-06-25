"""R3D-Tap-GEF Output Validator.

Validates the outputs of a production pipeline run:
- validation_tap_long_table.csv covers D-30 ~ D-1
- tap_fold_id has exactly 10 values
- each fold covers 3 days
- horizon_day is only 1, 2, or 3
- age_block ranges 0..9
- final files have 24 rows
- weights sum ≈ 1 per task+period
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _check(condition: bool, msg: str) -> tuple[bool, str]:
    return (True, f"PASS: {msg}") if condition else (False, f"FAIL: {msg}")


def validate_tap_long_table(
    csv_path: Path,
    predict_date: str,
    *,
    expected_folds: int = 10,
    expected_block_days: int = 3,
) -> list[tuple[bool, str]]:
    """Validate validation_tap_long_table.csv."""
    results = []

    if not csv_path.exists():
        results.append(_check(False, f"tap_long_table not found: {csv_path}"))
        return results

    df = pd.read_csv(csv_path)

    # 1. Required columns
    required_cols = [
        "tap_fold_id", "age_block", "horizon_day", "target_day",
        "ds", "model_name", "y_pred", "y_true", "period", "task",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    results.append(_check(len(missing) == 0, f"required columns present (missing: {missing})" if missing else "all required columns present"))

    if missing:
        return results  # cannot validate further

    # 2. Date coverage: D-30 ~ D-1
    d = pd.Timestamp(predict_date)
    expected_start = (d - timedelta(days=30)).strftime("%Y-%m-%d")
    expected_end = (d - timedelta(days=1)).strftime("%Y-%m-%d")
    actual_dates = sorted(df["target_day"].unique())
    actual_start = actual_dates[0] if actual_dates else "?"
    actual_end = actual_dates[-1] if actual_dates else "?"
    results.append(_check(
        actual_start == expected_start and actual_end == expected_end,
        f"date coverage D-30~D-1 (expected {expected_start}~{expected_end}, got {actual_start}~{actual_end})"
    ))

    # 3. tap_fold_id count
    fold_ids = sorted(df["tap_fold_id"].unique())
    results.append(_check(
        len(fold_ids) == expected_folds,
        f"tap_fold_id has {expected_folds} values (got {len(fold_ids)}: {fold_ids})"
    ))

    # 4. Each fold covers exactly 3 days
    for fid in fold_ids:
        fold_dates = df[df["tap_fold_id"] == fid]["target_day"].nunique()
        results.append(_check(
            fold_dates == expected_block_days,
            f"fold {fid} covers {expected_block_days} days (got {fold_dates})"
        ))

    # 5. horizon_day is only 1, 2, or 3
    horizons = sorted(df["horizon_day"].unique())
    results.append(_check(
        horizons == [1, 2, 3],
        f"horizon_day is [1,2,3] (got {horizons})"
    ))

    # 6. age_block is 0..9
    age_blocks = sorted(df["age_block"].unique())
    results.append(_check(
        age_blocks == list(range(expected_folds)),
        f"age_block is 0..{expected_folds-1} (got {age_blocks})"
    ))

    return results


def validate_final_csv(
    csv_path: Path,
    *,
    expected_rows: int = 24,
) -> list[tuple[bool, str]]:
    """Validate a final predictions CSV has 24 rows and required columns."""
    results = []

    if not csv_path.exists():
        results.append(_check(False, f"final CSV not found: {csv_path}"))
        return results

    df = pd.read_csv(csv_path)
    results.append(_check(
        len(df) == expected_rows,
        f"final CSV {csv_path.name} has {expected_rows} rows (got {len(df)})"
    ))

    # Check for period and prediction columns
    for col in ["period", "hour_business"]:
        results.append(_check(col in df.columns, f"column '{col}' present"))

    return results


def validate_weights_csv(
    csv_path: Path,
) -> list[tuple[bool, str]]:
    """Validate weights.csv: each task+period group sums ≈ 1."""
    results = []

    if not csv_path.exists():
        results.append(_check(False, f"weights CSV not found: {csv_path}"))
        return results

    df = pd.read_csv(csv_path)
    for col in ["task", "period", "model_name", "weight"]:
        if col not in df.columns:
            results.append(_check(False, f"missing column '{col}' in weights.csv"))
            return results

    groups = df.groupby(["task", "period"])
    for (task, period), grp in groups:
        total = grp["weight"].sum()
        results.append(_check(
            abs(total - 1.0) < 0.02,
            f"weights sum ≈ 1 for {task}/{period} (got {total:.4f})"
        ))

    return results


def validate_run_manifest(
    manifest_path: Path,
) -> list[tuple[bool, str]]:
    """Validate run_manifest.json exists and has required steps."""
    results = []

    if not manifest_path.exists():
        results.append(_check(False, f"run_manifest.json not found: {manifest_path}"))
        return results

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    results.append(_check("predict_date" in manifest, "predict_date in manifest"))
    results.append(_check("steps" in manifest, "steps in manifest"))

    if "steps" in manifest:
        for step_name in ["validation_tap", "real_forecast", "learner", "fusion", "final_outputs"]:
            results.append(_check(
                step_name in manifest["steps"],
                f"step '{step_name}' recorded in manifest"
            ))

    return results


def run_all_validations(
    output_dir: Path,
    predict_date: str | None = None,
    *,
    tasks: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Run full validation suite on a production pipeline output.

    Parameters
    ----------
    output_dir : Path
        The date directory (outputs/{date}).
    predict_date : str, optional
        Target date string. If None, inferred from output_dir name.
    tasks : list[str], optional
        Which targets to validate (default: ["dayahead", "realtime"]).

    Returns (all_passed, list_of_messages).
    """
    if predict_date is None:
        predict_date = output_dir.name

    if tasks is None:
        tasks = ["dayahead", "realtime"]

    all_results = []

    # Manifest
    all_results.extend(validate_run_manifest(output_dir / "run_manifest.json"))

    for target in tasks:
        target_dir = output_dir / target

        # Validation tap
        tap_csv = target_dir / "validation" / "validation_tap_long_table.csv"
        all_results.extend(validate_tap_long_table(tap_csv, predict_date))

        # Final predictions
        final_csv = target_dir / "final" / f"{target}_final_predictions.csv"
        all_results.extend(validate_final_csv(final_csv))

        # Weights
        weights_csv = target_dir / "fused" / "weights.csv"
        all_results.extend(validate_weights_csv(weights_csv))

    passed = all(ok for ok, _ in all_results)
    messages = [msg for _, msg in all_results]

    if passed:
        logger.info("All %d validation checks passed.", len(all_results))
    else:
        fails = sum(1 for ok, _ in all_results if not ok)
        logger.warning("%d/%d validation checks failed.", fails, len(all_results))

    return passed, messages
