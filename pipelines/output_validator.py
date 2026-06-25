"""Output validation for production pipeline.

10 checks to ensure correctness of each pipeline stage output.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

VALID_PERIODS = {"1_8", "9_16", "17_24"}


def validate_forecast_csv(
    csv_path: str | Path,
    *,
    label: str = "",
) -> tuple[bool, list[str]]:
    """Validate a forecast/fusion long-table CSV.

    Checks 1-6 from the spec.

    Returns
    -------
    passed : bool
        True if all checks pass.
    errors : list[str]
        List of error messages (empty if passed).
    """
    csv_path = Path(csv_path)
    errors: list[str] = []
    prefix = f"[{label}] " if label else ""

    if not csv_path.exists():
        return False, [f"{prefix}File not found: {csv_path}"]

    df = pd.read_csv(csv_path)
    if df.empty:
        return False, [f"{prefix}File is empty: {csv_path}"]

    # Check 1: each target_day exactly 24 rows
    if "target_day" in df.columns:
        counts = df.groupby("target_day").size()
        bad = counts[counts != 24]
        if not bad.empty:
            errors.append(f"{prefix}target_day with !=24 rows: {bad.to_dict()}")

    # Check 2: hour_business covers 1..24
    if "hour_business" in df.columns:
        hours = set(df["hour_business"].dropna().astype(int).unique())
        expected = set(range(1, 25))
        missing = expected - hours
        if missing:
            errors.append(f"{prefix}hour_business missing: {sorted(missing)}")

    # Check 3: period only has valid values
    if "period" in df.columns:
        periods = set(df["period"].dropna().astype(str).str.strip().unique())
        bad_periods = periods - VALID_PERIODS
        if bad_periods:
            errors.append(f"{prefix}Invalid period labels: {bad_periods}")

    # Check 4: each period has 8 rows per target_day
    if "period" in df.columns and "target_day" in df.columns:
        period_counts = df.groupby(["target_day", "period"]).size()
        bad_period_counts = period_counts[period_counts != 8]
        if not bad_period_counts.empty:
            errors.append(
                f"{prefix}Period with !=8 rows per day: "
                f"{bad_period_counts.to_dict()}"
            )

    # Check 5: business_day not null
    if "business_day" in df.columns:
        null_bd = df["business_day"].isna().sum()
        if null_bd > 0:
            errors.append(f"{prefix}{null_bd} rows with null business_day")

    # Check 6: y_pred not null
    pred_col = None
    for c in ["y_pred", "y_fused", "y_pred_learner"]:
        if c in df.columns:
            pred_col = c
            break
    if pred_col:
        null_pred = df[pred_col].isna().sum()
        if null_pred > 0:
            errors.append(f"{prefix}{null_pred} rows with null {pred_col}")

    passed = len(errors) == 0
    return passed, errors


def validate_model_alignment(
    forecast_dir: str | Path,
    models: list[str],
    *,
    task: str = "",
) -> tuple[bool, list[str]]:
    """Check 7: all model forecasts align on the same ds values."""
    forecast_dir = Path(forecast_dir)
    errors: list[str] = []
    prefix = f"[{task}/forecast] "

    ds_sets = {}
    for model in models:
        pred_file = forecast_dir / model / "forecast_predictions.csv"
        if pred_file.exists():
            df = pd.read_csv(pred_file)
            if "ds" in df.columns:
                ds_sets[model] = set(df["ds"].unique())

    if len(ds_sets) > 1:
        ref_model = next(iter(ds_sets))
        ref_ds = ds_sets[ref_model]
        for model, ds_set in ds_sets.items():
            if ds_set != ref_ds:
                only_in_ref = ref_ds - ds_set
                only_in_model = ds_set - ref_ds
                if only_in_ref:
                    errors.append(
                        f"{prefix}{model} missing {len(only_in_ref)} ds "
                        f"vs {ref_model}"
                    )
                if only_in_model:
                    errors.append(
                        f"{prefix}{model} has {len(only_in_model)} extra ds "
                        f"vs {ref_model}"
                    )

    passed = len(errors) == 0
    return passed, errors


def validate_weights_csv(
    csv_path: str | Path,
    *,
    label: str = "",
) -> tuple[bool, list[str]]:
    """Check 8: weights non-negative and sum≈1 per task+period."""
    csv_path = Path(csv_path)
    errors: list[str] = []
    prefix = f"[{label}] " if label else ""

    if not csv_path.exists():
        return False, [f"{prefix}File not found: {csv_path}"]

    df = pd.read_csv(csv_path)
    if df.empty:
        return False, [f"{prefix}File is empty: {csv_path}"]

    # Check non-negative
    if "weight" in df.columns:
        neg = (df["weight"] < -1e-9).sum()
        if neg > 0:
            errors.append(f"{prefix}{neg} negative weights found")

    # Check sum≈1 per task+period
    group_cols = []
    if "task" in df.columns:
        group_cols.append("task")
    if "period" in df.columns:
        group_cols.append("period")

    if group_cols and "weight" in df.columns:
        sums = df.groupby(group_cols)["weight"].sum()
        bad_sums = sums[~((sums - 1.0).abs() < 0.02)]
        if not bad_sums.empty:
            errors.append(
                f"{prefix}Weight sums not ≈1: {bad_sums.to_dict()}"
            )

    passed = len(errors) == 0
    return passed, errors


def validate_routing_table(
    csv_path: str | Path,
    *,
    tasks: list[str] | None = None,
    periods: list[str] | None = None,
    label: str = "",
) -> tuple[bool, list[str]]:
    """Check 9: routing_table has exactly one entry per task+period."""
    csv_path = Path(csv_path)
    errors: list[str] = []
    prefix = f"[{label}] " if label else ""

    if not csv_path.exists():
        return False, [f"{prefix}File not found: {csv_path}"]

    df = pd.read_csv(csv_path)
    if df.empty:
        return False, [f"{prefix}File is empty: {csv_path}"]

    if "task" in df.columns and "period" in df.columns:
        counts = df.groupby(["task", "period"]).size()
        dupes = counts[counts > 1]
        if not dupes.empty:
            errors.append(f"{prefix}Duplicate routing entries: {dupes.to_dict()}")

        if tasks and periods:
            expected = {(t, p) for t in tasks for p in periods}
            actual = set(df.groupby(["task", "period"]).groups.keys())
            missing = expected - actual
            if missing:
                errors.append(f"{prefix}Missing routing entries: {missing}")

    passed = len(errors) == 0
    return passed, errors


def validate_classifier_output(
    fusion_csv: str | Path,
    classifier_csv: str | Path,
    *,
    label: str = "",
) -> tuple[bool, list[str]]:
    """Check 10: classifier output row count matches fusion input."""
    errors: list[str] = []
    prefix = f"[{label}] " if label else ""

    fusion_path = Path(fusion_csv)
    clf_path = Path(classifier_csv)

    if not fusion_path.exists():
        errors.append(f"{prefix}Fusion input not found: {fusion_path}")
        return False, errors

    if not clf_path.exists():
        errors.append(f"{prefix}Classifier output not found: {clf_path}")
        return False, errors

    fusion_df = pd.read_csv(fusion_path)
    clf_df = pd.read_csv(clf_path)

    if len(fusion_df) != len(clf_df):
        errors.append(
            f"{prefix}Row count mismatch: fusion={len(fusion_df)}, "
            f"classifier={len(clf_df)}"
        )

    passed = len(errors) == 0
    return passed, errors


def run_all_validations(
    date_dir: str | Path,
    *,
    tasks: list[str] | None = None,
    models_by_task: dict[str, list[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Run all applicable validations for a date directory.

    Returns
    -------
    all_passed : bool
    all_errors : list[str]
    """
    date_dir = Path(date_dir)
    all_errors: list[str] = []

    if tasks is None:
        tasks = ["dayahead", "realtime"]
    if models_by_task is None:
        models_by_task = {
            "dayahead": ["lightgbm", "timesfm", "timemixer"],
            "realtime": ["sgdfnet", "timemixer", "rt916", "timesfm"],
        }

    for task in tasks:
        task_dir = date_dir / task

        # Validate model forecasts alignment
        forecast_dir = task_dir / "02_model_forecasts"
        if forecast_dir.exists():
            passed, errs = validate_model_alignment(
                forecast_dir, models_by_task.get(task, []), task=task,
            )
            all_errors.extend(errs)

        # Validate fusion output
        fused_csv = task_dir / "04_fusion" / "fused_predictions.csv"
        if fused_csv.exists():
            passed, errs = validate_forecast_csv(fused_csv, label=f"{task}/fusion")
            all_errors.extend(errs)

        # Validate learner weights
        weights_csv = task_dir / "03_learner" / "weights.csv"
        if weights_csv.exists():
            passed, errs = validate_weights_csv(weights_csv, label=f"{task}/weights")
            all_errors.extend(errs)

        # Validate routing table
        routing_csv = task_dir / "03_learner" / "routing_table.csv"
        if routing_csv.exists():
            passed, errs = validate_routing_table(
                routing_csv, tasks=[task], periods=["1_8", "9_16", "17_24"],
                label=f"{task}/routing",
            )
            all_errors.extend(errs)

        # Validate classifier output (realtime only)
        if task == "realtime":
            clf_csv = task_dir / "05_classifier" / "fused_predictions_corrected.csv"
            if clf_csv.exists():
                passed, errs = validate_classifier_output(
                    fused_csv, clf_csv, label=f"{task}/classifier",
                )
                all_errors.extend(errs)

    all_passed = len(all_errors) == 0
    return all_passed, all_errors
