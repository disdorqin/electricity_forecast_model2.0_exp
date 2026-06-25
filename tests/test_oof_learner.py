"""Unit tests for OOF learner system.

Uses synthetic data to test all components without requiring real electricity data.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from fusion.learners.oof_contracts import load_and_normalize_oof_table, load_forecast_long
from fusion.learners.data_checks import compute_coverage_report, get_eligible_models
from fusion.learners.metrics import compute_all_metrics, smape_floor50
from fusion.learners.static_convex import fit_static_convex
from fusion.learners.bgew import fit_bgew
from fusion.learners.candidate_selector import select_best_candidate
from fusion.learners.roel import run_roel_bgew_fallback
from fusion.learners.apply_learner import apply_learner_to_forecast


def create_synthetic_oof_data(
    n_days=30,
    models=("model_a", "model_b", "model_c"),
    tasks=("dayahead",),
    periods=("1_8", "9_16", "17_24"),
    noise_level=10.0,
):
    """Create synthetic OOF long-table for testing."""
    rows = []
    base_date = pd.Timestamp("2026-08-01")

    for task in tasks:
        for period in periods:
            for day_offset in range(n_days):
                target_day = base_date + pd.Timedelta(days=day_offset)
                for hour in range(1, 25):
                    ds = target_day + pd.Timedelta(hours=hour - 1)
                    if hour == 24:
                        ds = target_day + pd.Timedelta(hours=23)

                    # True value varies by period
                    if period == "1_8":
                        y_true = 100.0 + 20.0 * np.sin(hour / 24 * 2 * np.pi)
                    elif period == "9_16":
                        y_true = 200.0 + 50.0 * np.sin(hour / 24 * 2 * np.pi)
                    else:  # 17_24
                        y_true = 150.0 + 30.0 * np.sin(hour / 24 * 2 * np.pi)

                    for model in models:
                        # Each model has different bias
                        if model == "model_a":
                            bias = 5.0
                        elif model == "model_b":
                            bias = -3.0
                        else:  # model_c
                            bias = 0.0

                        y_pred = y_true + bias + np.random.randn() * noise_level

                        rows.append({
                            "task": task,
                            "model_name": model,
                            "fold_id": 0,
                            "train_start": "2025-01-01",
                            "train_end": "2026-07-31",
                            "test_start": target_day.strftime("%Y-%m-%d"),
                            "test_end": target_day.strftime("%Y-%m-%d"),
                            "target_day": target_day.strftime("%Y-%m-%d"),
                            "business_day": target_day.strftime("%Y-%m-%d"),
                            "ds": ds.strftime("%Y-%m-%d %H:%M:%S"),
                            "period": period,
                            "hour_business": hour,
                            "y_true": y_true,
                            "y_pred": y_pred,
                            "source": "synthetic",
                            "run_mode": "test",
                            "created_at": pd.Timestamp.now().isoformat(),
                        })

    return pd.DataFrame(rows)


def create_synthetic_forecast_data(
    target_day="2026-09-01",
    models=("model_a", "model_b", "model_c"),
    tasks=("dayahead",),
):
    """Create synthetic forecast long-table."""
    rows = []
    target_dt = pd.Timestamp(target_day)

    for task in tasks:
        for hour in range(1, 25):
            ds = target_dt + pd.Timedelta(hours=hour - 1)
            if hour <= 8:
                period = "1_8"
                y_true = 100.0
            elif hour <= 16:
                period = "9_16"
                y_true = 200.0
            else:
                period = "17_24"
                y_true = 150.0

            for model in models:
                if model == "model_a":
                    bias = 5.0
                elif model == "model_b":
                    bias = -3.0
                else:
                    bias = 0.0

                y_pred = y_true + bias + np.random.randn() * 5.0

                rows.append({
                    "task": task,
                    "model_name": model,
                    "target_day": target_day,
                    "ds": ds.strftime("%Y-%m-%d %H:%M:%S"),
                    "period": period,
                    "hour_business": hour,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "business_day": target_day,
                })

    return pd.DataFrame(rows)


# Test 1: normalize_oof_table fills business_day/hour_business/period
def test_normalize_oof_table():
    """Test that normalize_oof_table auto-fills missing fields."""
    # Create minimal data without business_day, hour_business, period
    df = pd.DataFrame({
        "task": ["dayahead", "dayahead"],
        "model_name": ["model_a", "model_a"],
        "target_day": ["2026-08-01", "2026-08-01"],
        "ds": ["2026-08-01 00:00:00", "2026-08-01 12:00:00"],
        "y_true": [100.0, 200.0],
        "y_pred": [105.0, 195.0],
    })

    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    try:
        result = load_and_normalize_oof_table(temp_path)
        assert "business_day" in result.columns
        assert "hour_business" in result.columns
        assert "period" in result.columns
        assert result["hour_business"].iloc[0] == 24  # 00:00 -> 24
        assert result["hour_business"].iloc[1] == 12
        assert result["period"].iloc[0] == "17_24"  # hour 24 -> 17_24
        assert result["period"].iloc[1] == "9_16"  # hour 12 -> 9_16
    finally:
        Path(temp_path).unlink()


# Test 2: coverage_report identifies low-coverage models
def test_coverage_report():
    """Test that coverage_report flags models below threshold."""
    oof_df = create_synthetic_oof_data(n_days=10, models=("model_a", "model_b", "model_c"))

    # Remove MOST model_c data for 9_16 (keep only 1 row to make it low coverage)
    mask = (oof_df["model_name"] == "model_c") & (oof_df["period"] == "9_16")
    model_c_916 = oof_df[mask]
    # Keep only first row
    keep_idx = model_c_916.index[:1]
    remove_idx = model_c_916.index[1:]
    oof_df_partial = oof_df.drop(remove_idx)

    report = compute_coverage_report(oof_df_partial, coverage_threshold=0.95)

    # model_c in 9_16 should be ineligible
    model_c_916_report = report[
        (report["model_name"] == "model_c") & (report["period"] == "9_16")
    ]
    assert len(model_c_916_report) > 0
    assert not model_c_916_report.iloc[0]["eligible"]
    assert model_c_916_report.iloc[0]["coverage"] < 0.95


# Test 3: static_convex weights are non-negative and sum to 1
def test_static_convex_weights():
    """Test that static_convex produces valid weights."""
    oof_df = create_synthetic_oof_data(n_days=20, models=("model_a", "model_b"))

    result = fit_static_convex(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
    )

    assert all(w >= 0.0 for w in result.weights.values())
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# Test 4: bgew weights are non-negative and sum to 1
def test_bgew_weights():
    """Test that BGEW produces valid weights."""
    oof_df = create_synthetic_oof_data(n_days=20, models=("model_a", "model_b"))

    result = fit_bgew(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
    )

    assert all(w >= 0.0 for w in result.weights.values())
    assert abs(sum(result.weights.values()) - 1.0) < 1e-6


# Test 5: BGEW gate gives higher weight to recent samples
def test_bgew_gate():
    """Test that BGEW gate decreases with age."""
    oof_df = create_synthetic_oof_data(n_days=30, models=("model_a", "model_b"))

    result = fit_bgew(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
        tau=30.0,
    )

    # Check trace has gate values
    assert len(result.trace) > 0
    gates = [t["gate_time"] for t in result.trace]
    # Gates should be <= 1.0
    assert all(g <= 1.0 for g in gates)
    # Most recent day should have gate close to 1.0
    assert max(gates) > 0.9


# Test 6: High-loss model gets lower weight in BGEW
def test_bgew_high_loss():
    """Test that BGEW reduces weight for high-loss models."""
    # Create data where model_b is much worse
    oof_df = create_synthetic_oof_data(n_days=20, models=("model_a", "model_b"), noise_level=5.0)

    # Make model_b predictions much worse
    mask = oof_df["model_name"] == "model_b"
    oof_df.loc[mask, "y_pred"] = oof_df.loc[mask, "y_true"] + 100.0  # Large bias

    result = fit_bgew(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
    )

    # model_a should have higher weight than model_b
    assert result.weights["model_a"] > result.weights["model_b"]


# Test 7: candidate_selector MUST choose one-hot when single model is clearly best
def test_candidate_selector_one_hot():
    """When one model is clearly better, selector must pick selected_mode == 'single_model'."""
    np.random.seed(123)
    oof_df = create_synthetic_oof_data(n_days=60, models=("model_a", "model_b"), noise_level=3.0)

    # Make model_b have a massive systematic bias — far worse than model_a
    mask = oof_df["model_name"] == "model_b"
    oof_df.loc[mask, "y_pred"] = oof_df.loc[mask, "y_true"] + 80.0  # Huge bias

    best, metrics = select_best_candidate(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
    )

    # Must select single_model with model_a
    assert best.selected_mode == "single_model", (
        f"Expected single_model but got {best.selected_mode}; "
        f"metrics:\n{metrics.to_string()}"
    )
    assert best.selected_model == "model_a"


# Test 8: candidate_selector chooses fusion when fusion is better
def test_candidate_selector_fusion():
    """Test that candidate_selector can select fusion."""
    # Create data where models have complementary errors
    np.random.seed(42)
    oof_df = create_synthetic_oof_data(n_days=30, models=("model_a", "model_b"), noise_level=20.0)

    # Make errors anti-correlated
    mask_a = oof_df["model_name"] == "model_a"
    mask_b = oof_df["model_name"] == "model_b"
    oof_df.loc[mask_a, "y_pred"] = oof_df.loc[mask_a, "y_true"] + 15.0
    oof_df.loc[mask_b, "y_pred"] = oof_df.loc[mask_b, "y_true"] - 15.0

    best, metrics = select_best_candidate(
        oof_df,
        task="dayahead",
        period="1_8",
        eligible_models=["model_a", "model_b"],
    )

    # Fusion should be better than either single model
    # (equal weight would give ~0 error)
    assert best.selected_mode in ["equal_weight", "static_convex", "bgew"]


# Test 9: apply_learner handles missing selected_model with fallback
def test_apply_learner_fallback():
    """Test that apply_learner falls back when selected model is missing."""
    oof_df = create_synthetic_oof_data(n_days=20, models=("model_a", "model_b", "model_c"))

    # Train learner
    output = run_roel_bgew_fallback(oof_df, coverage_threshold=0.5)

    # Create forecast without model_a
    forecast_df = create_synthetic_forecast_data(
        target_day="2026-09-01",
        models=("model_b", "model_c"),  # model_a missing
    )

    # Apply learner
    result = apply_learner_to_forecast(
        forecast_df,
        output.routing_table,
        output.weights,
    )

    # Should still produce output (with fallback)
    assert len(result) > 0
    # Should have 24 rows for dayahead
    assert len(result[result["task"] == "dayahead"]) == 24


# Test 10: apply_learner output has 24 rows per target_day
def test_apply_learner_24_rows():
    """Test that apply_learner produces exactly 24 rows per target_day."""
    oof_df = create_synthetic_oof_data(n_days=20, models=("model_a", "model_b"))

    # Train learner
    output = run_roel_bgew_fallback(oof_df, coverage_threshold=0.5)

    # Create forecast
    forecast_df = create_synthetic_forecast_data(
        target_day="2026-09-01",
        models=("model_a", "model_b"),
    )

    # Apply learner
    result = apply_learner_to_forecast(
        forecast_df,
        output.routing_table,
        output.weights,
    )

    # Should have exactly 24 rows per (task, target_day)
    counts = result.groupby(["task", "target_day"]).size()
    assert all(counts == 24)

    # business_day must never be None
    assert result["business_day"].notna().all(), (
        f"Found {result['business_day'].isna().sum()} rows with None business_day"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
