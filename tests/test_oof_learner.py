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


# Test 11: meta-validation uses TRUE fitted weights, not equal_weight proxy
def test_meta_validation_uses_true_weights():
    """Prove that eval stage uses static_convex true weights, not equal_weight proxy.

    Design: model_a is good (small noise), model_b is bad (large bias).
    static_convex will learn to give model_a ~0.8+ weight on fit months.
    equal_weight gives 0.5/0.5 which is much worse.
    We verify the eval_metric for static_convex matches its TRUE weights,
    NOT the equal_weight proxy.
    """
    np.random.seed(42)
    n_days_per_month = 15
    models = ("model_a", "model_b")
    tasks = ("dayahead",)
    periods = ("1_8",)

    # Build 5 months of data (Aug-Dec)
    rows = []
    base_date = pd.Timestamp("2026-08-01")
    total_days = n_days_per_month * 5  # 75 days

    for day_offset in range(total_days):
        target_day = base_date + pd.Timedelta(days=day_offset)
        for hour in range(1, 25):
            ds = target_day + pd.Timedelta(hours=hour - 1)
            if hour == 24:
                ds = target_day + pd.Timedelta(hours=23)
            y_true = 200.0 + 30.0 * np.sin(hour / 24 * 2 * np.pi)

            # model_a: small noise (good)
            y_pred_a = y_true + np.random.randn() * 5.0
            # model_b: large systematic bias (bad)
            y_pred_b = y_true + 40.0 + np.random.randn() * 10.0

            for model, y_pred in [("model_a", y_pred_a), ("model_b", y_pred_b)]:
                rows.append({
                    "task": "dayahead",
                    "model_name": model,
                    "fold_id": 0,
                    "train_start": "2025-01-01",
                    "train_end": "2026-07-31",
                    "test_start": target_day.strftime("%Y-%m-%d"),
                    "test_end": target_day.strftime("%Y-%m-%d"),
                    "target_day": target_day.strftime("%Y-%m-%d"),
                    "business_day": target_day.strftime("%Y-%m-%d"),
                    "ds": ds.strftime("%Y-%m-%d %H:%M:%S"),
                    "period": "1_8",
                    "hour_business": hour,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "source": "synthetic",
                    "run_mode": "test",
                    "created_at": pd.Timestamp.now().isoformat(),
                })

    oof_df = pd.DataFrame(rows)

    # Run ROEL with last_block_holdout
    output = run_roel_bgew_fallback(oof_df, coverage_threshold=0.5)

    # --- Verify candidate_metrics has required columns ---
    cm = output.candidate_metrics
    assert "fit_metric" in cm.columns, "candidate_metrics missing fit_metric"
    assert "eval_metric" in cm.columns, "candidate_metrics missing eval_metric"
    assert "selected_by" in cm.columns, "candidate_metrics missing selected_by"
    assert "final_refit_metric" in cm.columns, "candidate_metrics missing final_refit_metric"

    # All entries should be selected_by=eval_holdout (we have 5 months)
    assert all(cm["selected_by"] == "eval_holdout")

    # --- Verify static_convex eval_metric uses TRUE weights ---
    sc_row = cm[cm["candidate_name"] == "static_convex"]
    eq_row = cm[cm["candidate_name"] == "equal_weight"]
    assert len(sc_row) > 0, "static_convex not in candidate_metrics"
    assert len(eq_row) > 0, "equal_weight not in candidate_metrics"

    sc_eval_metric = sc_row.iloc[0]["eval_metric"]
    eq_eval_metric = eq_row.iloc[0]["eval_metric"]

    # static_convex should be significantly better (lower) than equal_weight
    assert sc_eval_metric < eq_eval_metric, (
        f"static_convex eval_metric ({sc_eval_metric:.4f}) should be < "
        f"equal_weight eval_metric ({eq_eval_metric:.4f})"
    )

    # --- Manually verify: compute static_convex eval metric from its TRUE fit weights ---
    from fusion.learners.candidate_selector import fit_all_candidates, _evaluate_candidate

    # Reproduce the fit on the same fit months (Aug-Nov, i.e. first 4 months)
    sub = oof_df[(oof_df["task"] == "dayahead") & (oof_df["period"] == "1_8")]
    months = sorted(sub["target_day"].apply(lambda x: x[:7]).unique())
    fit_months = months[:-1]
    eval_months = months[-1:]
    fit_mask = sub["target_day"].apply(lambda x: x[:7]).isin(fit_months)
    eval_mask = sub["target_day"].apply(lambda x: x[:7]).isin(eval_months)
    fit_df = sub[fit_mask]
    eval_df = sub[eval_mask]

    fitted = fit_all_candidates(fit_df, "dayahead", "1_8", ["model_a", "model_b"])
    sc_fit = [f for f in fitted if f[0] == "static_convex"][0]
    sc_true_weights = sc_fit[3]  # The true fitted weights

    # Verify static_convex gives model_a dominant weight (not 50/50)
    assert sc_true_weights["model_a"] > 0.7, (
        f"static_convex should give model_a >0.7 weight, got {sc_true_weights['model_a']:.3f}"
    )

    # Manually compute eval metric with TRUE weights
    manual_eval_score, _ = _evaluate_candidate(
        eval_df, "dayahead", "1_8", sc_true_weights, metric_name="sMAPE_floor50"
    )

    # The eval_metric in candidate_metrics must match the manual computation
    assert abs(sc_eval_metric - manual_eval_score) < 1e-6, (
        f"static_convex eval_metric ({sc_eval_metric:.6f}) != "
        f"manual eval with true weights ({manual_eval_score:.6f}). "
        f"Eval is NOT using true fitted weights!"
    )

    # Also verify it does NOT match equal_weight eval
    eq_weights = {"model_a": 0.5, "model_b": 0.5}
    eq_manual_score, _ = _evaluate_candidate(
        eval_df, "dayahead", "1_8", eq_weights, metric_name="sMAPE_floor50"
    )
    assert abs(sc_eval_metric - eq_manual_score) > 0.1, (
        f"static_convex eval_metric ({sc_eval_metric:.6f}) matches equal_weight "
        f"({eq_manual_score:.6f}). Eval is using equal_weight proxy!"
    )

    # --- Verify routing selected static_convex ---
    route = output.routing_table.iloc[0]
    assert route["selected_mode"] == "static_convex", (
        f"Expected static_convex to be selected, got {route['selected_mode']}"
    )

    # Verify final_refit_metric exists in routing
    assert "final_refit_metric" in route.index
    assert pd.notna(route["final_refit_metric"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
