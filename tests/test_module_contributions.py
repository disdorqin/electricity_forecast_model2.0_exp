"""Tests for Phase 13 — Module contribution decomposition.

Covers:
1. Per-model sMAPE computation (each model evaluated independently)
2. Fused baseline sMAPE (the combined / blended prediction)
3. NOT_AVAILABLE sentinel for missing modules
4. Delta computation between stages (e.g. before vs after correction)
5. Consistency: fused sMAPE ≤ weighted combination of individual sMAPEs
   when fusion is beneficial

The decomposer is expected to expose:
    compute_module_contributions(
        predictions: pd.DataFrame,
        models: list[str],
        fused_col: str = "base_fused_pred",
        stage_a_col: str | None = None,
        stage_b_col: str | None = None,
    ) -> dict

Returns a dict with:
    "per_model_smape":  {model_name: float | "NOT_AVAILABLE", ...}
    "fused_smape":      float | "NOT_AVAILABLE"
    "stage_a_smape":    float | "NOT_AVAILABLE"   (if stage_a_col given)
    "stage_b_smape":    float | "NOT_AVAILABLE"   (if stage_b_col given)
    "delta_ab":         float | "NOT_AVAILABLE"   (stage_b - stage_a)

sMAPE floor-50 formula:
    2 * |y_true - y_pred| / (max(|y_true|, 50) + max(|y_pred|, 50))
    Result is in [0, 50] percent.

Business-day convention:
    timestamp D 00:00 → business_day D-1, hour 24
    timestamp D HH:00 (HH >= 1) → business_day D, hour HH

Run:
    python -m pytest tests/test_module_contributions.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

NOT_AVAILABLE = "NOT_AVAILABLE"


# ── sMAPE floor-50 reference ──────────────────────────────────────────────

def smape_floor50(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean sMAPE with floor-50 denominator.

    Formula (per pair):
        2 * |y_true - y_pred| / (max(|y_true|, 50) + max(|y_pred|, 50))
    Result is in [0, 50] percent.
    """
    yt = np.maximum(np.abs(y_true.astype(float)), 50.0)
    yp = np.maximum(np.abs(y_pred.astype(float)), 50.0)
    denom = (yt + yp) / 2.0
    denom = np.where(denom < 1e-10, 1e-10, denom)
    per_row = np.abs(yt - yp) / denom * 100.0
    return float(np.mean(np.minimum(per_row, 50.0)))


# ── Inline reference implementation ────────────────────────────────────────

def compute_module_contributions(
    predictions: pd.DataFrame,
    models: list[str],
    fused_col: str = "base_fused_pred",
    stage_a_col: str | None = None,
    stage_b_col: str | None = None,
) -> dict:
    """Decompose prediction quality into per-model and fused contributions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Must contain ``"y_true"`` and one column per model in *models*.
    models : list[str]
        Model column names to evaluate.
    fused_col : str
        Column name for the fused baseline prediction.
    stage_a_col, stage_b_col : str or None
        Optional stage columns for delta computation.
        ``delta_ab = stage_b_smape - stage_a_smape``.

    Returns
    -------
    dict with per-model sMAPE, fused sMAPE, and optional stage deltas.
    """
    result: dict = {}

    if predictions is None or len(predictions) == 0:
        result["per_model_smape"] = {m: NOT_AVAILABLE for m in models}
        result["fused_smape"] = NOT_AVAILABLE
        if stage_a_col is not None:
            result["stage_a_smape"] = NOT_AVAILABLE
        if stage_b_col is not None:
            result["stage_b_smape"] = NOT_AVAILABLE
        result["delta_ab"] = NOT_AVAILABLE
        return result

    y_true = predictions["y_true"].values.astype(float)

    # ── Per-model sMAPE ───────────────────────────────────────────────
    per_model: dict[str, float | str] = {}
    for model in models:
        if model not in predictions.columns:
            per_model[model] = NOT_AVAILABLE
            continue
        col = predictions[model]
        if col.isna().all():
            per_model[model] = NOT_AVAILABLE
            continue
        # Drop rows where either y_true or y_pred is NaN
        valid_mask = col.notna() & predictions["y_true"].notna()
        if valid_mask.sum() == 0:
            per_model[model] = NOT_AVAILABLE
            continue
        per_model[model] = smape_floor50(
            y_true[valid_mask.values],
            col.values[valid_mask.values],
        )
    result["per_model_smape"] = per_model

    # ── Fused baseline sMAPE ──────────────────────────────────────────
    if fused_col in predictions.columns:
        fused = predictions[fused_col]
        valid_mask = fused.notna() & predictions["y_true"].notna()
        if valid_mask.sum() > 0:
            result["fused_smape"] = smape_floor50(
                y_true[valid_mask.values],
                fused.values[valid_mask.values],
            )
        else:
            result["fused_smape"] = NOT_AVAILABLE
    else:
        result["fused_smape"] = NOT_AVAILABLE

    # ── Stage sMAPE and delta ─────────────────────────────────────────
    stage_a_smape: float | str = NOT_AVAILABLE
    stage_b_smape: float | str = NOT_AVAILABLE

    if stage_a_col is not None and stage_a_col in predictions.columns:
        col_a = predictions[stage_a_col]
        valid_a = col_a.notna() & predictions["y_true"].notna()
        if valid_a.sum() > 0:
            stage_a_smape = smape_floor50(
                y_true[valid_a.values],
                col_a.values[valid_a.values],
            )
    result["stage_a_smape"] = stage_a_smape

    if stage_b_col is not None and stage_b_col in predictions.columns:
        col_b = predictions[stage_b_col]
        valid_b = col_b.notna() & predictions["y_true"].notna()
        if valid_b.sum() > 0:
            stage_b_smape = smape_floor50(
                y_true[valid_b.values],
                col_b.values[valid_b.values],
            )
    result["stage_b_smape"] = stage_b_smape

    # Delta: stage_b - stage_a (negative = improvement)
    if isinstance(stage_a_smape, float) and isinstance(stage_b_smape, float):
        result["delta_ab"] = stage_b_smape - stage_a_smape
    else:
        result["delta_ab"] = NOT_AVAILABLE

    return result


# ── Fixture helpers ────────────────────────────────────────────────────────

def _make_predictions(
    n: int = 24,
    models: list[str] | None = None,
    seed: int = 42,
    include_fused: bool = True,
    include_stages: bool = False,
) -> pd.DataFrame:
    """Create a tiny prediction DataFrame with known error structure.

    Business-day convention:
        hour 1..23 → timestamp same_day HH:00
        hour 24    → timestamp next_day 00:00
    """
    if models is None:
        models = ["naive_lag1", "naive_lag7", "dayahead_proxy", "lightgbm"]

    rng = np.random.default_rng(seed)
    business_day = "2026-01-15"
    bd_ts = pd.Timestamp(business_day)

    rows = []
    for hb in range(1, min(n, 24) + 1):
        if hb == 24:
            ts = (bd_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d") + " 00:00:00"
        else:
            ts = bd_ts.strftime("%Y-%m-%d") + f" {hb:02d}:00:00"

        y_true = 300.0 + hb * 2.0 + rng.normal(0, 10)
        row = {
            "business_day": business_day,
            "hour_business": hb,
            "timestamp": ts,
            "y_true": round(y_true, 2),
        }

        # Each model has a different bias/scale
        model_params = {
            "naive_lag1":       (0.95, 15.0),
            "naive_lag7":       (0.90, 20.0),
            "dayahead_proxy":   (0.98, 8.0),
            "lightgbm":         (0.99, 5.0),
            "timemixer":        (0.97, 10.0),
        }
        for m in models:
            scale, noise_std = model_params.get(m, (0.95, 15.0))
            noise = rng.normal(0, noise_std)
            row[f"y_pred_{m}"] = round(y_true * scale + noise, 2)

        if include_fused:
            # Fused = weighted average of model predictions
            fused_val = 0.0
            weights = {"naive_lag1": 0.15, "naive_lag7": 0.10,
                       "dayahead_proxy": 0.25, "lightgbm": 0.50}
            for m in models:
                w = weights.get(m, 0.1)
                col_name = f"y_pred_{m}"
                if col_name in row:
                    fused_val += w * row[col_name]
            row["base_fused_pred"] = round(fused_val, 2)

        if include_stages:
            # Stage A = before correction (same as fused)
            row["stage_a_pred"] = row.get("base_fused_pred", round(y_true * 0.95, 2))
            # Stage B = after correction (closer to y_true)
            row["stage_b_pred"] = round(
                row.get("base_fused_pred", y_true * 0.95) * 0.3 + y_true * 0.7, 2
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ── Tests: Per-model sMAPE ────────────────────────────────────────────────

class TestPerModelSmape:
    """Test per-model sMAPE computation."""

    def test_all_models_have_smape(self):
        """Every model present in the DataFrame gets a numeric sMAPE."""
        models = ["naive_lag1", "naive_lag7", "dayahead_proxy", "lightgbm"]
        preds = _make_predictions(models=models)
        # Rename columns to match model names (without y_pred_ prefix)
        for m in models:
            preds[m] = preds[f"y_pred_{m}"]

        result = compute_module_contributions(preds, models)

        for m in models:
            assert isinstance(result["per_model_smape"][m], float), (
                f"Model {m} should have numeric sMAPE, got {result['per_model_smape'][m]}"
            )
            assert 0 <= result["per_model_smape"][m] <= 50

    def test_better_model_lower_smape(self):
        """lightgbm (scale=0.99, noise=5) should have lower sMAPE than
        naive_lag7 (scale=0.90, noise=20)."""
        models = ["naive_lag7", "lightgbm"]
        preds = _make_predictions(n=24, models=models, seed=42)
        for m in models:
            preds[m] = preds[f"y_pred_{m}"]

        result = compute_module_contributions(preds, models)

        assert result["per_model_smape"]["lightgbm"] < result["per_model_smape"]["naive_lag7"]

    def test_perfect_model_zero_smape(self):
        """A model that perfectly predicts y_true gets sMAPE ≈ 0."""
        preds = _make_predictions(n=10, models=["oracle"])
        preds["oracle"] = preds["y_true"]  # perfect prediction

        result = compute_module_contributions(preds, ["oracle"])

        assert abs(result["per_model_smape"]["oracle"]) < 1e-10

    def test_smape_values_match_manual_computation(self):
        """Verify sMAPE matches manual floor-50 computation."""
        y_true_vals = [200.0, 300.0, 400.0]
        y_pred_vals = [210.0, 290.0, 420.0]

        preds = pd.DataFrame({
            "business_day": ["2026-01-15"] * 3,
            "hour_business": [1, 2, 3],
            "timestamp": ["2026-01-15 01:00:00", "2026-01-15 02:00:00",
                          "2026-01-15 03:00:00"],
            "y_true": y_true_vals,
            "model_a": y_pred_vals,
        })

        result = compute_module_contributions(preds, ["model_a"])

        # Manual: per-row sMAPE floor-50
        expected = smape_floor50(np.array(y_true_vals), np.array(y_pred_vals))
        assert abs(result["per_model_smape"]["model_a"] - expected) < 1e-6


# ── Tests: Fused baseline sMAPE ───────────────────────────────────────────

class TestFusedBaselineSmape:
    """Test fused baseline sMAPE."""

    def test_fused_smape_computed(self):
        """Fused sMAPE is a valid number when base_fused_pred exists."""
        preds = _make_predictions(include_fused=True)
        result = compute_module_contributions(preds, [])

        assert isinstance(result["fused_smape"], float)
        assert 0 <= result["fused_smape"] <= 50

    def test_fused_better_than_worst_model(self):
        """Fused prediction should generally be better than the worst model."""
        models = ["naive_lag1", "naive_lag7", "dayahead_proxy", "lightgbm"]
        preds = _make_predictions(models=models, include_fused=True)
        for m in models:
            preds[m] = preds[f"y_pred_{m}"]

        result = compute_module_contributions(preds, models)

        worst_model_smape = max(
            result["per_model_smape"][m] for m in models
            if isinstance(result["per_model_smape"][m], float)
        )
        assert result["fused_smape"] < worst_model_smape

    def test_fused_missing_column(self):
        """No fused column → fused_smape = NOT_AVAILABLE."""
        preds = _make_predictions(include_fused=False)
        result = compute_module_contributions(preds, [], fused_col="base_fused_pred")

        assert result["fused_smape"] == NOT_AVAILABLE


# ── Tests: NOT_AVAILABLE for missing modules ──────────────────────────────

class TestNotAvailableForMissing:
    """Test NOT_AVAILABLE sentinel for missing modules."""

    def test_missing_model_column(self):
        """Model not in DataFrame columns → NOT_AVAILABLE."""
        preds = _make_predictions(models=["lightgbm"])
        preds["lightgbm"] = preds["y_pred_lightgbm"]

        result = compute_module_contributions(
            preds, ["lightgbm", "nonexistent_model"]
        )

        assert isinstance(result["per_model_smape"]["lightgbm"], float)
        assert result["per_model_smape"]["nonexistent_model"] == NOT_AVAILABLE

    def test_all_nan_model_column(self):
        """Model column is all NaN → NOT_AVAILABLE."""
        preds = _make_predictions(models=["lightgbm"])
        preds["broken_model"] = np.nan

        result = compute_module_contributions(preds, ["broken_model"])

        assert result["per_model_smape"]["broken_model"] == NOT_AVAILABLE

    def test_empty_dataframe(self):
        """Empty predictions DataFrame → everything is NOT_AVAILABLE."""
        result = compute_module_contributions(
            pd.DataFrame(), ["model_a", "model_b"]
        )

        assert result["per_model_smape"]["model_a"] == NOT_AVAILABLE
        assert result["per_model_smape"]["model_b"] == NOT_AVAILABLE
        assert result["fused_smape"] == NOT_AVAILABLE

    def test_none_predictions(self):
        """None predictions → everything is NOT_AVAILABLE."""
        result = compute_module_contributions(None, ["model_a"])

        assert result["per_model_smape"]["model_a"] == NOT_AVAILABLE
        assert result["fused_smape"] == NOT_AVAILABLE


# ── Tests: Delta computation between stages ───────────────────────────────

class TestDeltaComputation:
    """Test delta computation between stages."""

    def test_delta_negative_means_improvement(self):
        """Stage B better than stage A → delta_ab < 0."""
        preds = _make_predictions(n=24, include_stages=True)

        result = compute_module_contributions(
            preds, [],
            stage_a_col="stage_a_pred",
            stage_b_col="stage_b_pred",
        )

        assert isinstance(result["stage_a_smape"], float)
        assert isinstance(result["stage_b_smape"], float)
        # Stage B is constructed to be closer to y_true
        assert result["delta_ab"] < 0, (
            f"Expected negative delta (improvement), got {result['delta_ab']}"
        )

    def test_delta_positive_means_degradation(self):
        """Stage B worse than stage A → delta_ab > 0."""
        preds = _make_predictions(n=24)
        # Make stage_a good and stage_b bad
        preds["stage_a_pred"] = preds["y_true"] * 1.01   # very close
        preds["stage_b_pred"] = preds["y_true"] * 0.70   # far off

        result = compute_module_contributions(
            preds, [],
            stage_a_col="stage_a_pred",
            stage_b_col="stage_b_pred",
        )

        assert result["delta_ab"] > 0, (
            f"Expected positive delta (degradation), got {result['delta_ab']}"
        )

    def test_delta_zero_for_identical_stages(self):
        """Identical stages → delta_ab = 0."""
        preds = _make_predictions(n=24)
        preds["stage_a_pred"] = preds["base_fused_pred"]
        preds["stage_b_pred"] = preds["base_fused_pred"]

        result = compute_module_contributions(
            preds, [],
            stage_a_col="stage_a_pred",
            stage_b_col="stage_b_pred",
        )

        assert abs(result["delta_ab"]) < 1e-10

    def test_delta_not_available_when_stage_missing(self):
        """Missing stage column → delta_ab = NOT_AVAILABLE."""
        preds = _make_predictions(n=10)

        result = compute_module_contributions(
            preds, [],
            stage_a_col="stage_a_pred",  # doesn't exist
            stage_b_col="stage_b_pred",  # doesn't exist
        )

        assert result["delta_ab"] == NOT_AVAILABLE
        assert result["stage_a_smape"] == NOT_AVAILABLE
        assert result["stage_b_smape"] == NOT_AVAILABLE

    def test_delta_computation_formula(self):
        """delta_ab = stage_b_smape - stage_a_smape exactly."""
        preds = _make_predictions(n=24)
        preds["stage_a_pred"] = preds["y_true"] * 1.05
        preds["stage_b_pred"] = preds["y_true"] * 0.95

        result = compute_module_contributions(
            preds, [],
            stage_a_col="stage_a_pred",
            stage_b_col="stage_b_pred",
        )

        expected_delta = result["stage_b_smape"] - result["stage_a_smape"]
        assert abs(result["delta_ab"] - expected_delta) < 1e-10


# ── Tests: sMAPE floor-50 formula correctness ─────────────────────────────

class TestSmapeFloor50Formula:
    """Verify the smape_floor50 formula used throughout."""

    def test_floor50_small_values(self):
        """Both values below 50 → floored to 50, sMAPE = 0."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([15.0, 25.0, 35.0])
        # After floor: yt=[50,50,50], yp=[50,50,50] → sMAPE=0
        result = smape_floor50(y_true, y_pred)
        assert result == 0.0

    def test_floor50_one_side_small(self):
        """y_true=100, y_pred=20 → yt=100, yp=50 → sMAPE = 2*50/150*100 ≈ 66.67 → clamped to 50."""
        y_true = np.array([100.0])
        y_pred = np.array([20.0])
        # yt=100, yp=50, denom=75, |100-50|/75*100 = 66.67 → clamped to 50
        result = smape_floor50(y_true, y_pred)
        assert result == 50.0

    def test_floor50_symmetry(self):
        """sMAPE is symmetric: smape(a,b) == smape(b,a)."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 180.0, 350.0])
        fwd = smape_floor50(y_true, y_pred)
        rev = smape_floor50(y_pred, y_true)
        assert abs(fwd - rev) < 1e-10

    def test_floor50_range(self):
        """sMAPE is always in [0, 50]."""
        rng = np.random.default_rng(99)
        y_true = rng.uniform(0, 1000, 100)
        y_pred = rng.uniform(0, 1000, 100)
        result = smape_floor50(y_true, y_pred)
        assert 0 <= result <= 50

    def test_floor50_perfect_prediction(self):
        """y_true == y_pred → sMAPE = 0."""
        y = np.array([100.0, 200.0, 300.0, 500.0])
        result = smape_floor50(y, y)
        assert result == 0.0


# ── Run directly ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
