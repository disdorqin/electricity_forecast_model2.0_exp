"""Tests for P4 Canonical Evaluation Pack.

Covers:
1. Canonical pack CSV has correct columns and timestamp-level dedup
2. (business_day, hour_business) key is unique — no duplicate timestamps
3. No reports/local/ files committed — only pack builder references
4. Baseline metrics reproduce Phase2 champion (sMAPE~20.86, severe~63)
5. No leakage columns present in the pack
6. Risk predictions align 1:1 with prediction pack timestamps
7. Model coverage completeness is reported

Run:
    python -m pytest tests/test_p4_canonical_eval_pack.py -v
    # or
    python tests/test_p4_canonical_eval_pack.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

# ── Paths to canonical pack outputs ──────────────────────────────────────

CANONICAL_DIR = _PROJECT_ROOT / "reports" / "local" / "p4_canonical"

PREDICTION_PACK_PATH = CANONICAL_DIR / "canonical_prediction_pack.csv"
RISK_PATH = CANONICAL_DIR / "canonical_risk_predictions.csv"
METRICS_PATH = CANONICAL_DIR / "canonical_metrics_baseline.json"
MANIFEST_PATH = CANONICAL_DIR / "canonical_manifest.json"

# Phase2 expected values
PHASE2_EXPECTED_SMAPE = 20.86
PHASE2_EXPECTED_SEVERE = 63
SMAPE_TOLERANCE = 0.05

# Allowed prediction-time columns (no leakage)
ALLOWED_PREDICTION_COLS = {
    "business_day", "hour_business", "timestamp", "period",
    "base_fused_pred",
    "y_pred_dayahead_proxy", "y_pred_naive_lag1", "y_pred_naive_lag7", "y_pred_lightgbm",
    "high_spike", "high_spike_flag",
    "final_pred_reference", "lift_applied", "reason_code",
}

# Forbidden leakage columns
FORBIDDEN_LEAKAGE_COLS = {
    "y_true",  # y_true is allowed in the pack (for evaluation), but must not be used at
               # prediction time. This test checks it's the only 'actual' column.
}

LEAKAGE_PATTERNS = [
    "actual", "实", "真值", "future", "d+1", "d_plus_1",
]


# ── Test 1: Pack exists and has correct shape ────────────────────────────

def test_canonical_pack_exists():
    """All canonical pack output files exist."""
    assert PREDICTION_PACK_PATH.exists(), f"Missing: {PREDICTION_PACK_PATH}"
    assert RISK_PATH.exists(), f"Missing: {RISK_PATH}"
    assert METRICS_PATH.exists(), f"Missing: {METRICS_PATH}"
    assert MANIFEST_PATH.exists(), f"Missing: {MANIFEST_PATH}"


def test_canonical_pack_columns():
    """Prediction pack has expected columns."""
    pp = pd.read_csv(PREDICTION_PACK_PATH)

    required_cols = {
        "business_day", "hour_business", "base_fused_pred", "y_true",
        "y_pred_lightgbm", "y_pred_dayahead_proxy",
        "y_pred_naive_lag1", "y_pred_naive_lag7",
        "high_spike", "high_spike_flag",
        "final_pred_reference", "lift_applied", "reason_code",
    }
    missing = required_cols - set(pp.columns)
    assert not missing, f"Missing required columns: {missing}"

    extra = set(pp.columns) - ALLOWED_PREDICTION_COLS - {"y_true"}
    unexpected = [c for c in extra if not c.startswith("y_pred_")]
    assert len(unexpected) == 0, f"Unexpected columns: {unexpected}"


# ── Test 2: Timestamp dedup — (business_day, hour_business) is unique ────

def test_timestamp_key_unique():
    """(business_day, hour_business) is a unique key — no duplicate timestamps."""
    pp = pd.read_csv(PREDICTION_PACK_PATH)
    n_before = len(pp)
    n_after = len(pp.drop_duplicates(subset=["business_day", "hour_business"]))
    assert n_before == n_after, (
        f"Found {n_before - n_after} duplicate timestamps"
    )


# ── Test 3: No reports/local committed (check .gitignore) ────────────────

def test_no_reports_local_in_git():
    """reports/local/ is gitignored — verify .gitignore covers it."""
    gitignore_path = _PROJECT_ROOT / ".gitignore"
    if not gitignore_path.exists():
        return  # skip if no .gitignore

    content = gitignore_path.read_text(encoding="utf-8")
    has_report = "reports/local" in content or "reports/local/" in content
    assert has_report, ".gitignore may not cover reports/local/ — check"


# ── Test 4: Baseline metrics reproduce Phase2 champion ───────────────────

def test_baseline_metrics_reproduce_phase2():
    """Baseline metrics reproduce Phase2 champion within tolerance."""
    assert METRICS_PATH.exists(), "Metrics file not found"
    with open(METRICS_PATH, encoding="utf-8") as f:
        metrics = json.load(f)

    smape = metrics["smape_floor50"]
    severe = metrics["severe_underestimate"]

    smape_delta = abs(smape - PHASE2_EXPECTED_SMAPE)
    assert smape_delta <= SMAPE_TOLERANCE, (
        f"sMAPE {smape} differs from expected {PHASE2_EXPECTED_SMAPE} "
        f"by {smape_delta:.4f} (tolerance ±{SMAPE_TOLERANCE})"
    )
    assert severe == PHASE2_EXPECTED_SEVERE, (
        f"Severe {severe} != expected {PHASE2_EXPECTED_SEVERE}"
    )


# ── Test 5: No leakage columns in prediction pack ────────────────────────

def test_no_leakage_columns():
    """No future-actual or D+1 leakage columns in the pack."""
    pp = pd.read_csv(PREDICTION_PACK_PATH)
    for col in pp.columns:
        col_lower = col.lower()
        for pattern in LEAKAGE_PATTERNS:
            if pattern in col_lower:
                # y_true is allowed (it's the eval target, clearly labelled)
                if col == "y_true":
                    continue
                raise AssertionError(
                    f"Potential leakage column found: '{col}' "
                    f"(matches pattern '{pattern}')"
                )


# ── Test 6: Risk predictions align 1:1 with prediction pack ─────────────

def test_risk_predictions_aligned():
    """Risk predictions have same timestamps as prediction pack."""
    pp = pd.read_csv(PREDICTION_PACK_PATH)
    rp = pd.read_csv(RISK_PATH)

    assert len(rp) == len(pp), (
        f"Risk rows ({len(rp)}) != prediction pack rows ({len(pp)})"
    )

    # Check key alignment
    pp_keys = set(zip(pp["business_day"], pp["hour_business"]))
    rp_keys = set(zip(rp["business_day"], rp["hour_business"]))

    missing_in_risk = pp_keys - rp_keys
    assert len(missing_in_risk) == 0, (
        f"{len(missing_in_risk)} timestamps in pack but not in risk predictions"
    )

    extra_in_risk = rp_keys - pp_keys
    assert len(extra_in_risk) == 0, (
        f"{len(extra_in_risk)} timestamps in risk but not in pack"
    )


# ── Test 7: Model coverage completeness ──────────────────────────────────

def test_model_coverage_reported():
    """Manifest reports model coverage with missing counts."""
    assert MANIFEST_PATH.exists(), "Manifest file not found"
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    model_cov = manifest.get("completeness", {}).get("model_coverage", {})
    assert len(model_cov) > 0, "Model coverage not reported in manifest"

    for model_name, coverage in model_cov.items():
        assert "present" in coverage, f"Missing 'present' for {model_name}"
        assert "missing" in coverage, f"Missing 'missing' for {model_name}"
        assert "pct_missing" in coverage, f"Missing 'pct_missing' for {model_name}"
        assert isinstance(coverage["present"], int)
        assert isinstance(coverage["missing"], int)
        assert isinstance(coverage["pct_missing"], (int, float))

    # Verify total rows consistency
    n_timestamps = manifest["date_range"]["n_actual_timestamps"]
    assert n_timestamps > 0, "Zero timestamps in manifest"


# ── Test 8: Manifest reports correct date range and completeness ─────────

def test_manifest_date_range():
    """Manifest date range covers 2025-11-01 to 2026-02-28."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    dr = manifest["date_range"]
    assert dr["start"] == "2025-11-01", f"Unexpected start: {dr['start']}"
    assert dr["end"] == "2026-02-28", f"Unexpected end: {dr['end']}"
    assert dr["n_expected_timestamps"] == 2880, (
        f"Expected 2880 timestamps (120 days × 24h), got {dr['n_expected_timestamps']}"
    )


# ── Test 9: Phase2 champion reproduction in manifest ─────────────────────

def test_phase2_reproduction_pass():
    """Manifest reports Phase2 champion reproduction pass."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    repro = manifest.get("phase2_champion_reproduction", {})
    assert repro.get("pass") is True, (
        f"Phase2 reproduction failed: smape={repro.get('actual_smape_floor50')}, "
        f"severe={repro.get('actual_severe')}"
    )


# ── Run directly ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("pack_exists", test_canonical_pack_exists),
        ("pack_columns", test_canonical_pack_columns),
        ("timestamp_key_unique", test_timestamp_key_unique),
        ("no_reports_local_in_git", test_no_reports_local_in_git),
        ("baseline_metrics_reproduce_phase2", test_baseline_metrics_reproduce_phase2),
        ("no_leakage_columns", test_no_leakage_columns),
        ("risk_predictions_aligned", test_risk_predictions_aligned),
        ("model_coverage_reported", test_model_coverage_reported),
        ("manifest_date_range", test_manifest_date_range),
        ("phase2_reproduction_pass", test_phase2_reproduction_pass),
    ]

    n_pass = 0
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [OK] {name}")
            n_pass += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(1 if n_fail > 0 else 0)
