"""Smoke test for the Step 4b intraday pipeline integration — Phase 12.

Creates tiny fixture data (fused predictions + intraday pack) and runs
apply_intraday_tracker_correction() under four scenarios:
  a. shadow mode   — y_fused must remain unchanged
  b. low_weight    — y_fused must change (fusion applied)
  c. FULL_DAY mode — y_fused must remain unchanged (correction disabled)
  d. missing pack  — safe fallback, y_fused must remain unchanged

Outputs per-scenario manifests and reports, plus a combined:
  smoke_manifest.json
  smoke_report.md

Exit code 0 if all scenarios pass, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from corrections.intraday_tracker.apply import apply_intraday_tracker_correction
from corrections.intraday_tracker.manifest import (
    IntradayManifest,
    build_manifest,
    write_manifest_and_report,
)
from corrections.intraday_tracker.policy import IntradayTrackerMainlineConfig

class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp,)):
            return str(obj)
        return super().default(obj)



# ---------------------------------------------------------------------------
# Fixture data builders
# ---------------------------------------------------------------------------
BUSINESS_DAY = pd.Timestamp("2026-02-15")


def _build_fused_predictions() -> pd.DataFrame:
    """Tiny fused_predictions.csv: 8 rows, hours 9-16."""
    rows = []
    for h in range(9, 17):
        rows.append({
            "business_day": BUSINESS_DAY,
            "hour_business": h,
            "ds": pd.Timestamp(f"2026-02-15 {h:02d}:00:00"),
            "target_day": BUSINESS_DAY,
            "rt_pred": 100.0 + h,
            "y_fused": 100.0 + h,
            "y_pred": 100.0 + h,
            "model_name": "sgdfnet",
        })
    return pd.DataFrame(rows)


def _build_intraday_pack() -> pd.DataFrame:
    """Tiny intraday pack: 4 rows, cutoff=12, hours 13-16."""
    rows = []
    for h in range(13, 17):
        rows.append({
            "business_day": BUSINESS_DAY,
            "target_hour": h,
            "hour_business": h,
            "cutoff_hour": 12,
            "ds": pd.Timestamp(f"2026-02-15 {h:02d}:00:00"),
            "mode": "INTRADAY",
            "base_model_name": "sgdfnet",
            "base_pred": 100.0 + h,
            "intraday_corrected_pred": 100.0 + h - 10.0,
            "intraday_final_correction": -10.0,
            "intraday_confidence": 0.6,
            "policy_decision": "LOW_WEIGHT",
            "fusion_weight": 0.12,
            "shadow_only_flag": False,
            "guardrail_reason": "",
            "n_observed": 5,
            "residual_std_today": 50.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------

def run_shadow(base: pd.DataFrame, pack: pd.DataFrame, out_dir: Path) -> dict:
    """Scenario a: shadow mode — y_fused must remain unchanged."""
    scenario_dir = out_dir / "shadow"
    original_yf = base["y_fused"].copy()

    result, stats = apply_intraday_tracker_correction(base, pack, mode="shadow")

    passed = (result["y_fused"].values == original_yf.values).all()
    passed = passed and stats["intraday_mode"] == "shadow"
    passed = passed and stats["applied_rows"] == 0
    passed = passed and stats["shadow_rows"] > 0

    manifest = build_manifest(stats, pack_path="fixture_intraday_pack.csv")
    write_manifest_and_report(manifest, str(scenario_dir))

    return {
        "scenario": "shadow",
        "mode": "shadow",
        "prediction_mode": "INTRADAY",
        "passed": passed,
        "stats": stats,
        "detail": (
            "y_fused unchanged, shadow_rows > 0, applied_rows == 0"
            if passed else
            "FAIL: y_fused was modified or stats incorrect"
        ),
    }


def run_low_weight(base: pd.DataFrame, pack: pd.DataFrame, out_dir: Path) -> dict:
    """Scenario b: low_weight mode — y_fused must change via fusion."""
    scenario_dir = out_dir / "low_weight"
    original_yf = base["y_fused"].copy()

    result, stats = apply_intraday_tracker_correction(base, pack, mode="low_weight")

    applied = result[result["intraday_applied"] == True]
    yf_changed = (result["y_fused"].values != original_yf.values).any()
    passed = yf_changed
    passed = passed and stats["applied_rows"] > 0
    passed = passed and len(applied) > 0

    # Verify fusion formula on a sample row
    if len(applied) > 0:
        row = applied.iloc[0]
        w = row["intraday_fusion_weight"]
        expected = (1.0 - w) * row["y_fused_before_intraday"] + w * row["intraday_shadow_pred"]
        passed = passed and abs(row["y_fused"] - expected) < 0.01

    manifest = build_manifest(stats, pack_path="fixture_intraday_pack.csv")
    write_manifest_and_report(manifest, str(scenario_dir))

    return {
        "scenario": "low_weight",
        "mode": "low_weight",
        "prediction_mode": "INTRADAY",
        "passed": passed,
        "stats": stats,
        "detail": (
            f"y_fused changed, {stats['applied_rows']} rows applied, fusion formula verified"
            if passed else
            "FAIL: y_fused not changed or fusion formula mismatch"
        ),
    }


def run_full_day(base: pd.DataFrame, pack: pd.DataFrame, out_dir: Path) -> dict:
    """Scenario c: FULL_DAY mode — y_fused must remain unchanged."""
    scenario_dir = out_dir / "full_day"
    original_yf = base["y_fused"].copy()

    result, stats = apply_intraday_tracker_correction(
        base, pack, mode="low_weight", prediction_mode="FULL_DAY"
    )

    passed = (result["y_fused"].values == original_yf.values).all()
    passed = passed and stats["applied_rows"] == 0

    manifest = build_manifest(stats, pack_path="fixture_intraday_pack.csv")
    write_manifest_and_report(manifest, str(scenario_dir))

    return {
        "scenario": "full_day",
        "mode": "low_weight",
        "prediction_mode": "FULL_DAY",
        "passed": passed,
        "stats": stats,
        "detail": (
            "FULL_DAY correctly disabled, y_fused unchanged, applied_rows == 0"
            if passed else
            "FAIL: FULL_DAY did not disable correction"
        ),
    }


def run_missing_pack(base: pd.DataFrame, out_dir: Path) -> dict:
    """Scenario d: missing pack — safe fallback, y_fused unchanged."""
    scenario_dir = out_dir / "missing_pack"
    original_yf = base["y_fused"].copy()
    empty_pack = pd.DataFrame()

    result, stats = apply_intraday_tracker_correction(base, empty_pack, mode="low_weight")

    passed = (result["y_fused"].values == original_yf.values).all()
    passed = passed and stats["fallback_reason"] == "empty_pack"
    passed = passed and stats["safe_fallback"] is True

    manifest = build_manifest(stats, pack_path="")
    write_manifest_and_report(manifest, str(scenario_dir))

    return {
        "scenario": "missing_pack",
        "mode": "low_weight",
        "prediction_mode": "INTRADAY",
        "passed": passed,
        "stats": stats,
        "detail": (
            "Safe fallback on empty pack, y_fused unchanged"
            if passed else
            "FAIL: fallback did not work correctly"
        ),
    }


# ---------------------------------------------------------------------------
# Combined output
# ---------------------------------------------------------------------------

def write_combined_outputs(results: list[dict], out_dir: Path) -> None:
    """Write smoke_manifest.json and smoke_report.md summarising all scenarios."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined manifest ---
    combined_manifest = {
        "smoke_test": "intraday_pipeline",
        "phase": 12,
        "fixture_only": True,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_passed": all(r["passed"] for r in results),
        "scenarios": [],
    }
    for r in results:
        combined_manifest["scenarios"].append({
            "scenario": r["scenario"],
            "mode": r["mode"],
            "prediction_mode": r["prediction_mode"],
            "passed": r["passed"],
            "applied_rows": r["stats"].get("applied_rows", 0),
            "shadow_rows": r["stats"].get("shadow_rows", 0),
            "disabled_rows": r["stats"].get("disabled_rows", 0),
            "matched_rows": r["stats"].get("matched_rows", 0),
            "fallback_reason": r["stats"].get("fallback_reason"),
            "safe_fallback": r["stats"].get("safe_fallback"),
            "detail": r["detail"],
        })

    manifest_path = out_dir / "smoke_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(combined_manifest, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    # --- Combined report ---
    lines = [
        "# Smoke Test: Intraday Pipeline (Phase 12)",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "**Fixture data only** — not real pack or real predictions.",
        "",
        "## Overall Result",
        "",
        f"**{'ALL PASSED' if combined_manifest['all_passed'] else 'SOME FAILED'}**",
        "",
        "## Scenarios",
        "",
        "| Scenario | Mode | Pred Mode | Passed | Applied | Shadow | Disabled | Fallback |",
        "|----------|------|-----------|--------|---------|--------|----------|----------|",
    ]
    for r in results:
        s = r["stats"]
        status = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| {r['scenario']} | {r['mode']} | {r['prediction_mode']} "
            f"| {status} | {s.get('applied_rows', 0)} | {s.get('shadow_rows', 0)} "
            f"| {s.get('disabled_rows', 0)} | {s.get('fallback_reason', '-')} |"
        )

    lines.append("")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        lines.extend([
            f"### {r['scenario']} — {status}",
            "",
            r["detail"],
            "",
        ])

    report_path = out_dir / "smoke_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test: Step 4b intraday pipeline (Phase 12)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/local/phase12/smoke_intraday_pipeline",
        help="Output directory for manifests and reports",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build fixture data
    base = _build_fused_predictions()
    pack = _build_intraday_pack()

    # Also write fixture CSVs for traceability
    base.to_csv(out_dir / "fixture_fused_predictions.csv", index=False, encoding="utf-8-sig")
    pack.to_csv(out_dir / "fixture_intraday_pack.csv", index=False, encoding="utf-8-sig")

    print("=" * 60)
    print("Smoke Test: Intraday Pipeline — Phase 12")
    print("=" * 60)
    print(f"  Fused predictions: {len(base)} rows, hours {base['hour_business'].min()}-{base['hour_business'].max()}")
    print(f"  Intraday pack:     {len(pack)} rows, cutoff=12, hours {pack['target_hour'].min()}-{pack['target_hour'].max()}")
    print(f"  Output directory:  {out_dir}")
    print()

    # Run scenarios
    results: list[dict] = []

    # a. Shadow mode
    r = run_shadow(base, pack, out_dir)
    results.append(r)
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"  [{tag}] shadow mode   — {r['detail']}")

    # b. Low-weight mode
    r = run_low_weight(base, pack, out_dir)
    results.append(r)
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"  [{tag}] low_weight    — {r['detail']}")

    # c. FULL_DAY mode
    r = run_full_day(base, pack, out_dir)
    results.append(r)
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"  [{tag}] FULL_DAY      — {r['detail']}")

    # d. Missing pack
    r = run_missing_pack(base, out_dir)
    results.append(r)
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"  [{tag}] missing pack  — {r['detail']}")

    # Combined outputs
    write_combined_outputs(results, out_dir)

    all_passed = all(r["passed"] for r in results)
    print()
    print("-" * 60)
    if all_passed:
        print("ALL 4 SCENARIOS PASSED")
    else:
        n_fail = sum(1 for r in results if not r["passed"])
        print(f"{n_fail} SCENARIO(S) FAILED")
    print(f"Manifest: {out_dir / 'smoke_manifest.json'}")
    print(f"Report:   {out_dir / 'smoke_report.md'}")
    print("-" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
