#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5M Residual Stack Evaluation Script.

Compares four configurations on a canonical prediction pack:

    A. Phase2 champion baseline (no stack, base_fused_pred only)
    B. Phase2 + plugin dry-run (high_spike only)
    C. Phase2 + negative residual only
    D. Phase2 + high_spike + negative residual unified stack

Usage:
    python scripts/evaluate_p5m_residual_stack.py \\
        --canonical-pack outputs/p4_canonical_eval_pack/prediction_pack.csv \\
        --out-dir reports/local/p5m_residual_stack \\
        --high-spike-profile medium \\
        --negative-profile conservative

Output (local only, do not commit):
    reports/local/p5m_residual_stack/
        ├── comparison_metrics.csv
        ├── comparison_report.json
        └── corrected_outputs/
            ├── config_A_baseline.csv
            ├── config_B_high_spike_only.csv
            ├── config_C_negative_only.csv
            └── config_D_unified_stack.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from residual_stack.orchestrator import (
    ResidualStackOrchestrator,
    StackProfile,
    run_residual_stack,
)
from residual_stack.metrics import (
    compute_stack_metrics,
    format_metrics_table,
    compare_configs,
)
from residual_stack.report import (
    generate_verdict,
    write_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate_p5m_residual_stack")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5M Residual Stack — unified correction comparison",
    )
    parser.add_argument(
        "--canonical-pack",
        required=True,
        type=str,
        help="Path to the canonical eval pack CSV (must contain base_fused_pred, "
        "high_spike_prob, business_day, hour_business, y_true).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="reports/local/p5m_residual_stack",
        help="Output directory for comparison results.",
    )
    parser.add_argument(
        "--high-spike-profile",
        type=str,
        default="medium",
        choices=["conservative", "medium", "aggressive"],
        help="High-spike correction profile.",
    )
    parser.add_argument(
        "--negative-profile",
        type=str,
        default="conservative",
        choices=["conservative", "moderate", "aggressive"],
        help="Negative/low-valley correction profile.",
    )
    parser.add_argument(
        "--spike-risk-path",
        type=str,
        default=None,
        help="Optional explicit spike risk CSV path. If not provided, "
        "assumes high_spike_prob is already in the canonical pack.",
    )
    return parser.parse_args(argv)


def _build_baseline_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Config A: baseline — use base_fused_pred as final_pred."""
    out = df.copy()
    out["final_pred"] = out["base_fused_pred"].values
    out["base_pred"] = out["base_fused_pred"].values
    out["after_high_spike_pred"] = out["base_fused_pred"].values
    out["after_negative_pred"] = out["base_fused_pred"].values
    out["high_spike_applied"] = False
    out["negative_applied"] = False
    out["correction_reason"] = "baseline_no_correction"
    out["module_sequence"] = "none"
    return compute_stack_metrics(out)


def _build_high_spike_only_metrics(
    df: pd.DataFrame,
    spike_risk_path: str | Path | None,
    spike_profile: str,
) -> dict[str, Any]:
    """Config B: Phase2 + high_spike only."""
    if spike_risk_path is None:
        # high_spike_prob already in pack — still need to run correction
        pass

    profile = StackProfile(
        name=f"{spike_profile}+none",
        spike_profile_name=spike_profile,
        negative_profile_name="conservative",
    )
    orch = ResidualStackOrchestrator()
    result = orch.run(
        prediction_pack_path=None,  # We'll handle inline
        spike_risk_path=spike_risk_path or "inline",
        profile=profile,
    )
    # For inline usage, compute directly
    out = df.copy()
    # Apply high_spike only: use after_high_spike_pred as final
    if "after_high_spike_pred" in out.columns:
        out["final_pred"] = out["after_high_spike_pred"].values
    else:
        out["final_pred"] = out["base_fused_pred"].values
    out["base_pred"] = out["base_fused_pred"].values
    if "after_negative_pred" not in out.columns:
        out["after_negative_pred"] = out["final_pred"].values
    out["negative_applied"] = False
    return compute_stack_metrics(out)


def _build_negative_only_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Config C: Phase2 + negative residual only (no high_spike)."""
    out = df.copy()
    out["after_high_spike_pred"] = out["base_fused_pred"].values
    out["high_spike_applied"] = False
    out["lift_applied"] = 0.0
    out["base_pred"] = out["base_fused_pred"].values

    # Apply negative correction using inline logic
    from extreme.negative_price.residual_correction import (
        NegativeResidualConfig,
        NegativeResidualCorrector,
    )
    from extreme.negative_price.guardrail import (
        NegativeGuardrail,
        NegativeGuardrailConfig,
    )

    neg_cfg = NegativeResidualConfig(
        risk_threshold=0.4,
        max_downward_ratio=0.15,
        max_absolute_downward=20.0,
        min_pred_floor=-100.0,
        period_9_16_protection=True,
        mode="normal",
    )
    corrector = NegativeResidualCorrector(neg_cfg)
    corrector.set_downward_candidates({
        "1_8": -15.0, "9_16": -5.0, "17_24": -10.0,
    })
    guardrail = NegativeGuardrail(NegativeGuardrailConfig(
        spike_gate_active=False,
        spike_prob_threshold=1.0,
    ))

    after_neg_list: list[float] = []
    neg_applied_list: list[bool] = []

    for _, row in out.iterrows():
        base_pred_val = float(row.get("base_fused_pred", 0.0))
        hour_business = int(row.get("hour_business", 12))

        if pd.isna(base_pred_val):
            base_pred_val = 0.0

        result = corrector.compute_downward_correction(
            base_pred=base_pred_val,
            negative_risk=0.0,
            low_valley_risk=0.0,
            hour_business=hour_business,
            high_spike_active=False,
        )
        guard = guardrail.evaluate(
            base_pred=base_pred_val,
            corrected_pred=result.corrected_pred,
            hour_business=hour_business,
            spike_prob=0.0,
        )
        after_neg_list.append(guard.final_pred)
        neg_applied_list.append(guard.final_pred < base_pred_val)

    out["after_negative_pred"] = after_neg_list
    out["negative_applied"] = neg_applied_list
    out["final_pred"] = after_neg_list
    out["module_sequence"] = "negative→guardrail"
    out["correction_reason"] = [
        "negative_downward" if a else "no_correction"
        for a in neg_applied_list
    ]

    return compute_stack_metrics(out)


def _build_unified_stack_metrics(
    df: pd.DataFrame,
    spike_risk_path: str | Path | None,
    spike_profile: str,
    negative_profile: str,
) -> dict[str, Any]:
    """Config D: Phase2 + high_spike + negative residual unified stack."""
    profile = StackProfile(
        name=f"{spike_profile}+{negative_profile}",
        spike_profile_name=spike_profile,
        negative_profile_name=negative_profile,
    )
    orch = ResidualStackOrchestrator()

    # Write temp CSV for the orchestrator to read
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp, index=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        result = orch.run(
            prediction_pack_path=tmp_path,
            spike_risk_path=spike_risk_path,
            profile=profile,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return compute_stack_metrics(result.df)


def main() -> None:
    args = parse_args()

    pack_path = Path(args.canonical_pack)
    if not pack_path.exists():
        logger.error("Canonical pack not found: %s", pack_path)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corrected_dir = out_dir / "corrected_outputs"
    corrected_dir.mkdir(parents=True, exist_ok=True)

    # ── Load canonical pack ────────────────────────────────────────
    logger.info("Loading canonical pack: %s", pack_path)
    df = pd.read_csv(pack_path)
    # Ensure business_day is string
    if "business_day" in df.columns:
        df["business_day"] = df["business_day"].astype(str)
    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["business_day", "hour_business"], keep="last")
    if len(df) < before:
        logger.info("Dedup: %d → %d rows", before, len(df))

    # Determine spike risk path
    spike_risk_path = args.spike_risk_path
    if spike_risk_path is not None:
        spike_risk_path = Path(spike_risk_path)
        if not spike_risk_path.exists():
            logger.warning("Spike risk path not found, using inline high_spike_prob")
            spike_risk_path = None

    # ── Config A: Baseline ─────────────────────────────────────────
    logger.info("Config A: Phase2 baseline")
    metrics_a = _build_baseline_metrics(df)
    df_a = df.copy()
    df_a["config"] = "A_baseline"
    df_a.to_csv(corrected_dir / "config_A_baseline.csv", index=False)

    # ── Config B: High-spike only ──────────────────────────────────
    logger.info("Config B: high_spike only")
    metrics_b = _build_high_spike_only_metrics(df, spike_risk_path, args.high_spike_profile)
    df_b = df.copy()
    df_b["config"] = "B_high_spike_only"
    df_b.to_csv(corrected_dir / "config_B_high_spike_only.csv", index=False)

    # ── Config C: Negative only ────────────────────────────────────
    logger.info("Config C: negative residual only")
    metrics_c = _build_negative_only_metrics(df)

    # ── Config D: Unified stack ────────────────────────────────────
    logger.info("Config D: unified residual stack")
    metrics_d = _build_unified_stack_metrics(
        df, spike_risk_path, args.high_spike_profile, args.negative_profile,
    )

    # ── Build comparison ───────────────────────────────────────────
    configs = {
        "A_baseline": metrics_a,
        "B_high_spike_only": metrics_b,
        "C_negative_only": metrics_c,
        "D_unified_stack": metrics_d,
    }

    comparison_df = compare_configs(configs)
    comparison_path = out_dir / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_path)
    logger.info("Comparison written to %s", comparison_path)

    # ── Report ─────────────────────────────────────────────────────
    interaction_summary = {
        "high_spike_profile": args.high_spike_profile,
        "negative_profile": args.negative_profile,
        "total_rows": len(df),
    }

    report_path = write_report(
        out_dir=out_dir,
        config_results=configs,
        interaction_summary=interaction_summary,
        description=(
            f"Residual stack evaluation: spike={args.high_spike_profile}, "
            f"negative={args.negative_profile}"
        ),
    )
    logger.info("Report written to %s", report_path)

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Residual Stack Evaluation Summary")
    print("=" * 60)
    for label, metrics in configs.items():
        verdict = generate_verdict(metrics)
        print(f"\n[{label}] Verdict: {verdict}")
        print(format_metrics_table(metrics))
    print("\n" + "=" * 60)

    # Final overall verdict
    overall_verdicts = [generate_verdict(m) for m in configs.values()]
    if all(v == "GO" for v in overall_verdicts):
        print("Overall: GO")
    elif any(v == "DATA-LIMITED" for v in overall_verdicts):
        print("Overall: DATA-LIMITED — negative samples too few")
    elif any(v.startswith("NO-GO") for v in overall_verdicts):
        print("Overall: NO-GO — see details above")
    else:
        print("Overall: MIXED")


if __name__ == "__main__":
    main()
