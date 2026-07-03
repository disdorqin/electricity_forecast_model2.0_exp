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
from residual_stack.risk_source import (
    RiskSource,
    detect_risk_source,
    resolve_risk_policy,
    format_risk_verdict,
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
    parser.add_argument(
        "--allow-synthetic-risk",
        action="store_true",
        default=False,
        help="Allow synthetic risk (high_spike_flag → 0.9/0.05) for "
        "dry-run evaluation. Without this flag, configs B/D are "
        "DATA-MISSING when only a binary flag is available.",
    )
    parser.add_argument(
        "--negative-risk-path",
        type=str,
        default=None,
        help="Path to negative risk predictions CSV (must contain "
        "negative_prob, low_valley_prob, overestimate_low_prob). "
        "If provided, these probabilities are used for Config C/D "
        "negative correction instead of heuristic fallback.",
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
    risk_source: RiskSource = RiskSource.MISSING,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Config B: Phase2 + high_spike only.

    Loads spike probabilities from *spike_risk_path*, inline
    ``high_spike_prob``, or (if *allow_synthetic*) ``high_spike_flag``.

    Returns metrics dict with ``_risk_source`` / ``_run_status`` metadata.
    """
    out = df.copy()
    out["base_pred"] = out["base_fused_pred"].values

    # ── Resolve spike probability ───────────────────────────────────
    if spike_risk_path is not None:
        risk_df = pd.read_csv(spike_risk_path)
        merged = pd.merge(
            out, risk_df[["business_day", "hour_business", "high_spike_prob"]],
            on=["business_day", "hour_business"], how="left",
        )
        spike_probs = merged["high_spike_prob"].fillna(0.0).values
    elif "high_spike_prob" in out.columns:
        spike_probs = out["high_spike_prob"].fillna(0.0).values
    elif "high_spike_flag" in out.columns:
        if allow_synthetic and risk_source == RiskSource.SYNTHETIC_FLAG:
            # Dry-run: synthesise probability from flag
            spike_probs = np.where(out["high_spike_flag"].astype(bool), 0.9, 0.05)
        else:
            # DATA-MISSING — no real spike risk data
            out["after_high_spike_pred"] = out["base_fused_pred"].values
            out["final_pred"] = out["base_fused_pred"].values
            out["after_negative_pred"] = out["base_fused_pred"].values
            out["high_spike_applied"] = False
            out["negative_applied"] = False
            out["correction_reason"] = "no_spike_signal"
            out["module_sequence"] = "none"
            m = compute_stack_metrics(out)
            m["_risk_source"] = risk_source.value
            m["_run_status"] = "data_missing"
            m["_allow_synthetic"] = allow_synthetic
            return m
    else:
        # No spike signal at all — skip correction
        out["after_high_spike_pred"] = out["base_fused_pred"].values
        out["final_pred"] = out["base_fused_pred"].values
        out["after_negative_pred"] = out["base_fused_pred"].values
        out["high_spike_applied"] = False
        out["negative_applied"] = False
        out["correction_reason"] = "no_spike_signal"
        out["module_sequence"] = "none"
        return compute_stack_metrics(out)

    # ── Apply high-spike correction row-by-row ──────────────────────
    from extreme.realtime_high_spike.residual_lift import (
        CorrectionMode, ResidualLiftConfig, ResidualLiftCorrector, get_period,
    )
    from extreme.realtime_high_spike.guardrail import (
        GuardrailConfig, SpikeGuardrail,
    )
    from extreme.realtime_high_spike.apply_correction import CorrectionProfile
    spike_cfg = CorrectionProfile(
        name=spike_profile,
        mode=CorrectionMode.NORMAL,
    )

    lcfg = spike_cfg.to_lift_config()
    gcfg = spike_cfg.to_guardrail_config()
    corrector = ResidualLiftCorrector(lcfg)
    corrector.set_lift_candidates({p: 50.0 for p in ["1_8", "9_16", "17_24"]})
    corrector._lift_candidates["9_16"] *= lcfg.period_9_16_boost
    guardrail = SpikeGuardrail(gcfg)

    spike_list: list[float] = []
    final_list: list[float] = []
    lift_list: list[float] = []

    for _, row in out.iterrows():
        base_pred = float(row.get("base_fused_pred", 0.0))
        prob = float(spike_probs[row.name]) if row.name < len(spike_probs) else 0.0
        hour_biz = int(row.get("hour_business", 12))

        if pd.isna(base_pred):
            base_pred = 0.0
        if pd.isna(prob):
            prob = 0.0

        lift_result = corrector.compute_lift(base_pred, prob, hour_biz)
        guard_result = guardrail.evaluate(
            base_pred=base_pred, corrected_pred=lift_result.corrected_pred,
            spike_prob=prob, hour_business=hour_biz,
        )
        spike_list.append(lift_result.corrected_pred)
        final_list.append(guard_result.final_pred)
        lift_list.append(guard_result.final_pred - base_pred)

    out["after_high_spike_pred"] = final_list
    out["high_spike_applied"] = [l > 0 for l in lift_list]
    out["after_negative_pred"] = final_list
    out["negative_applied"] = False
    out["final_pred"] = final_list
    out["module_sequence"] = "high_spike→guardrail"
    out["correction_reason"] = [
        "high_spike_lifted" if l > 0 else "no_correction" for l in lift_list
    ]
    m = compute_stack_metrics(out)
    _, run_status = resolve_risk_policy(risk_source, allow_synthetic)
    m["_risk_source"] = risk_source.value
    m["_run_status"] = run_status
    m["_allow_synthetic"] = allow_synthetic
    return m


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

        # Use calibrated negative risk if available, else heuristic fallback
        neg_risk = float(row.get("negative_prob", 0.0))
        lv_risk = float(row.get("low_valley_prob", 0.0))

        result = corrector.compute_downward_correction(
            base_pred=base_pred_val,
            negative_risk=neg_risk,
            low_valley_risk=lv_risk,
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
    risk_source: RiskSource = RiskSource.MISSING,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    """Config D: Phase2 + high_spike + negative residual unified stack.

    Returns metrics dict with ``_risk_source`` / ``_run_status`` metadata.
    """
    # Build synthetic spike risk CSV if none was provided but we have a flag
    import tempfile

    tmp_pack = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp_pack, index=False)
    tmp_pack_path = tmp_pack.name
    tmp_pack.close()

    tmp_risk: str | None = None
    try:
        if spike_risk_path is not None:
            effective_risk_path = str(spike_risk_path)
        elif "high_spike_prob" in df.columns:
            # Write just the prob column
            risk_df = df[["business_day", "hour_business", "high_spike_prob"]].copy()
            tmp_risk = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w").name
            risk_df.to_csv(tmp_risk, index=False)
            effective_risk_path = tmp_risk
        elif "high_spike_flag" in df.columns:
            if allow_synthetic and risk_source == RiskSource.SYNTHETIC_FLAG:
                risk_df = df[["business_day", "hour_business"]].copy()
                risk_df["high_spike_prob"] = np.where(
                    df["high_spike_flag"].astype(bool), 0.9, 0.05
                )
                tmp_risk = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w").name
                risk_df.to_csv(tmp_risk, index=False)
                effective_risk_path = tmp_risk
            else:
                # DATA-MISSING — no real spike risk data
                tmp_risk = None
                md: dict[str, Any] = {}
                md["_risk_source"] = risk_source.value
                md["_run_status"] = "data_missing"
                md["_allow_synthetic"] = allow_synthetic
                Path(tmp_pack_path).unlink(missing_ok=True)
                return md
        else:
            effective_risk_path = None

        profile = StackProfile(
            name=f"{spike_profile}+{negative_profile}",
            spike_profile_name=spike_profile,
            negative_profile_name=negative_profile,
        )
        orch = ResidualStackOrchestrator()
        result = orch.run(
            prediction_pack_path=tmp_pack_path,
            spike_risk_path=effective_risk_path,
            profile=profile,
        )
        m = compute_stack_metrics(result.df)
        _, run_status = resolve_risk_policy(risk_source, allow_synthetic)
        m["_risk_source"] = risk_source.value
        m["_run_status"] = run_status
        m["_allow_synthetic"] = allow_synthetic
        return m
    finally:
        Path(tmp_pack_path).unlink(missing_ok=True)
        if tmp_risk is not None:
            Path(tmp_risk).unlink(missing_ok=True)


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

    # ── Load negative risk path ──────────────────────────────────────
    negative_risk_path: Path | None = None
    if args.negative_risk_path is not None:
        neg_path = Path(args.negative_risk_path)
        if neg_path.exists():
            logger.info("Loading negative risk: %s", neg_path)
            neg_risk_df = pd.read_csv(neg_path)
            for col in ("business_day",):
                if col in neg_risk_df.columns:
                    neg_risk_df[col] = neg_risk_df[col].astype(str)
            merge_cols = [c for c in ["business_day", "hour_business",
                                       "negative_prob", "low_valley_prob",
                                       "overestimate_low_prob"]
                         if c in neg_risk_df.columns]
            if len(merge_cols) >= 4:
                df = pd.merge(df, neg_risk_df[merge_cols],
                              on=["business_day", "hour_business"],
                              how="left", suffixes=("", "_risk"))
                # Fill missing risk with 0
                for c in ["negative_prob", "low_valley_prob", "overestimate_low_prob"]:
                    if c in df.columns:
                        df[c] = df[c].fillna(0.0)
                logger.info("Merged negative risk: %d rows, cols=%s",
                           len(df), merge_cols)
                negative_risk_path = neg_path
            else:
                logger.warning("Negative risk CSV missing required columns")
        else:
            logger.warning("Negative risk path not found: %s", neg_path)

    # Determine spike risk path
    spike_risk_path = args.spike_risk_path
    if spike_risk_path is not None:
        spike_risk_path = Path(spike_risk_path)
        if not spike_risk_path.exists():
            logger.warning("Spike risk path not found, using inline high_spike_prob")
            spike_risk_path = None

    # ── Detect risk source ─────────────────────────────────────────
    risk_source = detect_risk_source(
        spike_risk_path=str(spike_risk_path) if spike_risk_path else None,
        df=df,
    )
    _, rs = resolve_risk_policy(risk_source, args.allow_synthetic_risk)
    logger.info(
        "Risk source: %s → %s (allow_synthetic=%s)",
        risk_source.value, rs, args.allow_synthetic_risk,
    )

    # ── Config A: Baseline ─────────────────────────────────────────
    logger.info("Config A: Phase2 baseline")
    metrics_a = _build_baseline_metrics(df)
    metrics_a["_risk_source"] = risk_source.value
    metrics_a["_run_status"] = "official"  # A doesn't use spike risk
    metrics_a["_allow_synthetic"] = args.allow_synthetic_risk
    df_a = df.copy()
    df_a["config"] = "A_baseline"
    df_a.to_csv(corrected_dir / "config_A_baseline.csv", index=False)

    # ── Config B: High-spike only ──────────────────────────────────
    logger.info("Config B: high_spike only (risk=%s)", risk_source.value)
    metrics_b = _build_high_spike_only_metrics(
        df, spike_risk_path, args.high_spike_profile,
        risk_source=risk_source, allow_synthetic=args.allow_synthetic_risk,
    )
    df_b = df.copy()
    df_b["config"] = "B_high_spike_only"
    df_b.to_csv(corrected_dir / "config_B_high_spike_only.csv", index=False)

    # ── Config C: Negative only ────────────────────────────────────
    logger.info("Config C: negative residual only")
    metrics_c = _build_negative_only_metrics(df)
    metrics_c["_risk_source"] = risk_source.value
    metrics_c["_run_status"] = "official"  # C doesn't use spike risk
    metrics_c["_allow_synthetic"] = args.allow_synthetic_risk

    # ── Config D: Unified stack ────────────────────────────────────
    logger.info("Config D: unified residual stack (risk=%s)", risk_source.value)
    metrics_d = _build_unified_stack_metrics(
        df, spike_risk_path, args.high_spike_profile, args.negative_profile,
        risk_source=risk_source, allow_synthetic=args.allow_synthetic_risk,
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
        run_status = metrics.get("_run_status", "official")
        raw = generate_verdict(metrics, run_status=run_status)
        prefixed = format_risk_verdict(run_status, raw)
        print(f"\n[{label}] Verdict: {prefixed}")
        print(f"  Risk source: {metrics.get('_risk_source', 'unknown')} | Status: {run_status}")
        print(format_metrics_table(metrics))
    print("\n" + "=" * 60)

    # Final overall verdict from official results only
    official_raw = [
        generate_verdict(m, run_status=m.get("_run_status", "official"))
        for m in configs.values()
        if m.get("_run_status") == "official"
    ]
    if not official_raw:
        print("Overall: NO-OFFICIAL-RESULTS — no config had official risk data")
    elif all(v == "GO" for v in official_raw):
        print("Overall: GO")
    elif any("DATA-MISSING" in v or "DATA-LIMITED" in v for v in official_raw):
        print("Overall: DATA-MISSING")
    elif any(v.startswith("NO-GO") for v in official_raw):
        print("Overall: NO-GO — see details above")
    else:
        print("Overall: MIXED")


if __name__ == "__main__":
    main()
