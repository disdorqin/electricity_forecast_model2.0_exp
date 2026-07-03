# -*- coding: utf-8 -*-
"""Report generation for the residual stack evaluation.

Writes structured JSON and Markdown reports that include:
    - Stack configuration
    - Metrics for each combination (A/B/C/D)
    - GO / NO-GO / DATA-LIMITED verdict
    - Interaction summary (high_spike blocked how many negative corrections)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd


def generate_verdict(metrics: dict[str, Any], run_status: str = "official") -> str:
    """Determine GO / NO-GO / DATA-MISSING based on GO conditions.

    Parameters
    ----------
    run_status : str
        ``"official"`` → full GO/NO-GO evaluation.
        ``"dry_run"`` → same evaluation but will be prefixed ``[dry-run]``.
        ``"data_missing"`` → immediate return ``DATA-MISSING`` (no evaluation).

    GO conditions (from spec):
        1. overall_sMAPE improvement >= -0.3 (i.e. at most 0.3 worse)
        2. severe <= 63 或不恶化
        3. high_spike_MAE improvement >= -3.0% (i.e. at most 3% worse)
        4. low_valley_MAE improvement >= 0 (must not worsen)
        5. normal_degradation <= 0.5
    """
    if run_status == "data_missing":
        return "DATA-MISSING"

    if metrics.get("data_limited", False):
        return "DATA-LIMITED"

    reasons: list[str] = []

    # Condition 1: sMAPE
    smape_improvement = metrics.get("overall_sMAPE_improvement", 0)
    if smape_improvement < -0.3:
        reasons.append(f"sMAPE worse by {abs(smape_improvement):.2f} (limit 0.3)")

    # Condition 2: severe
    severe = metrics.get("severe_underestimate", 0)
    severe_before = metrics.get("severe_underestimate_before", severe)
    if severe > 63 and severe > severe_before:
        reasons.append(f"severe {severe} > 63 and worsened")

    # Condition 3: high_spike MAE
    hs_improvement = metrics.get("high_spike_MAE_improvement", 0)
    if hs_improvement < -3.0:
        reasons.append(f"high_spike_MAE worse by {abs(hs_improvement):.1f}% (limit 3%)")

    # Condition 4: low_valley_MAE must not worsen
    lv_improvement = metrics.get("low_valley_MAE_improvement", 0)
    if lv_improvement < 0:
        reasons.append(f"low_valley_MAE worse by {abs(lv_improvement):.2f}%")

    # Condition 5: normal_degradation
    normal_degradation = metrics.get("normal_degradation", 0)
    if normal_degradation > 0.5:
        reasons.append(f"normal_degradation {normal_degradation:.2f} > 0.5")

    if reasons:
        return f"NO-GO: {'; '.join(reasons)}"

    return "GO"


def write_report(
    out_dir: str | Path,
    config_results: dict[str, dict[str, Any]],
    interaction_summary: Optional[dict[str, Any]] = None,
    description: str = "",
) -> Path:
    """Write a risk source-aware structured JSON report.

    Each config's verdict is prefixed with its run status:

        ``[official] GO``          — real/calibrated spike risk, all GO conditions met
        ``[official] NO-GO``       — real/calibrated spike risk, condition(s) failed
        ``[official] DATA-LIMITED`` — too few negative samples (still official)
        ``[dry-run] ...``          — synthetic risk data, informative only
        ``[data-missing] DATA-MISSING`` — no spike risk data (configs B/D skip)

    Overall verdict is computed from **official** results only.

    Parameters
    ----------
    out_dir : str | Path
        Output directory.
    config_results : dict[str, dict[str, Any]]
        Mapping of config label (e.g. "A", "B") → metrics dict.
        Each metrics dict may contain:
        - ``_risk_source`` (str): raw RiskSource value.
        - ``_run_status`` (str): ``official`` / ``dry_run`` / ``data_missing``.
        - ``_allow_synthetic`` (bool): whether synthetic risk was allowed.
    interaction_summary : dict | None
        Optional interaction statistics (e.g. high_spike blocked counts).
    description : str
        Optional description / notes.

    Returns
    -------
    Path
        Path to the written JSON report.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    verdicts: dict[str, str] = {}
    official_raw: list[str] = []

    for label, metrics in config_results.items():
        run_status = metrics.get("_run_status", "official")
        raw = generate_verdict(metrics, run_status=run_status)

        if run_status == "official":
            verdicts[label] = f"[official] {raw}"
            official_raw.append(raw)
        elif run_status == "dry_run":
            verdicts[label] = f"[dry-run] {raw}"
        else:
            verdicts[label] = "[data-missing] DATA-MISSING"

    report: dict[str, Any] = {
        "report_type": "residual_stack_evaluation",
        "description": description,
        "verdicts": verdicts,
        "risk_source_policy": {
            label: {
                "risk_source": metrics.get("_risk_source", "unknown"),
                "run_status": metrics.get("_run_status", "unknown"),
                "allow_synthetic": metrics.get("_allow_synthetic", False),
            }
            for label, metrics in config_results.items()
        },
        "configurations": config_results,
    }

    if interaction_summary:
        report["interaction_summary"] = interaction_summary

    # Overall verdict — official results only
    if official_raw:
        if all(v == "GO" for v in official_raw):
            report["overall_verdict"] = "GO"
        elif any("DATA-MISSING" in v or "DATA-LIMITED" in v for v in official_raw):
            report["overall_verdict"] = "DATA-MISSING"
        elif any(v.startswith("NO-GO") for v in official_raw):
            report["overall_verdict"] = "NO-GO"
        else:
            report["overall_verdict"] = "MIXED"
    else:
        report["overall_verdict"] = "NO-OFFICIAL-RESULTS"

    report_path = out_dir / "residual_stack_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path
