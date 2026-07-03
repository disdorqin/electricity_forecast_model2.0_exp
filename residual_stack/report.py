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


def generate_verdict(metrics: dict[str, Any]) -> str:
    """Determine GO / NO-GO / DATA-LIMITED based on GO conditions.

    GO conditions (from spec):
        1. overall_sMAPE 不恶化 > 0.3 (delta <= 0.3)
        2. severe <= 63 或不恶化
        3. high_spike_MAE 不恶化 > 3%
        4. low_valley_MAE 改善 (delta < 0)
        5. normal_degradation <= 0.5
    """
    if metrics.get("data_limited", False):
        return "DATA-LIMITED"

    reasons: list[str] = []

    # Condition 1: sMAPE
    smape_delta = metrics.get("overall_sMAPE_delta", 0)
    if smape_delta > 0.3:
        reasons.append(f"sMAPE worsened by {smape_delta:.2f} (max 0.3)")

    # Condition 2: severe
    severe = metrics.get("severe_underestimate", 0)
    severe_before = metrics.get("severe_underestimate_before", severe)
    if severe > 63 and severe > severe_before:
        reasons.append(f"severe {severe} > 63 and worsened")

    # Condition 3: high_spike MAE
    hs_delta = metrics.get("high_spike_MAE_delta_pct", 0)
    if hs_delta > 3.0:
        reasons.append(f"high_spike_MAE worsened by {hs_delta:.1f}% (> 3%)")

    # Condition 4: low_valley_MAE
    lv_delta = metrics.get("low_valley_MAE_delta", 0)
    if lv_delta >= 0:
        reasons.append(f"low_valley_MAE not improved (delta={lv_delta})")

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
    """Write a structured JSON report and return the path.

    Parameters
    ----------
    out_dir : str | Path
        Output directory.
    config_results : dict[str, dict[str, Any]]
        Mapping of config label (e.g. "A", "B") → metrics dict.
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

    verdicts = {
        label: generate_verdict(metrics)
        for label, metrics in config_results.items()
    }

    report: dict[str, Any] = {
        "report_type": "residual_stack_evaluation",
        "description": description,
        "verdicts": verdicts,
        "configurations": config_results,
    }

    if interaction_summary:
        report["interaction_summary"] = interaction_summary

    # Overall verdict
    if all(v == "GO" for v in verdicts.values()):
        report["overall_verdict"] = "GO"
    elif any("DATA-LIMITED" in v for v in verdicts.values()):
        report["overall_verdict"] = "DATA-LIMITED"
    elif any(v.startswith("NO-GO") for v in verdicts.values()):
        report["overall_verdict"] = "NO-GO"
    else:
        report["overall_verdict"] = "MIXED"

    report_path = out_dir / "residual_stack_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report_path
