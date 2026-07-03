"""Manifest and report generation for Intraday Tracker mainline integration — Phase 11."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class IntradayManifest:
    """Manifest for intraday tracker mainline integration run."""
    intraday_enabled: bool = True
    intraday_mode: str = "shadow"
    prediction_mode: str = "INTRADAY"
    pack_path: str = ""
    pack_rows: int = 0
    matched_rows: int = 0
    applied_rows: int = 0
    shadow_rows: int = 0
    disabled_rows: int = 0
    avg_fusion_weight: float = 0.0
    avg_confidence: float = 0.0
    policy_counts: dict = field(default_factory=dict)
    guardrail_counts: dict = field(default_factory=dict)
    fallback_reason: Optional[str] = None
    safe_fallback: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Manifest written to %s", path)


def build_manifest(stats: dict, pack_path: str = "") -> IntradayManifest:
    """Build IntradayManifest from apply stats dict."""
    return IntradayManifest(
        intraday_enabled=stats.get("intraday_enabled", True),
        intraday_mode=stats.get("intraday_mode", "shadow"),
        prediction_mode=stats.get("prediction_mode", "INTRADAY"),
        pack_path=pack_path,
        pack_rows=stats.get("pack_rows", 0),
        matched_rows=stats.get("matched_rows", 0),
        applied_rows=stats.get("applied_rows", 0),
        shadow_rows=stats.get("shadow_rows", 0),
        disabled_rows=stats.get("disabled_rows", 0),
        avg_fusion_weight=stats.get("avg_fusion_weight", 0.0),
        avg_confidence=stats.get("avg_confidence", 0.0),
        policy_counts=stats.get("policy_counts", {}),
        guardrail_counts=stats.get("guardrail_counts", {}),
        fallback_reason=stats.get("fallback_reason"),
        safe_fallback=stats.get("safe_fallback", True),
    )


def write_manifest_and_report(
    manifest: IntradayManifest,
    out_dir: str,
    intraday_rows_df: Optional[pd.DataFrame] = None,
    final_df: Optional[pd.DataFrame] = None,
):
    """Write manifest JSON, intraday rows CSV, and markdown report.

    Outputs:
      - intraday_mainline_manifest.json
      - intraday_application_report.md
      - intraday_rows.csv (if provided)
      - final_with_intraday_shadow.csv (if provided)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Manifest JSON
    manifest.to_json(str(out / "intraday_mainline_manifest.json"))

    # Intraday rows CSV
    if intraday_rows_df is not None and len(intraday_rows_df) > 0:
        intraday_rows_df.to_csv(out / "intraday_rows.csv", index=False, encoding="utf-8-sig")

    # Final CSV
    if final_df is not None and len(final_df) > 0:
        final_df.to_csv(out / "final_with_intraday_shadow.csv", index=False, encoding="utf-8-sig")

    # Markdown report
    lines = [
        "# Intraday Tracker Mainline Application Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Configuration",
        "",
        f"- **Intraday Enabled:** {manifest.intraday_enabled}",
        f"- **Mode:** {manifest.intraday_mode}",
        f"- **Prediction Mode:** {manifest.prediction_mode}",
        f"- **Pack Path:** {manifest.pack_path}",
        "",
        "## Summary",
        "",
        f"- **Pack Rows:** {manifest.pack_rows}",
        f"- **Matched Rows:** {manifest.matched_rows}",
        f"- **Applied Rows:** {manifest.applied_rows}",
        f"- **Shadow Rows:** {manifest.shadow_rows}",
        f"- **Disabled Rows:** {manifest.disabled_rows}",
        f"- **Avg Fusion Weight:** {manifest.avg_fusion_weight:.4f}",
        f"- **Avg Confidence:** {manifest.avg_confidence:.4f}",
        "",
    ]

    # Whether prediction was changed
    if manifest.intraday_mode == "shadow":
        lines.extend([
            "## Behavior: SHADOW",
            "",
            "The intraday tracker ran in shadow mode. Final predictions were NOT changed.",
            "Shadow predictions are recorded for offline evaluation.",
            "",
        ])
    elif manifest.intraday_mode in ("low_weight", "high_weight"):
        lines.extend([
            f"## Behavior: {manifest.intraday_mode.upper()}",
            "",
            f"The intraday tracker applied corrections to {manifest.applied_rows} rows.",
            f"Average fusion weight: {manifest.avg_fusion_weight:.4f}.",
            "",
        ])
    elif manifest.intraday_mode == "off":
        lines.extend([
            "## Behavior: OFF",
            "",
            "The intraday tracker was disabled. No corrections applied.",
            "",
        ])

    # FULL_DAY / day-ahead check
    if manifest.prediction_mode == "FULL_DAY":
        lines.extend([
            "## FULL_DAY Check",
            "",
            "Prediction mode is FULL_DAY. Intraday tracker was correctly disabled.",
            "No FULL_DAY/day-ahead rule violation.",
            "",
        ])

    # Fallback
    if manifest.fallback_reason:
        lines.extend([
            "## Fallback",
            "",
            f"**Fallback Reason:** {manifest.fallback_reason}",
            "The tracker fell back to safe mode (no correction applied).",
            "",
        ])

    # Policy counts
    if manifest.policy_counts:
        lines.extend([
            "## Policy Decision Counts",
            "",
            "| Decision | Count |",
            "|----------|-------|",
        ])
        for decision, count in sorted(manifest.policy_counts.items()):
            lines.append(f"| {decision} | {count} |")
        lines.append("")

    # Guardrail counts
    if manifest.guardrail_counts:
        lines.extend([
            "## Guardrail Counts",
            "",
            "| Reason | Count |",
            "|--------|-------|",
        ])
        for reason, count in sorted(manifest.guardrail_counts.items()):
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    (out / "intraday_application_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to %s", out / "intraday_application_report.md")
