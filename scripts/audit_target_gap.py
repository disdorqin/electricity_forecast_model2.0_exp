"""Audit gap between current performance and targets — Phase 13.

Reads a ``metrics_summary.json`` (produced by the shadow replay or mainline
evaluation scripts), extracts the per-mode sMAPE values, and computes the gap
to the business targets (monthly average realtime sMAPE_floor50 < 15 and < 20).

Also computes the 9_16 segment gap if per-segment data is available.

Usage::

    python scripts/audit_target_gap.py \
        --metrics-summary reports/local/phase12/intraday_shadow_replay/replay_metrics_summary.json \
        --output-dir      reports/local/phase13/real_mainline_replay
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "reports/local/phase13/real_mainline_replay"

# Business targets (sMAPE_floor50 percentage)
TARGET_15 = 15.0   # stretch target: monthly avg < 15%
TARGET_20 = 20.0   # acceptable target: monthly avg < 20%

# Reference values (current best)
REFERENCE_OVERALL = 16.59
REFERENCE_9_16 = 21.19


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialiser fallback for numpy / pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Core audit logic
# ---------------------------------------------------------------------------

def load_metrics_summary(path: str) -> dict:
    """Load metrics_summary.json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Metrics summary loaded from %s", path)
    return data


def extract_smape_values(summary: dict) -> Dict[str, Optional[float]]:
    """Extract sMAPE values from the metrics summary.

    Handles multiple formats:
    - Top-level keys: baseline_sMAPE, low_weight_sMAPE, high_weight_sMAPE
    - Nested per_mode_metrics: {mode: {overall_smape: ...}}
    - Nested metrics: {baseline_sMAPE: ...}
    """
    result: Dict[str, Optional[float]] = {
        "baseline_sMAPE": None,
        "low_weight_sMAPE": None,
        "high_weight_sMAPE": None,
        "shadow_sMAPE": None,
    }

    # Try direct top-level keys first
    for key in result:
        if key in summary:
            val = summary[key]
            if isinstance(val, (int, float)):
                result[key] = float(val)

    # Try nested "metrics" key
    if "metrics" in summary and isinstance(summary["metrics"], dict):
        metrics = summary["metrics"]
        for key in result:
            if key in metrics and result[key] is None:
                val = metrics[key]
                if isinstance(val, (int, float)):
                    result[key] = float(val)

    # Try per_mode_metrics
    if "per_mode_metrics" in summary and isinstance(summary["per_mode_metrics"], dict):
        pmm = summary["per_mode_metrics"]
        mode_map = {
            "baseline": "baseline_sMAPE",
            "shadow": "shadow_sMAPE",
            "low_weight": "low_weight_sMAPE",
            "high_weight": "high_weight_sMAPE",
        }
        for mode_name, smape_key in mode_map.items():
            if mode_name in pmm and result[smape_key] is None:
                mode_data = pmm[mode_name]
                if isinstance(mode_data, dict):
                    val = mode_data.get("overall_smape", mode_data.get("smape_floor50"))
                    if isinstance(val, (int, float)):
                        result[smape_key] = float(val)

    return result


def extract_segment_smape(summary: dict) -> Dict[str, Optional[float]]:
    """Extract 9_16 segment sMAPE if available.

    Looks for keys like:
    - segment_9_16_sMAPE
    - per_segment -> 9_16 -> overall_smape
    - hourly breakdown that can be filtered to hours 9-16
    """
    result: Dict[str, Optional[float]] = {
        "baseline_9_16": None,
        "low_weight_9_16": None,
    }

    # Direct keys
    for key in result:
        if key in summary:
            val = summary[key]
            if isinstance(val, (int, float)):
                result[key] = float(val)

    # Nested per_segment
    if "per_segment" in summary and isinstance(summary["per_segment"], dict):
        seg = summary["per_segment"]
        if "9_16" in seg:
            seg_data = seg["9_16"]
            if isinstance(seg_data, dict):
                for mode_key in ("baseline", "low_weight"):
                    if mode_key in seg_data and result[f"{mode_key}_9_16"] is None:
                        val = seg_data[mode_key]
                        if isinstance(val, (int, float)):
                            result[f"{mode_key}_9_16"] = float(val)

    # Try from hourly metrics if segment not directly available
    # (This is a fallback — we would need hourly CSV for this)

    return result


def compute_gaps(smape_values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute gap to target 15 and target 20 for each mode.

    Gap is in percentage points: positive means above target (needs improvement),
    negative means below target (already meeting target).
    """
    gaps: Dict[str, Optional[float]] = {}

    for mode_key, smape_val in smape_values.items():
        if smape_val is None:
            gaps[f"{mode_key}_gap_to_15"] = None
            gaps[f"{mode_key}_gap_to_20"] = None
            continue

        # Convert from ratio (0-1) to percentage if needed
        pct = smape_val * 100 if smape_val < 1 else smape_val

        gaps[f"{mode_key}_gap_to_15"] = pct - TARGET_15
        gaps[f"{mode_key}_gap_to_20"] = pct - TARGET_20

    return gaps


def compute_segment_gaps(segment_values: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Compute gap to reference for 9_16 segment."""
    gaps: Dict[str, Optional[float]] = {}

    for key, val in segment_values.items():
        if val is None:
            gaps[f"{key}_gap_to_reference"] = None
            continue

        pct = val * 100 if val < 1 else val
        gaps[f"{key}_gap_to_reference"] = pct - REFERENCE_9_16

    return gaps


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_target_gap_report(
    smape_values: Dict[str, Optional[float]],
    gaps: Dict[str, Optional[float]],
    segment_values: Dict[str, Optional[float]],
    segment_gaps: Dict[str, Optional[float]],
    metrics_path: str,
    out_dir: Path,
) -> None:
    """Write target_gap_report.md."""

    def _fmt(val: Optional[float], unit: str = "pp") -> str:
        if val is None:
            return "N/A"
        return f"{val:.2f}{unit}"

    def _fmt_smape(val: Optional[float]) -> str:
        if val is None:
            return "N/A"
        pct = val * 100 if val < 1 else val
        return f"{pct:.2f}%"

    lines: List[str] = [
        "# Phase 13 — Target Gap Audit Report",
        "",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Source:** `{metrics_path}`",
        "",
        "## Business Targets",
        "",
        f"- **Stretch target:** monthly avg realtime sMAPE_floor50 < {TARGET_15:.0f}%",
        f"- **Acceptable target:** monthly avg realtime sMAPE_floor50 < {TARGET_20:.0f}%",
        f"- **Current best reference:** overall ~{REFERENCE_OVERALL:.2f}%, 9_16 ~{REFERENCE_9_16:.2f}%",
        "",
        "## Current Performance (sMAPE_floor50)",
        "",
        "| Mode | sMAPE | Gap to 15% | Gap to 20% |",
        "|------|-------|------------|------------|",
    ]

    mode_labels = {
        "baseline_sMAPE": "Baseline (no correction)",
        "shadow_sMAPE": "Shadow corrected",
        "low_weight_sMAPE": "Low-weight corrected",
        "high_weight_sMAPE": "High-weight corrected",
    }

    for mode_key, label in mode_labels.items():
        smape = smape_values.get(mode_key)
        gap15 = gaps.get(f"{mode_key}_gap_to_15")
        gap20 = gaps.get(f"{mode_key}_gap_to_20")

        gap15_str = _fmt(gap15) if gap15 is not None else "N/A"
        gap20_str = _fmt(gap20) if gap20 is not None else "N/A"

        # Add status indicator
        if gap15 is not None:
            if gap15 <= 0:
                gap15_str += " (MET)"
            elif gap15 <= 2:
                gap15_str += " (close)"
            else:
                gap15_str += " (far)"

        lines.append(f"| {label} | {_fmt_smape(smape)} | {gap15_str} | {gap20_str} |")

    lines.append("")

    # 9_16 segment section
    has_segment = any(v is not None for v in segment_values.values())
    if has_segment:
        lines += [
            "## 9_16 Segment Performance",
            "",
            "| Mode | sMAPE | Gap to Reference ({:.2f}%) |".format(REFERENCE_9_16),
            "|------|-------|---------------------------|",
        ]

        seg_labels = {
            "baseline_9_16": "Baseline",
            "low_weight_9_16": "Low-weight corrected",
        }

        for seg_key, label in seg_labels.items():
            val = segment_values.get(seg_key)
            gap = segment_gaps.get(f"{seg_key}_gap_to_reference")

            gap_str = _fmt(gap) if gap is not None else "N/A"
            if gap is not None:
                if gap <= 0:
                    gap_str += " (improved)"
                else:
                    gap_str += " (worse)"

            lines.append(f"| {label} | {_fmt_smape(val)} | {gap_str} |")

        lines.append("")
    else:
        lines += [
            "## 9_16 Segment Performance",
            "",
            "No 9_16 segment data available in the metrics summary.",
            "Run per-segment evaluation to populate this section.",
            "",
        ]

    # Verdict
    lines += [
        "## Assessment",
        "",
    ]

    baseline_smape = smape_values.get("baseline_sMAPE")
    low_weight_smape = smape_values.get("low_weight_sMAPE")

    if baseline_smape is not None:
        baseline_pct = baseline_smape * 100 if baseline_smape < 1 else baseline_smape
        if baseline_pct < TARGET_15:
            lines.append(f"- Baseline already meets the stretch target (< {TARGET_15:.0f}%).")
        elif baseline_pct < TARGET_20:
            lines.append(f"- Baseline meets the acceptable target (< {TARGET_20:.0f}%) but not stretch.")
        else:
            lines.append(f"- Baseline does NOT meet even the acceptable target ({baseline_pct:.2f}% >= {TARGET_20:.0f}%).")

    if low_weight_smape is not None and baseline_smape is not None:
        lw_pct = low_weight_smape * 100 if low_weight_smape < 1 else low_weight_smape
        bl_pct = baseline_smape * 100 if baseline_smape < 1 else baseline_smape
        gain = bl_pct - lw_pct
        if gain > 0:
            lines.append(f"- Low-weight correction improves by {gain:.2f} percentage points.")
        else:
            lines.append(f"- Low-weight correction shows no improvement ({gain:+.2f} pp).")

    lines += [
        "",
        "## Output Files",
        "",
        "- `target_gap_report.md` — This report",
        "- `target_gap_summary.json` — Machine-readable gap summary",
        "",
        "---",
        f"*Generated by scripts/audit_target_gap.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path = out_dir / "target_gap_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Target gap report written to %s", path)


# ---------------------------------------------------------------------------
# Main audit pipeline
# ---------------------------------------------------------------------------

def audit_target_gap(
    metrics_summary_path: str,
    output_dir: str,
) -> dict:
    """Run the full target gap audit.

    Returns a summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load metrics summary ----
    logger.info("Loading metrics summary from %s", metrics_summary_path)
    summary = load_metrics_summary(metrics_summary_path)

    # ---- Extract sMAPE values ----
    smape_values = extract_smape_values(summary)
    logger.info("Extracted sMAPE values: %s", smape_values)

    # ---- Compute gaps ----
    gaps = compute_gaps(smape_values)
    logger.info("Computed gaps: %s", {k: v for k, v in gaps.items() if v is not None})

    # ---- Extract segment sMAPE ----
    segment_values = extract_segment_smape(summary)
    segment_gaps = compute_segment_gaps(segment_values)

    # ---- Write report ----
    write_target_gap_report(
        smape_values=smape_values,
        gaps=gaps,
        segment_values=segment_values,
        segment_gaps=segment_gaps,
        metrics_path=metrics_summary_path,
        out_dir=out,
    )

    # ---- Write JSON summary ----
    audit_summary = {
        "audit_timestamp": datetime.now().isoformat(timespec="seconds"),
        "metrics_summary_path": metrics_summary_path,
        "targets": {
            "stretch_target": TARGET_15,
            "acceptable_target": TARGET_20,
            "reference_overall": REFERENCE_OVERALL,
            "reference_9_16": REFERENCE_9_16,
        },
        "current_smape": smape_values,
        "gaps": gaps,
        "segment_smape": segment_values,
        "segment_gaps": segment_gaps,
    }

    summary_path = out / "target_gap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Target gap summary written to %s", summary_path)

    # ---- Log summary ----
    logger.info("=" * 60)
    logger.info("Target Gap Audit:")
    for mode_key, smape_val in smape_values.items():
        if smape_val is not None:
            pct = smape_val * 100 if smape_val < 1 else smape_val
            gap15 = gaps.get(f"{mode_key}_gap_to_15")
            gap15_str = f"{gap15:+.2f}pp" if gap15 is not None else "N/A"
            logger.info("  %-25s  sMAPE=%.2f%%  gap_to_15=%s", mode_key, pct, gap15_str)
    logger.info("=" * 60)

    return audit_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 — Audit Target Gap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--metrics-summary", required=True,
        help="Path to metrics_summary.json (from shadow replay or mainline evaluation).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args(argv)

    logger.info("=" * 60)
    logger.info("Phase 13 — Audit Target Gap")
    logger.info("=" * 60)
    logger.info("Metrics summary : %s", args.metrics_summary)
    logger.info("Output dir      : %s", args.output_dir)

    if not Path(args.metrics_summary).is_file():
        logger.error("Metrics summary not found: %s", args.metrics_summary)
        return 1

    audit_target_gap(
        metrics_summary_path=args.metrics_summary,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
