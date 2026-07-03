"""Align an intraday correction pack to real fused predictions — Phase 13.

Loads a monthly fused predictions CSV and an intraday correction pack,
matches them on ``(business_day, hour_business / target_hour)``, deduplicates
the pack by keeping the highest ``cutoff_hour`` per key, and reports alignment
statistics.

Usage::

    python scripts/align_intraday_pack_to_mainline.py \
        --fused-predictions reports/local/phase13/real_mainline_replay/monthly_fused_predictions.csv \
        --intraday-pack     path/to/intraday_pack.csv \
        --output-dir        reports/local/phase13/real_mainline_replay
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup — ensure the corrections package is importable
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                      # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MAIN_REPO_CANDIDATE = PROJECT_ROOT.parent / "electricity_forecast_model2.0_exp"
if _MAIN_REPO_CANDIDATE.is_dir() and str(_MAIN_REPO_CANDIDATE) not in sys.path:
    sys.path.insert(0, str(_MAIN_REPO_CANDIDATE))

from corrections.intraday_tracker.adapter import (
    load_intraday_pack,
    normalize_intraday_pack,
    validate_intraday_pack,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "reports/local/phase13/real_mainline_replay"


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
# Data loading
# ---------------------------------------------------------------------------

def load_fused_predictions(path: str) -> pd.DataFrame:
    """Load monthly fused predictions CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    logger.info("Fused predictions loaded: %d rows, columns=%s", len(df), list(df.columns))

    for col in ("business_day", "hour_business"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")

    df["business_day"] = pd.to_datetime(df["business_day"], errors="coerce")
    df["hour_business"] = pd.to_numeric(df["hour_business"], errors="coerce").astype("Int64")
    return df


def load_and_prepare_pack(pack_path: str) -> pd.DataFrame:
    """Load, normalise, and validate the intraday pack (no deduplication yet)."""
    raw = load_intraday_pack(pack_path)
    pack = normalize_intraday_pack(raw, source_pack_path=pack_path)
    validation = validate_intraday_pack(pack, mode="offline")
    if not validation.valid:
        logger.warning("Intraday pack validation issues: %s", validation.errors)
    logger.info("Intraday pack loaded: %d rows", len(pack))
    return pack


# ---------------------------------------------------------------------------
# Alignment logic
# ---------------------------------------------------------------------------

def deduplicate_pack(pack: pd.DataFrame) -> tuple:
    """Deduplicate pack by keeping highest cutoff_hour per (business_day, target_hour).

    Returns (deduped_pack, n_duplicates_removed).
    """
    if "business_day" not in pack.columns or "target_hour" not in pack.columns:
        logger.warning("Pack missing business_day or target_hour — cannot deduplicate.")
        return pack, 0

    n_before = len(pack)

    # Count duplicates
    dupes = pack.groupby(["business_day", "target_hour"]).size()
    n_dup_groups = int((dupes > 1).sum())

    if n_dup_groups == 0:
        logger.info("Pack has no duplicate (business_day, target_hour) groups.")
        return pack, 0

    logger.info(
        "Pack has %d duplicate (business_day, target_hour) groups "
        "(%d total rows). Deduplicating by keeping highest cutoff_hour.",
        n_dup_groups, n_before,
    )

    if "cutoff_hour" in pack.columns:
        pack = pack.sort_values("cutoff_hour", ascending=False)

    pack = pack.drop_duplicates(
        subset=["business_day", "target_hour"], keep="first",
    )
    n_removed = n_before - len(pack)
    logger.info("After deduplication: %d rows (removed %d)", len(pack), n_removed)
    return pack, n_removed


def align_pack_to_fused(
    fused: pd.DataFrame,
    pack: pd.DataFrame,
) -> dict:
    """Align the intraday pack to fused predictions.

    Returns a dict with alignment statistics and the aligned pack DataFrame.
    """
    # Build the fused key set
    fused_keys = set(
        zip(
            fused["business_day"].dt.strftime("%Y-%m-%d"),
            fused["hour_business"].astype(int),
        )
    )

    # Build the pack key set
    if "target_hour" in pack.columns:
        pack_hour_col = "target_hour"
    elif "hour_business" in pack.columns:
        pack_hour_col = "hour_business"
    else:
        raise ValueError("Pack has neither 'target_hour' nor 'hour_business' column.")

    pack["business_day"] = pd.to_datetime(pack["business_day"], errors="coerce")
    pack_keys = set(
        zip(
            pack["business_day"].dt.strftime("%Y-%m-%d"),
            pack[pack_hour_col].astype(int),
        )
    )

    matched_keys = fused_keys & pack_keys
    missing_pack_keys = fused_keys - pack_keys
    extra_pack_keys = pack_keys - fused_keys

    matched_rows = len(matched_keys)
    missing_pack_rows = len(missing_pack_keys)

    # Filter pack to only matched keys
    pack["_bd_str"] = pack["business_day"].dt.strftime("%Y-%m-%d")
    pack["_hour_key"] = pack[pack_hour_col].astype(int)
    pack["_key"] = list(zip(pack["_bd_str"], pack["_hour_key"]))

    aligned_pack = pack[pack["_key"].isin(matched_keys)].copy()
    aligned_pack = aligned_pack.drop(columns=["_bd_str", "_hour_key", "_key"])

    # ---- Distribution statistics ----
    cutoff_dist: Dict[str, int] = {}
    if "cutoff_hour" in aligned_pack.columns:
        cutoff_counts = aligned_pack["cutoff_hour"].value_counts().sort_index()
        cutoff_dist = {str(k): int(v) for k, v in cutoff_counts.items()}

    policy_dist: Dict[str, int] = {}
    if "policy_decision" in aligned_pack.columns:
        policy_counts = aligned_pack["policy_decision"].value_counts()
        policy_dist = {str(k): int(v) for k, v in policy_counts.items()}

    confidence_dist: Dict[str, float] = {}
    if "intraday_confidence" in aligned_pack.columns:
        conf = aligned_pack["intraday_confidence"].dropna()
        if len(conf) > 0:
            confidence_dist = {
                "mean": float(conf.mean()),
                "median": float(conf.median()),
                "min": float(conf.min()),
                "max": float(conf.max()),
                "std": float(conf.std()) if len(conf) > 1 else 0.0,
            }

    return {
        "matched_rows": matched_rows,
        "missing_pack_rows": missing_pack_rows,
        "extra_pack_rows": len(extra_pack_keys),
        "aligned_pack_size": len(aligned_pack),
        "cutoff_distribution": cutoff_dist,
        "policy_distribution": policy_dist,
        "confidence_distribution": confidence_dist,
        "aligned_pack": aligned_pack,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_alignment_report(
    alignment: dict,
    n_duplicates: int,
    fused_path: str,
    pack_path: str,
    out_dir: Path,
) -> None:
    """Write alignment_report.md."""
    lines: List[str] = [
        "# Phase 13 — Intraday Pack Alignment Report",
        "",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Inputs",
        "",
        f"- **Fused predictions:** `{fused_path}`",
        f"- **Intraday pack:** `{pack_path}`",
        "",
        "## Alignment Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Matched rows | {alignment['matched_rows']} |",
        f"| Missing pack rows | {alignment['missing_pack_rows']} |",
        f"| Extra pack rows (no fused match) | {alignment['extra_pack_rows']} |",
        f"| Duplicate rows removed | {n_duplicates} |",
        f"| Aligned pack size | {alignment['aligned_pack_size']} |",
        "",
    ]

    # Cutoff distribution
    cutoff_dist = alignment.get("cutoff_distribution", {})
    if cutoff_dist:
        lines += [
            "## Cutoff Hour Distribution",
            "",
            "| Cutoff Hour | Count |",
            "|-------------|-------|",
        ]
        for cutoff, count in sorted(cutoff_dist.items()):
            lines.append(f"| {cutoff} | {count} |")
        lines.append("")
    else:
        lines += [
            "## Cutoff Hour Distribution",
            "",
            "No cutoff_hour data available.",
            "",
        ]

    # Policy distribution
    policy_dist = alignment.get("policy_distribution", {})
    if policy_dist:
        lines += [
            "## Policy Decision Distribution",
            "",
            "| Policy Decision | Count |",
            "|-----------------|-------|",
        ]
        for policy, count in sorted(policy_dist.items()):
            lines.append(f"| {policy} | {count} |")
        lines.append("")
    else:
        lines += [
            "## Policy Decision Distribution",
            "",
            "No policy_decision data available.",
            "",
        ]

    # Confidence distribution
    conf_dist = alignment.get("confidence_distribution", {})
    if conf_dist:
        lines += [
            "## Confidence Distribution",
            "",
            "| Statistic | Value |",
            "|-----------|-------|",
        ]
        for stat, val in conf_dist.items():
            lines.append(f"| {stat} | {val:.4f} |")
        lines.append("")
    else:
        lines += [
            "## Confidence Distribution",
            "",
            "No intraday_confidence data available.",
            "",
        ]

    lines += [
        "## Output Files",
        "",
        "- `aligned_pack.csv` — Deduplicated and aligned intraday pack",
        "- `alignment_report.md` — This report",
        "",
        "---",
        f"*Generated by scripts/align_intraday_pack_to_mainline.py at "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ]

    path = out_dir / "alignment_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Alignment report written to %s", path)


# ---------------------------------------------------------------------------
# Main alignment pipeline
# ---------------------------------------------------------------------------

def align(
    fused_predictions_path: str,
    intraday_pack_path: str,
    output_dir: str,
) -> dict:
    """Run the full alignment pipeline.

    Returns a summary dict.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    logger.info("Loading fused predictions from %s", fused_predictions_path)
    fused = load_fused_predictions(fused_predictions_path)

    logger.info("Loading intraday pack from %s", intraday_pack_path)
    pack = load_and_prepare_pack(intraday_pack_path)

    # ---- Deduplicate ----
    pack, n_duplicates = deduplicate_pack(pack)

    # ---- Align ----
    logger.info("Aligning pack to fused predictions...")
    alignment = align_pack_to_fused(fused, pack)

    # ---- Write outputs ----
    aligned_pack = alignment.pop("aligned_pack")
    aligned_path = out / "aligned_pack.csv"
    aligned_pack.to_csv(str(aligned_path), index=False, encoding="utf-8-sig")
    logger.info("Aligned pack written to %s (%d rows)", aligned_path, len(aligned_pack))

    write_alignment_report(
        alignment=alignment,
        n_duplicates=n_duplicates,
        fused_path=fused_predictions_path,
        pack_path=intraday_pack_path,
        out_dir=out,
    )

    # Write alignment summary JSON
    summary = {
        "alignment_timestamp": datetime.now().isoformat(timespec="seconds"),
        "fused_predictions_path": fused_predictions_path,
        "intraday_pack_path": intraday_pack_path,
        **alignment,
        "duplicate_rows_removed": n_duplicates,
    }
    summary_path = out / "alignment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Alignment summary written to %s", summary_path)

    # ---- Log summary ----
    logger.info("=" * 60)
    logger.info("Alignment summary:")
    logger.info("  Matched rows       : %d", alignment["matched_rows"])
    logger.info("  Missing pack rows  : %d", alignment["missing_pack_rows"])
    logger.info("  Extra pack rows    : %d", alignment["extra_pack_rows"])
    logger.info("  Duplicates removed : %d", n_duplicates)
    logger.info("  Aligned pack size  : %d", alignment["aligned_pack_size"])
    logger.info("=" * 60)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 — Align Intraday Pack to Mainline Fused Predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fused-predictions", required=True,
        help="Path to monthly fused predictions CSV.",
    )
    parser.add_argument(
        "--intraday-pack", required=True,
        help="Path to intraday correction pack CSV.",
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
    logger.info("Phase 13 — Align Intraday Pack to Mainline")
    logger.info("=" * 60)
    logger.info("Fused predictions : %s", args.fused_predictions)
    logger.info("Intraday pack     : %s", args.intraday_pack)
    logger.info("Output dir        : %s", args.output_dir)

    # Validate inputs exist
    for label, path_str in [
        ("fused-predictions", args.fused_predictions),
        ("intraday-pack", args.intraday_pack),
    ]:
        if not Path(path_str).is_file():
            logger.error("Input file not found: %s (%s)", label, path_str)
            return 1

    align(
        fused_predictions_path=args.fused_predictions,
        intraday_pack_path=args.intraday_pack,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
