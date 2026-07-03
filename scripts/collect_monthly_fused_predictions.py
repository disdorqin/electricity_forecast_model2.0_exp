"""Collect daily fused_predictions.csv into a single monthly file — Phase 13.

Scans ``outputs/<date>/realtime/fused/fused_predictions.csv`` for every day
in a given month, concatenates them, and writes a unified monthly CSV with
an added ``source_date`` column indicating which day each row came from.

Usage::

    python scripts/collect_monthly_fused_predictions.py \
        --month 2026-02 \
        --output-dir reports/local/phase13/real_mainline_replay
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
PROJECT_ROOT = SCRIPT_DIR.parent                      # repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    import numpy as np
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return str(obj)


def iter_days_in_month(year: int, month: int) -> List[str]:
    """Return a list of ``YYYY-MM-DD`` strings for every day in *year*-*month*."""
    import calendar
    n_days = calendar.monthrange(year, month)[1]
    return [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, n_days + 1)]


def find_outputs_root(project_root: Path) -> Path:
    """Locate the ``outputs/`` directory under *project_root*."""
    candidate = project_root / "outputs"
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Cannot find outputs/ directory under {project_root}. "
        "Please run from the project root or adjust --project-root."
    )


def collect_daily_fused(
    outputs_root: Path,
    date_str: str,
) -> Optional[pd.DataFrame]:
    """Try to load ``outputs/<date>/realtime/fused/fused_predictions.csv``.

    Returns the DataFrame with a ``source_date`` column, or *None* if the
    file does not exist.
    """
    fused_path = (
        outputs_root / date_str / "realtime" / "fused" / "fused_predictions.csv"
    )
    if not fused_path.is_file():
        logger.debug("No fused_predictions.csv for %s (%s)", date_str, fused_path)
        return None

    try:
        df = pd.read_csv(str(fused_path), encoding="utf-8-sig")
    except Exception as exc:
        logger.warning("Failed to read %s: %s", fused_path, exc)
        return None

    df["source_date"] = date_str
    logger.info("  %s: loaded %d rows from %s", date_str, len(df), fused_path)
    return df


# ---------------------------------------------------------------------------
# Main collection logic
# ---------------------------------------------------------------------------

def collect_monthly_fused(
    month: str,
    output_dir: str,
    project_root: Optional[str] = None,
) -> dict:
    """Collect daily fused predictions for *month* (``YYYY-MM``) into one CSV.

    Returns a summary dict with ``days_found``, ``total_rows``, ``date_range``.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    outputs_root = find_outputs_root(root)

    year, mon = map(int, month.split("-"))
    days = iter_days_in_month(year, mon)
    logger.info("Collecting fused predictions for %s (%d days) from %s",
                month, len(days), outputs_root)

    frames: List[pd.DataFrame] = []
    dates_found: List[str] = []

    for date_str in days:
        df = collect_daily_fused(outputs_root, date_str)
        if df is not None and len(df) > 0:
            frames.append(df)
            dates_found.append(date_str)

    # ---- Concatenate ----
    if not frames:
        logger.warning("No fused_predictions.csv found for any day in %s", month)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        empty = pd.DataFrame()
        empty.to_csv(out / "monthly_fused_predictions.csv",
                     index=False, encoding="utf-8-sig")
        return {
            "days_found": 0,
            "total_rows": 0,
            "date_range": None,
            "output_path": str(out / "monthly_fused_predictions.csv"),
        }

    monthly = pd.concat(frames, ignore_index=True)

    # ---- Write output ----
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "monthly_fused_predictions.csv"
    monthly.to_csv(str(out_path), index=False, encoding="utf-8-sig")
    logger.info("Monthly fused predictions written to %s (%d rows)", out_path, len(monthly))

    # ---- Summary ----
    date_min = min(dates_found)
    date_max = max(dates_found)
    summary = {
        "days_found": len(dates_found),
        "total_rows": len(monthly),
        "date_range": f"{date_min} .. {date_max}",
        "output_path": str(out_path),
    }

    logger.info("=" * 60)
    logger.info("Collection summary:")
    logger.info("  Days found  : %d / %d", len(dates_found), len(days))
    logger.info("  Total rows  : %d", len(monthly))
    logger.info("  Date range  : %s .. %s", date_min, date_max)
    logger.info("  Output      : %s", out_path)
    logger.info("=" * 60)

    # Write a small summary JSON for downstream scripts
    import json
    summary_path = out / "collection_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=_json_default)
    logger.info("Collection summary written to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 13 — Collect monthly fused predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--month", required=True,
        help="Target month in YYYY-MM format (e.g. 2026-02).",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--project-root", default=None,
        help="Override project root (default: auto-detect from script location).",
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
    logger.info("Phase 13 — Collect Monthly Fused Predictions")
    logger.info("=" * 60)
    logger.info("Month       : %s", args.month)
    logger.info("Output dir  : %s", args.output_dir)
    logger.info("Project root: %s", args.project_root or "(auto)")

    summary = collect_monthly_fused(
        month=args.month,
        output_dir=args.output_dir,
        project_root=args.project_root,
    )

    if summary["days_found"] == 0:
        logger.warning("No data found for %s. Check outputs/ directory.", args.month)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
