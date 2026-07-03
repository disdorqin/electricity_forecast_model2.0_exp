#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5M Plugin Interface — Dry-run smoke test for external prediction integration.

Usage:
    python scripts/run_p5m_plugin_pipeline_smoke.py \\
        --prediction-pack /path/to/predictions.csv \\
        --external-predictions /path/to/other_preds.csv \\
        --correction-modules spike_correction negative_correction \\
        --out-dir outputs/plugin_smoke

This script exercises the plugin interface layer (pipeline_ext/) without
touching production_pipeline.py.  It is intended for smoke-testing
external model CSV integration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from pipeline_ext.io import load_prediction_csv
from pipeline_ext.modules import CorrectionModule, MonitorModule, PredictionProvider
from pipeline_ext.registry import (
    register_correction_module,
    register_monitor_module,
    register_prediction_provider,
)
from pipeline_ext.pipeline import DryRunPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("p5m_plugin_smoke")


# ── Example inline modules (for smoke-testing) ─────────────────────────


class IdentityCorrection(CorrectionModule):
    """A no-op correction — passes the DataFrame through unchanged."""

    name = "identity"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("IdentityCorrection: passing %d rows through", len(df))
        return df.copy()


class ClampCorrection(CorrectionModule):
    """Clamp y_pred to a minimum value (demonstrates a real correction)."""

    name = "clamp_low"

    def __init__(self, min_val: float = -500.0):
        self.min_val = min_val

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("ClampCorrection: clamping y_pred to min=%.1f", self.min_val)
        out = df.copy()
        out["y_pred"] = out["y_pred"].clip(lower=self.min_val)
        return out


class RowCountMonitor(MonitorModule):
    """Simply counts rows and reports basic stats."""

    name = "row_count"

    def run(self, df: pd.DataFrame) -> dict:
        return {
            "row_count": len(df),
            "columns": list(df.columns),
            "has_y_pred": "y_pred" in df.columns,
        }


class NullCheckMonitor(MonitorModule):
    """Check for null values in key columns."""

    name = "null_check"

    def run(self, df: pd.DataFrame) -> dict:
        nulls = {}
        for col in ["y_pred", "business_day", "hour_business", "model_name"]:
            if col in df.columns:
                n_null = int(df[col].isna().sum())
                if n_null > 0:
                    nulls[col] = n_null
        return {"nulls_found": nulls, "pass": len(nulls) == 0}


# ── CLI ────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P5M Plugin Interface — Dry-run smoke test",
    )
    parser.add_argument(
        "--prediction-pack",
        type=str,
        default=None,
        help="Path to a prediction CSV. At least one of --prediction-pack or "
        "--external-predictions is required.",
    )
    parser.add_argument(
        "--external-predictions",
        type=str,
        default=None,
        help="Additional prediction CSV(s) separated by comma.",
    )
    parser.add_argument(
        "--correction-modules",
        type=str,
        default="identity,clamp_low",
        help="Comma-separated names of correction modules to apply. "
        "Default: identity,clamp_low",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/plugin_smoke",
        help="Output directory for corrected CSV and monitor report.",
    )
    parser.add_argument(
        "--allow-long-format",
        action="store_true",
        help="Skip (timestamp, model_name) uniqueness check.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    if not args.prediction_pack and not args.external_predictions:
        logger.error("At least one of --prediction-pack or --external-predictions is required.")
        sys.exit(1)

    # ── Register example modules ────────────────────────────────────
    register_correction_module("identity", IdentityCorrection())
    register_correction_module("clamp_low", ClampCorrection(min_val=-500.0))
    register_monitor_module("row_count", RowCountMonitor())
    register_monitor_module("null_check", NullCheckMonitor())

    # ── Build correction list ───────────────────────────────────────
    correction_list = [
        name.strip()
        for name in args.correction_modules.split(",")
        if name.strip()
    ]

    # ── Run pipeline(s) ─────────────────────────────────────────────
    pipeline = DryRunPipeline(out_dir=args.out_dir)

    sources: list[str] = []
    if args.prediction_pack:
        sources.append(args.prediction_pack)
    if args.external_predictions:
        sources.extend(
            p.strip() for p in args.external_predictions.split(",") if p.strip()
        )

    for source_path in sources:
        logger.info("=" * 60)
        logger.info("Processing: %s", source_path)
        try:
            result = pipeline.run_from_path(
                prediction_path=source_path,
                correction_modules=correction_list,
                allow_long_format=args.allow_long_format,
            )
            logger.info(
                "Done — %d rows, corrections=%s, output=%s",
                result.row_count,
                result.correction_order,
                result.output_path,
            )
        except Exception:
            logger.exception("Pipeline failed for: %s", source_path)

    logger.info("Smoke test complete. Results in: %s", args.out_dir)


if __name__ == "__main__":
    main()
