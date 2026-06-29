#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ledger Append CLI — 将 pipeline 产出追加到每日预测账本。

用法：
  python scripts/ledger_append.py 2026-06-29
  python scripts/ledger_append.py 2026-06-29 --output-root outputs --ledger-dir data/local_ledger
  python scripts/ledger_append.py 2026-06-29 --dry-run
  python scripts/ledger_append.py 2026-06-29 --force
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 pipeline 产出物追加到每日预测账本")
    parser.add_argument("run_date", help="运行日期 YYYY-MM-DD")
    parser.add_argument("--output-root", default="outputs", help="pipeline 输出根目录 (default: outputs)")
    parser.add_argument("--ledger-dir", default=None, help="账本目录 (default: data/local_ledger)")
    parser.add_argument("--dry-run", action="store_true", help="仅扫描不写入")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的相同键的行")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from ledger.append import ledger_append_from_pipeline_run
    result = ledger_append_from_pipeline_run(
        run_date=args.run_date,
        output_root=args.output_root,
        ledger_dir=args.ledger_dir,
        force=args.force,
        dry_run=args.dry_run,
    )
    status = result.get("status", "unknown")
    n_rows = result.get("appended_rows", 0)
    if status == "error":
        logger.error("FAILED: %s", result.get("message", ""))
        print(f"ERROR: {result.get('message', '')}")
        return 1
    if args.dry_run:
        print(f"DRY RUN: {n_rows} rows would be appended")
        print(f"  Models: {result.get('n_models', '?')}")
        print(f"  Dates: {result.get('forecast_dates', [])}")
    else:
        files = result.get("files", {})
        print(f"OK: {n_rows} rows appended")
        for fpath, n in files.items():
            print(f"  {fpath}: {n} rows")
    errors = result.get("errors", [])
    warnings = result.get("warnings", [])
    if errors:
        for e in errors:
            logger.warning("Error: %s", e)
    if warnings:
        for w in warnings:
            logger.warning("Warning: %s", w)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
