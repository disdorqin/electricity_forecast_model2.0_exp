#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ledger Quality Check CLI."""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check prediction ledger quality")
    parser.add_argument("ledger_path", help="Ledger CSV path")
    parser.add_argument("--strict", action="store_true", help="Strict mode: fail on missing final_pred")
    parser.add_argument("--html", default=None, help="Output HTML report path")
    parser.add_argument("--json", default=None, help="Output JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    from ledger.quality import run_ledger_quality_check
    report = run_ledger_quality_check(args.ledger_path, strict=args.strict)
    print()
    print(report.summary())
    print()
    if args.json:
        import dataclasses
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(report), f, indent=2, ensure_ascii=False, default=str)
        print(f"JSON report saved to {args.json}")
    if args.html:
        _write_html_report(report, args.html)
    return 0 if report.passed else 1


def _write_html_report(report, path: str):
    status = "PASSED" if report.passed else "FAILED"
    passed_class = "pass" if report.passed else "fail"
    rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(report.n_nulls_per_column.items()))
    err_html = "".join(f"<li>{e}</li>" for e in report.all_errors[:20]) if report.all_errors else "<li>None</li>"
    warn_html = "".join(f"<li>{w}</li>" for w in report.all_warnings[:10]) if report.all_warnings else "<li>None</li>"
    with open(path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html>" + chr(10))
        f.write("<html><head><meta charset=utf-8><title>Ledger Quality Report</title>" + chr(10))
        f.write("<style>body{font-family:sans-serif;margin:20px}</style>" + chr(10))
        f.write("</head><body>" + chr(10))
        f.write("<h1>Ledger Quality Report</h1>" + chr(10))
        f.write(f"<p>Path: {report.ledger_path}</p>" + chr(10))
        f.write(f"<p>Checked at: {report.checked_at}</p>" + chr(10))
        f.write(f"<h2 class={passed_class}>{status}</h2>" + chr(10))
        f.write("<h3>Overview</h3>" + chr(10))
        f.write("<table border=1>" + chr(10))
        f.write(f"<tr><td>Total rows</td><td>{report.total_rows}</td></tr>" + chr(10))
        f.write(f"<tr><td>Targets</td><td>{report.n_targets}</td></tr>" + chr(10))
        f.write(f"<tr><td>Forecast dates</td><td>{report.n_forecast_dates}</td></tr>" + chr(10))
        f.write(f"<tr><td>Models</td><td>{report.n_models}</td></tr>" + chr(10))
        f.write(f"<tr><td>Missing final_pred</td><td>{report.missing_final_pred}</td></tr>" + chr(10))
        f.write(f"<tr><td>Duplicate rows</td><td>{len(report.duplicate_rows)}</td></tr>" + chr(10))
        f.write(f"<tr><td>Hour business errors</td><td>{len(report.hour_business_out_of_range)}</td></tr>" + chr(10))
        f.write(f"<tr><td>Timestamp errors</td><td>{len(report.timestamp_mapping_errors)}</td></tr>" + chr(10))
        f.write(f"<tr><td>Missing models</td><td>{len(report.missing_models)}</td></tr>" + chr(10))
        f.write("</table>" + chr(10))
        f.write("<h3>Null counts per column</h3>" + chr(10))
        f.write(f"<table border=1><tr><th>Column</th><th>Nulls</th></tr>{rows_html}</table>" + chr(10))
        f.write(f"<h3>Errors ({len(report.all_errors)})</h3><ul>{err_html}</ul>" + chr(10))
        f.write(f"<h3>Warnings ({len(report.all_warnings)})</h3><ul>{warn_html}</ul>" + chr(10))
        f.write("</body></html>" + chr(10))
    print(f"HTML report saved to {path}")


if __name__ == "__main__":
    raise SystemExit(main())