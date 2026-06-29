from __future__ import annotations

import logging
import os

# Must be set before any CUDA/torch initialization to allow deterministic algorithms with CuBLAS
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from cli.parser import build_parser


def _normalize_date(raw: str) -> str:
    """Normalize date string like '2026.2.1' or '2026-02-01' to 'YYYY-MM-DD'."""
    import pandas as pd
    raw = raw.strip().replace(".", "-")
    return pd.Timestamp(raw).strftime("%Y-%m-%d")


def _try_parse_date_range(raw: str):
    """Try to parse 'YYYY.M.D-YYYY.M.D' or 'YYYY-MM-DD-YYYY-MM-DD' as (start, end).

    Returns (start_str, end_str) in 'YYYY-MM-DD' format, or None if not a valid range.
    """
    import re
    import pandas as pd
    # Pattern: date-date where date is YYYY.M.D or YYYY-MM-DD
    # We try splitting by a separator that looks like a date boundary
    # Strategy: find all date-like tokens
    tokens = re.split(r'(?<=\d)[-.](?=\d{4})', raw)
    if len(tokens) == 2:
        try:
            start = pd.Timestamp(tokens[0].strip().replace(".", "-")).strftime("%Y-%m-%d")
            end = pd.Timestamp(tokens[1].strip().replace(".", "-")).strftime("%Y-%m-%d")
            return start, end
        except Exception:
            return None
    # Also try simple split on the last "-" that separates two YYYY-MM-DD dates
    # e.g. "2026-02-01-2026-02-28" → split at position where second date starts
    m = re.match(r'(\d{4}[-.]\d{1,2}[-.]\d{1,2})\s*-\s*(\d{4}[-.]\d{1,2}[-.]\d{1,2})$', raw)
    if m:
        try:
            start = pd.Timestamp(m.group(1).replace(".", "-")).strftime("%Y-%m-%d")
            end = pd.Timestamp(m.group(2).replace(".", "-")).strftime("%Y-%m-%d")
            return start, end
        except Exception:
            return None
    return None


def main() -> int:
    args = build_parser().parse_args()
    # Positional date shortcut: `python main.py 2026-02-01` or `python main.py 2026.2.1-2026.2.28`
    if args.pos_date is not None and args.date is None and args.start is None:
        raw = args.pos_date.strip()
        # Check for range format: "YYYY.M.D-YYYY.M.D" or "YYYY-MM-DD-YYYY-MM-DD"
        if "-" in raw:
            # Try to split on "-" as date range separator
            # Handle both "2026.2.1-2026.2.28" and "2026-02-01-2026-02-28"
            parts = _try_parse_date_range(raw)
            if parts:
                args.start, args.end = parts
                args.date = None
            else:
                args.date = _normalize_date(raw)
        else:
            args.date = _normalize_date(raw)
    if args.pipeline == "predict":
        from pipelines.predict_pipeline import run_predict_pipeline
        results = run_predict_pipeline(args)
        for result in results:
            if result is None:
                continue
            print(f"{result.model_name}:{result.target} -> {result.output_path}")
        return 0
    if args.pipeline == "train":
        from pipelines.train_pipeline import run_train_pipeline
        results = run_train_pipeline(args)
        for result in results:
            print(f"{result.model_name}:{result.target} -> train done")
        return 0
    if args.pipeline == "evaluate":
        from pipelines.evaluate_pipeline import run_evaluate_pipeline
        output_path = run_evaluate_pipeline(args)
        print(output_path)
        return 0
    if args.pipeline == "fusion":
        from pipelines.fusion_pipeline import run_fusion_pipeline
        output_path = run_fusion_pipeline(args)
        print(output_path)
        return 0
    if args.pipeline == "sync_dataset":
        from pipelines.sync_dataset_pipeline import run_sync_dataset_pipeline
        output_path = run_sync_dataset_pipeline(args)
        print(output_path)
        return 0
    if args.pipeline == "model_stage":
        from pipelines.staged_pipeline import run_model_stage
        outputs = run_model_stage(args)
        for output in outputs:
            print(output)
        return 0
    if args.pipeline == "learner_stage":
        from pipelines.staged_pipeline import run_learner_stage
        outputs = run_learner_stage(args)
        for output in outputs:
            print(output)
        return 0
    if args.pipeline == "fuse_stage":
        from pipelines.staged_pipeline import run_fuse_stage
        outputs = run_fuse_stage(args)
        for output in outputs:
            print(output)
        return 0
    if args.pipeline == "classifier_stage":
        from pipelines.staged_pipeline import run_classifier_stage
        result = run_classifier_stage(args)
        print(result)
        return 0
    if args.pipeline == "full":
        from pipelines.production_pipeline import run_production_pipeline
        result = run_production_pipeline(args)
        if isinstance(result, list):
            for r in result:
                status = r.get("status", "?")
                dt = r.get("date", "?")
                warnings = r.get("warnings", [])
                line = f"  {dt}: {status}"
                if warnings:
                    line += f" ({len(warnings)} warnings)"
                print(line)
        elif isinstance(result, dict):
            status = result.get("status", "done")
            dt = result.get("date", "?")
            warnings = result.get("warnings", [])
            final = result.get("final_outputs", {})
            print(f"R3D-Tap-GEF {dt}: {status}")
            if warnings:
                for w in warnings[:5]:
                    print(f"  [WARN] {w}")
            if final:
                for key, path in final.items():
                    print(f"  {key}: {path}")
        return 0
    if args.pipeline == "rolling_oof":
        from rolling_oof.cli import run_rolling_oof
        result = run_rolling_oof(args)
        return 0
    if args.pipeline == "oof_learner":
        from fusion.experiments.run_oof_learner import run_oof_learner
        result = run_oof_learner(args)
        print(f"OOF learner complete. Outputs in: {result.get('output_root', 'learner_runs')}")
        return 0
    if args.pipeline == "apply_oof_learner":
        from fusion.experiments.run_oof_learner import run_apply_oof_learner
        result = run_apply_oof_learner(args)
        print(f"Apply complete. Output: {result['output_path']}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
