from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified electricity forecast entrypoint")
    parser.add_argument(
        "pos_date", nargs="?", default=None,
        help="Target date (YYYY-MM-DD). Shortcut for --pipeline full --date <DATE>",
    )
    parser.add_argument(
        "--pipeline",
        default="full",
        choices=[
            "predict",
            "train",
            "evaluate",
            "fusion",
            "sync_dataset",
            "model_stage",
            "learner_stage",
            "fuse_stage",
            "classifier_stage",
            "full",
            "rolling_oof",
            "oof_learner",
            "apply_oof_learner",
        ],
    )
    parser.add_argument("--target", default="both", choices=["dayahead", "realtime", "both"])
    parser.add_argument("--models", default="all", help="Comma-separated model names or all")
    parser.add_argument("--stage-models", default="formal", help="Staged execution model set: formal, all, or comma-separated names")
    parser.add_argument("--date", default=None, help="Single target day, YYYY-MM-DD")
    parser.add_argument("--start", default=None, help="Range start, YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Range end, YYYY-MM-DD")
    parser.add_argument("--data-path", default="data/shandong_pmos_hourly.csv")
    parser.add_argument("--output-root", default="outputs/unified_runs")
    parser.add_argument("--max-cpu-workers", type=int, default=2)
    parser.add_argument("--max-gpu-workers", type=int, default=1)
    parser.add_argument("--training-months", type=int, default=12)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--use-predicted-temp", action="store_true")
    parser.add_argument("--segment-count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--pred-path", default=None)
    parser.add_argument("--actual-path", default=None)
    parser.add_argument("--fusion-work-dir", default="fusion_runs/unified_entry")
    parser.add_argument("--train-length-decision", default="fusion_runs/repro_training_length_probe/repro_training_length_decision.json")
    parser.add_argument("--weight-lower-bound", type=float, default=-0.5)
    parser.add_argument("--weight-upper-bound", type=float, default=1.2)
    parser.add_argument("--conda-env", default="")
    parser.add_argument("--use-classifier", action="store_true", default=False)
    parser.add_argument("--clf-data", default=None)
    parser.add_argument("--daily-run-root", default="daily_runs")
    parser.add_argument("--validation-days", type=int, default=30, help="Number of days for the validation window (default: 30). Used by model_stage for weight fitting.")

    # --- rolling-origin OOF 池参数 ---
    rolling_group = parser.add_argument_group("Rolling OOF Pool Options")
    rolling_group.add_argument("--oof-output-root", default="oof_runs", help="OOF output root directory")
    rolling_group.add_argument("--oof-start-month", default=None, help="First target month, YYYY-MM")
    rolling_group.add_argument("--oof-end-month", default=None, help="Last target month, YYYY-MM")
    rolling_group.add_argument("--oof-expanding", action="store_true", default=True, help="Use expanding window (default)")
    rolling_group.add_argument("--oof-train-min-months", type=int, default=6, help="Min training months for sliding window")
    rolling_group.add_argument("--timemixer-rolling-mode", default="daily", choices=["window_once", "block", "daily"], help="TimeMixer rolling mode")
    rolling_group.add_argument("--timemixer-block-days", type=int, default=7, help="TimeMixer block mode: days per block")
    rolling_group.add_argument("--escort-date", default=None, help="Phase C escort prediction target date")
    rolling_group.add_argument("--skip-oof-audit", action="store_true", help="Skip fold-level audit")

    # --- OOF learner arguments ---
    learner_group = parser.add_argument_group("OOF Learner Options")
    learner_group.add_argument("--oof-path", default=None, help="Path to OOF long-table CSV")
    learner_group.add_argument("--learner-mode", default="roel_bgew_fallback", help="Learner mode (default: roel_bgew_fallback)")
    learner_group.add_argument("--metric", default="sMAPE_floor50", choices=["sMAPE_floor50", "MAE"], help="Optimization metric")
    learner_group.add_argument("--tau", type=float, default=30.0, help="BGEW time constant (days)")
    learner_group.add_argument("--eta", type=float, default=0.5, help="BGEW learning rate")
    learner_group.add_argument("--coverage-threshold", type=float, default=0.95, help="Minimum coverage for model eligibility")
    learner_group.add_argument("--forecast-path", default=None, help="Path to forecast long-table for apply_oof_learner")
    learner_group.add_argument("--learner-artifact", default=None, help="Path to learner artifact directory")

    return parser
