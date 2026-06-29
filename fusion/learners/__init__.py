"""OOF learner package -- ROEL-BGEW fusion system."""
from .oof_contracts import load_and_normalize_oof_table, load_forecast_long
from .data_checks import compute_coverage_report
from .metrics import compute_all_metrics
from .static_convex import fit_static_convex
from .bgew import fit_bgew
from .candidate_selector import select_best_candidate, fit_all_candidates
from .roel import run_roel_bgew_fallback
from .apply_learner import apply_learner_to_forecast

__all__ = [
    "load_and_normalize_oof_table",
    "load_forecast_long",
    "compute_coverage_report",
    "compute_all_metrics",
    "fit_static_convex",
    "fit_bgew",
    "select_best_candidate",
    "fit_all_candidates",
    "run_roel_bgew_fallback",
    "apply_learner_to_forecast",
]
