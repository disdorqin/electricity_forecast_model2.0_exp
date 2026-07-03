"""Schema definitions for Intraday Tracker main pipeline integration.

Defines input pack fields (from deep branch Phase 10 handoff),
eval optional fields, and mainline standardized output fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Input pack fields from deep branch Phase 10
INTRADAY_PACK_INPUT_FIELDS: List[str] = [
    "business_day",
    "cutoff_hour",
    "target_hour",
    "ds",
    "mode",
    "base_model_name",
    "base_pred",
    "intraday_base_correction",
    "intraday_model_weight",
    "intraday_pre_guardrail_correction",
    "intraday_guardrail_weight",
    "intraday_final_correction",
    "intraday_corrected_pred",
    "intraday_confidence",
    "policy_decision",
    "fusion_weight",
    "shadow_only_flag",
    "guardrail_reason",
    "observed_hours",
    "n_observed",
    "residual_std_today",
    "bias_direction",
]

# Eval-only optional fields
INTRADAY_PACK_EVAL_FIELDS: List[str] = [
    "y_true",
    "baseline_error",
    "corrected_error",
]

# Mainline standardized output fields
MAINLINE_OUTPUT_FIELDS: List[str] = [
    "business_day",
    "hour_business",
    "target_hour",
    "cutoff_hour",
    "ds",
    "base_model_name",
    "base_pred",
    "intraday_corrected_pred",
    "intraday_final_correction",
    "intraday_confidence",
    "policy_decision",
    "fusion_weight",
    "shadow_only_flag",
    "guardrail_reason",
    "mode",
    "source_pack_path",
]


@dataclass
class ValidationResult:
    """Result of intraday pack validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    fallback_reason: Optional[str] = None

    @property
    def safe_fallback(self) -> bool:
        """Whether the pack should be disabled due to validation failure."""
        return not self.valid or self.fallback_reason is not None

    def add_error(self, msg: str):
        self.errors.append(msg)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)
