# -*- coding: utf-8 -*-
"""Residual Stack — Unified high-spike + negative/low-valley correction orchestration.

Architecture::

    base_pred
        │
        ▼
    ┌──────────────────────────────┐
    │  High-spike correction       │  ← extreme.realtime_high_spike
    │  (upward lift only,          │
    │   period-aware quantiles)     │
    └──────────┬───────────────────┘
               │ after_high_spike_pred
               ▼
    ┌──────────────────────────────┐
    │  Negative/low-valley         │  ← extreme.negative_price
    │  correction (downward only,  │
    │   mutually exclusive w/      │
    │   high_spike via priority)   │
    └──────────┬───────────────────┘
               │ after_negative_pred
               ▼
    ┌──────────────────────────────┐
    │  Final guardrail             │
    │  (normal-hour protection,    │
    │   max lift/down cap)         │
    └──────────┬───────────────────┘
               │ final_pred
               ▼
          correction_reason
          module_sequence

Key rule — High-spike takes priority:
    If high_spike correction was applied (lift > 0):
        negative module MUST NOT apply downward correction.
"""

from residual_stack.schema import (
    STACK_OUTPUT_COLUMNS,
    REASON_CODES,
)
from residual_stack.priority import (
    check_high_spike_priority,
    should_apply_negative,
    format_module_sequence,
)
from residual_stack.orchestrator import (
    ResidualStackOrchestrator,
    StackProfile,
    StackResult,
    run_residual_stack,
)
from residual_stack.metrics import (
    compute_stack_metrics,
    compare_configs,
    format_metrics_table,
)
from residual_stack.risk_source import (
    RiskSource,
    detect_risk_source,
    is_official_source,
    resolve_risk_policy,
    format_risk_verdict,
)

__all__ = [
    # schema
    "STACK_OUTPUT_COLUMNS",
    "REASON_CODES",
    # priority
    "check_high_spike_priority",
    "should_apply_negative",
    "format_module_sequence",
    # orchestrator
    "ResidualStackOrchestrator",
    "StackProfile",
    "StackResult",
    "run_residual_stack",
    # metrics
    "compute_stack_metrics",
    "compare_configs",
    "format_metrics_table",
    # risk source
    "RiskSource",
    "detect_risk_source",
    "is_official_source",
    "resolve_risk_policy",
    "format_risk_verdict",
]
