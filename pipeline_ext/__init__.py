# -*- coding: utf-8 -*-
"""pipeline_ext — Plugin / Interface module for external model integration.

This package provides a lightweight, stable, testable interface layer
that allows external prediction pipelines to plug into the fusion,
correction, and monitoring chain without touching production_pipeline.py.

Sub-modules:
    schema      — Unified prediction schema definition and validation.
    registry    — Module registry for providers, corrections, monitors.
    io          — CSV load / schema enforcement / uniqueness checks.
    modules     — Base classes: PredictionProvider, CorrectionModule, MonitorModule.
    pipeline    — Dry-run pipeline orchestration (not production_pipeline).
"""

from pipeline_ext.schema import (
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    ALL_FIELDS,
    validate_schema,
    check_uniqueness,
    check_leakage_safe,
)
from pipeline_ext.registry import (
    register_prediction_provider,
    register_correction_module,
    register_monitor_module,
    get_module,
    list_modules,
    list_providers,
    list_corrections,
    list_monitors,
)
from pipeline_ext.modules import (
    PredictionProvider,
    CorrectionModule,
    MonitorModule,
)
from pipeline_ext.io import (
    load_prediction_csv,
    validate_prediction_dataframe,
    load_and_validate,
)
from pipeline_ext.pipeline import (
    DryRunPipeline,
    run_dry_run,
)

__all__ = [
    # schema
    "REQUIRED_FIELDS",
    "OPTIONAL_FIELDS",
    "ALL_FIELDS",
    "validate_schema",
    "check_uniqueness",
    "check_leakage_safe",
    # registry
    "register_prediction_provider",
    "register_correction_module",
    "register_monitor_module",
    "get_module",
    "list_modules",
    "list_providers",
    "list_corrections",
    "list_monitors",
    # modules
    "PredictionProvider",
    "CorrectionModule",
    "MonitorModule",
    # io
    "load_prediction_csv",
    "validate_prediction_dataframe",
    "load_and_validate",
    # pipeline
    "DryRunPipeline",
    "run_dry_run",
]
