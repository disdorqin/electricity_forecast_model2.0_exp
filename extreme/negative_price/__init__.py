# -*- coding: utf-8 -*-
"""
negative_price — Negative price and low-valley residual correction module.

Provides leakage-safe labeling, feature engineering, risk estimation,
downward residual correction, and guardrails for negative/low price regimes.
Must not degrade high-price spike correction performance.

Quick-start:
    >>> from extreme.negative_price import (
    ...     NegativeCorrectionProfile,
    ...     NegativeResidualCorrector,
    ...     NegativeGuardrail,
    ...     apply_negative_correction,
    ...     compute_metrics,
    ... )
    >>> profile = NegativeCorrectionProfile(name="conservative")
    >>> corrector = NegativeResidualCorrector()
    >>> corrector.fit_from_history(history_df)
    >>> guardrail = NegativeGuardrail()

Sub-modules:
    schema      — Label definitions, constants, column names
    labels      — Negative price / low valley / overestimate label generation
    features    — Prediction-time safe feature engineering
    risk_model  — Negative/low price risk estimation
    residual_correction — Downward residual correction computation
    guardrail   — Safety guardrails (mutual exclusion with high_spike)
    apply_negative_correction — Main correction pipeline
"""

from __future__ import annotations

from extreme.negative_price.schema import (
    NEGATIVE_PRICE_COL,
    LOW_VALLEY_COL,
    OVERESTIMATE_LOW_COL,
    NEGATIVE_PRICE_THRESHOLD,
    LOW_VALLEY_ABSOLUTE_THRESHOLD,
    LOW_VALLEY_PERCENTILE,
    OVERESTIMATE_LOW_THRESHOLD,
    TARGET_LEAKAGE_COLS,
    SAFE_FEATURE_FAMILIES,
)

from extreme.negative_price.labels import (
    generate_negative_price_labels,
    generate_low_valley_labels,
    generate_overestimate_low_labels,
    add_all_labels,
    compute_low_valley_percentile,
)

from extreme.negative_price.features import (
    engineer_negative_price_features,
    select_feature_columns,
    get_feature_columns,
)

from extreme.negative_price.risk_model import (
    NegativeRiskModel,
    NegativeRiskConfig,
    RiskTarget,
    fit_risk_model,
)

from extreme.negative_price.residual_correction import (
    NegativeResidualCorrector,
    NegativeResidualConfig,
    DownwardCorrectionResult,
    get_period,
    PERIOD_DEFS,
)

from extreme.negative_price.guardrail import (
    NegativeGuardrail,
    NegativeGuardrailConfig,
)

from extreme.negative_price.apply_negative_correction import (
    NegativeCorrectionProfile,
    apply_negative_correction,
    run_correction_with_profile,
    compute_metrics,
    get_profile,
    PROFILES,
)

__all__ = [
    # Schema constants
    "NEGATIVE_PRICE_COL",
    "LOW_VALLEY_COL",
    "OVERESTIMATE_LOW_COL",
    "NEGATIVE_PRICE_THRESHOLD",
    "LOW_VALLEY_ABSOLUTE_THRESHOLD",
    "LOW_VALLEY_PERCENTILE",
    "OVERESTIMATE_LOW_THRESHOLD",
    "TARGET_LEAKAGE_COLS",
    "SAFE_FEATURE_FAMILIES",
    # Label functions
    "generate_negative_price_labels",
    "generate_low_valley_labels",
    "generate_overestimate_low_labels",
    "add_all_labels",
    "compute_low_valley_percentile",
    # Feature functions
    "engineer_negative_price_features",
    "select_feature_columns",
    "get_feature_columns",
    # Risk model
    "NegativeRiskModel",
    "NegativeRiskConfig",
    "RiskTarget",
    "fit_risk_model",
    # Residual correction
    "NegativeResidualCorrector",
    "NegativeResidualConfig",
    "DownwardCorrectionResult",
    "get_period",
    "PERIOD_DEFS",
    # Guardrail
    "NegativeGuardrail",
    "NegativeGuardrailConfig",
    # Pipeline
    "NegativeCorrectionProfile",
    "apply_negative_correction",
    "run_correction_with_profile",
    "compute_metrics",
    "get_profile",
    "PROFILES",
]
