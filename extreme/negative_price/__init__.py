# -*- coding: utf-8 -*-
"""
negative_price — Negative price and low-valley residual correction module.

Provides leakage-safe labeling, feature engineering, risk estimation,
downward residual correction, and guardrails for negative/low price regimes.
Must not degrade high-price spike correction performance.

Sub-modules:
    schema      — Label definitions, constants, column names
    labels      — Negative price / low valley / overestimate label generation
    features     — Prediction-time safe feature engineering
    risk_model  — Negative/low price risk estimation
    residual_correction — Downward residual correction computation
    guardrail   — Safety guardrails (mutual exclusion with high_spike)
    apply_negative_correction — Main correction pipeline
"""

from __future__ import annotations
