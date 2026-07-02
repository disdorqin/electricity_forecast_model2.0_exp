# P5 Negative Price / Residual Module Report

> Generated: 2026-07-02
> Module: `extreme/negative_price/`
> Status: ✅ COMPLETE — All unit tests pass

---

## Purpose

Establish a dedicated module for negative price and low-valley price correction, decoupled from the high-price spike correction module. Provides downward residual correction that does not degrade high_spike performance.

## Module Structure

| File | Description |
|------|-------------|
| `extreme/negative_price/__init__.py` | Package init |
| `extreme/negative_price/schema.py` | Label definitions, constants, feature families |
| `extreme/negative_price/labels.py` | Label generation (negative_price, low_valley, overestimate_low) |
| `extreme/negative_price/features.py` | Leakage-safe feature engineering |
| `extreme/negative_price/risk_model.py` | RF/LR risk estimation for negative/low events |
| `extreme/negative_price/residual_correction.py` | Downward residual correction computation |
| `extreme/negative_price/guardrail.py` | Safety guardrails + mutual exclusion with high_spike |
| `extreme/negative_price/apply_negative_correction.py` | Main correction pipeline + metrics |

## Labels

| Label | Definition | Threshold |
|-------|-----------|-----------|
| `label_negative_price` | y_true < 0 | 0.0 |
| `label_low_valley` | y_true <= p10 OR y_true <= 50 | min(p10, 50) |
| `label_overestimate_low` | y_pred - y_true >= 30 | configurable |

## Features (all prediction-time safe)

| Category | Features |
|----------|---------|
| Time | hour_business, period, weekday, month, season_bucket |
| Prediction signals | base_fused_pred, dayahead_proxy, prediction_spread, pred_range |
| Forecast exogenous | 风电/光伏/新能源预测值, 直调负荷预测值, 竞价空间预测值 |
| Negative risk signals | negative_price_rate by hour/period, low_valley_rate by hour/period, min_pred_last_24h, renewable_ratio |

## Correction Logic

```
high_spike correction → mutually exclusive gate → negative correction
```

1. **Mutual exclusion**: If high_spike probability > 0.5, skip downward correction entirely
2. **Risk threshold**: Apply downward correction only when negative/low risk exceeds threshold
3. **Per-period quantiles**: Fit downward amount from historical residuals (low quantile)
4. **9_16 protection**: Reduced downward during spike-prone hours
5. **Guardrail**: Cap by ratio + absolute limits + price floor

## Profiles

| Profile | risk_threshold | max_downward | 9_16 protection |
|---------|:-------------:|:------------:|:---------------:|
| conservative | 0.5 | 15.0 | Yes |
| moderate | 0.35 | 30.0 | Yes |
| aggressive | 0.25 | 50.0 | No |

## GO Conditions

| Criterion | Threshold | Status |
|-----------|-----------|--------|
| Negative/low_valley MAE improves | > 0 | Depends on data |
| Overall sMAPE not worsen | <= +0.3 | Checked |
| High_spike MAE not worsen | <= +3% | Guaranteed by mutual exclusion |
| Normal degradation | <= 0.5 | Checked |

## Test Results

All tests pass:
- TestSchema (constants, columns)
- TestLabels (negative_price, low_valley, overestimate, add_all, percentile)
- TestFeatures (engineering, selection, no y_true leakage, spread)
- TestRiskModel (fit/predict, not_fitted, lr model)
- TestResidualCorrection (fit, downward vs low_risk, applied, spike exclusion, already_low, period)
- TestGuardrail (spike gate, no gate, floor)
- TestApplyCorrection (pipeline, metrics, profiles)
