# P5M Negative Residual Module Report

## Overview

The **negative price / low-valley residual correction module** applies a downward
correction to predictions when negative-price or low-valley risk is detected.
It is designed to be **leakage-safe** (no future/actual values at prediction time)
and **mutually exclusive** with the high-spike correction (never both active).

## 1. Labels

| Label | Rule | Threshold |
|-------|------|-----------|
| `negative_price` | `y_true < 0` | `NEGATIVE_PRICE_THRESHOLD = 0.0` |
| `low_valley` | `y_true <= max(50, rolling_p10)` | `LOW_VALLEY_ABSOLUTE = 50`, `LOW_VALLEY_PERCENTILE = 0.10` |
| `overestimate_low` | `y_pred - y_true >= threshold` (diagnostic only) | `OVERESTIMATE_LOW_THRESHOLD = 30` |

The low-valley threshold uses `max(50, p10)` (the **less conservative** of the two)
to capture all events below either threshold. This matches the spec requirement
`y_true <= max(50, rolling_p10)`.

## 2. Features

All features are **prediction-time safe** (no y_true, actual values, or residuals):

### Time features
`hour_business`, `hour`, `period`, `weekday`, `month`, `day_of_month`,
`is_weekend`, `season_bucket`

### Prediction signals
`base_fused_pred`, `final_pred`, `final_pred_before_negative`,
`sgdfnet_pred`, `timemixer_pred`, `rt916_pred`, `timesfm_pred`,
`dayahead_proxy`, `prediction_spread`, `model_disagreement`,
`pred_std`, `pred_range`

### Forecast exogenous
`地方电厂总加预测值`, `直调负荷预测值`, `风电总加预测值`, `光伏总加预测值`,
`核电总加预测值`, `竞价空间预测值`, `新能源总加预测值`, etc.

### Negative risk signals
`recent_negative_rate_by_hour`, `recent_negative_rate_by_period`,
`recent_low_price_rate_by_hour`, `recent_low_price_rate_by_period`,
`min_pred_last_24h`, `renewable_ratio`

Rate features use historical data when available (`history_df` parameter);
fall back to within-batch estimation otherwise.

## 3. Correction Logic

The module applies **downward-only** correction:

```
final_after_negative <= final_before_negative
```

Pipeline order:
```
base prediction
-> high_spike correction (if applicable)
-> negative/low-valley risk estimation
-> downward residual correction
-> guardrail
-> final_pred
```

### Risk estimation
- Option A: `NegativeRiskModel` (sklearn RF/LR classifier, trained on labels)
- Option B: Heuristic (prediction percentile as risk proxy)

### Correction amount
- Fitted from historical residuals (`y_true - y_pred`) at low quantile (default p10)
- Per-period quantiles (1_8, 9_16, 17_24)
- Capped by ratio (`max_downward_ratio`) and absolute limit (`max_absolute_downward`)
- Floor at `min_pred_floor`

### Correction profiles

| Profile | Risk Threshold | Max Ratio | Max Absolute | Min Floor | 9_16 Protection |
|---------|---------------|-----------|-------------|-----------|-----------------|
| conservative | 0.50 | 0.10 | 15 | -50 | Yes |
| moderate | 0.35 | 0.20 | 30 | -100 | Yes |
| aggressive | 0.25 | 0.30 | 50 | -200 | No |

## 4. Guardrails

1. **High-spike mutual exclusion**: If `high_spike_prob > 0.5`, downward correction
   is skipped and the base prediction is preserved.
2. **Max downward per period**: Per-period ratio and absolute caps prevent
   excessive correction.
3. **Absolute price floor**: Predictions cannot be pushed below `min_allowed_price`.
4. **9_16 protection**: Reduced correction during spike-prone hours when
   `low_valley_risk` is not high.
5. **Normal-hour protection**: Lower risk threshold for non-valley periods.
6. **False negative correction rate**: Statistic tracked via `negative_reason_code`.

## 5. Metrics

| Metric | Description |
|--------|-------------|
| `negative_count` | Number of negative price events |
| `low_valley_count` | Number of low valley events |
| `negative_MAE_before/after` | MAE on negative price hours |
| `low_valley_MAE_before/after` | MAE on low valley hours |
| `negative_miss_before/after` | Count of y_true<0 but y_pred>=0 |
| `low_valley_overestimate_before/after` | Count of y_pred - y_true >= 30 on low valley |
| `overall_sMAPE_before/after/delta` | sMAPE with floor=50 |
| `high_spike_MAE_before/after/delta` | MAE on high spike hours (y_true > 150) |
| `normal_degradation` | sMAPE delta for non-9_16 hours |

## 6. GO / NO-GO / DATA-LIMITED

### GO conditions
- `negative_MAE` improves **or** `low_valley_MAE` improves
- `overall_sMAPE` does not worsen > 0.3
- `high_spike_MAE` does not worsen > 3%
- `normal_degradation <= 0.5`

### DATA-LIMITED
If `negative_count == 0`, module emits `NEGATIVE MODULE DATA-LIMITED`.
Low-valley evaluation still applies.

### Current status
Requires running evaluation on the canonical pack to determine.

## 7. Interaction with High-Spike Correction

The negative correction module is **fully independent** of the high-spike module:

- **No shared parameters**: Separate configs, profiles, guardrails.
- **Mutual exclusion at inference**: When `high_spike_prob > threshold`,
  downward correction is not applied.
- **Independent metrics**: Both modules track their own MAE on their target hours.
- **Pipeline ordering**: high_spike correction runs first, then negative correction.

## 8. File Structure

```
extreme/negative_price/
  __init__.py              — Module docstring
  schema.py                — Constants, label names, feature families, leakage columns
  labels.py                — Label generation (negative, low_valley, overestimate_low)
  features.py              — Leakage-safe feature engineering
  risk_model.py            — Optional sklearn risk classifier
  residual_correction.py   — Downward correction quantile computation
  guardrail.py             — Safety guardrails and mutual exclusion
  apply_negative_correction.py — Main pipeline + metric computation

scripts/
  evaluate_p5m_negative_residual_module.py  — CLI evaluation entry point

tests/
  test_p5m_negative_residual_module.py      — Unit + integration tests

docs/reports/
  P5M_negative_residual_module_report.md    — This report
```
