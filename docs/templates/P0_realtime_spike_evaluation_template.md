# P0 Realtime High Spike Full Run — Evaluation Report

> **Status**: Placeholder (results to be filled after running Agents B, C, D, and this evaluation script)
> **Base branch**: `tune-timemixer`
> **Date**: 2026-06-29
> **Scope**: 2025-11, 2025-12 realtime predictions

---

## 1. Executive Summary

This report evaluates the P0 extreme high spike detection and correction pipeline on
realtime predictions for the two worst-performing months (2025-11, 2025-12).

The pipeline consists of three stages built on top of the base fused prediction:

1. **Spike label definition** (Agent B) — identifies extreme high price hours
2. **Risk model** (Agent C) — predicts spike probability from features
3. **Correction module** (Agent D) — applies targeted correction when risk is elevated

### Key Results

| Metric | Before (base_fused) | After (final) | Change |
|--------|--------------------|---------------|--------|
| Overall sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| Overall MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| High spike sMAPE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| Normal period sMAPE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| False lift rate | — | `[TO BE FILLED]` | — |

**Preliminary verdict**: `[TO BE FILLED — GO / NO-GO / CONDITIONAL]`

---

## 2. Methodology

### 2.1 Data

- **Period**: 2025-11-01 to 2025-12-31 (61 days, 1464 hourly realtime points)
- **Target**: realtime electricity price (yuan/MWh)
- **Base prediction**: fused prediction from the production pipeline
  (weights from BGEW + convex refit, 4 models: TimesFM, TimeMixer, RT916, SGDFNet)

### 2.2 Metric definitions

All metrics use the **sMAPE_floor50** definition from the project standard:

```python
def smape_floor50(y_true, y_pred, eps=1e-6):
    true_clip = np.where(y_true < 50.0, 50.0, y_true)
    pred_clip = np.where(y_pred < 50.0, 50.0, y_pred)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)
```

Additional metrics: MAE (mean absolute error), bias (mean signed error).

### 2.3 Pipeline stages

```
base_fused_prediction
    │
    ▼
Spike Detector ──► is_spike label (Agent B)
    │
    ▼
Risk Model ──► spike_probability (Agent C)
    │
    ▼
Correction Module ──► corrected prediction (Agent D)
    │
    ▼
final_prediction
```

### 2.4 Regime classification

Each hour is classified into one of four regimes based on the spike label:

| Regime | Criteria |
|--------|----------|
| `high_spike` | y_true is an extreme high price (above defined spike threshold) |
| `severe_underestimate` | base prediction severely underestimates y_true |
| `9_16` | Hours 9-16 (daytime solar period, known bottleneck) |
| `normal` | All other hours |

---

## 3. Overall Results

### 3.1 Before vs After — Full Dataset

| Metric | Before | After | Change | % Change |
|--------|--------|-------|--------|----------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| Bias | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| Count | 1464 | 1464 | — | — |

### 3.2 Model Risk Metrics

| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|
| AUC | `[TO BE FILLED]` | > 0.70 | `[TO BE FILLED]` |
| High spike recall | `[TO BE FILLED]` | > 60% | `[TO BE FILLED]` |
| High spike precision | `[TO BE FILLED]` | — | `[TO BE FILLED]` |
| 9_16 recall | `[TO BE FILLED]` | > 70% | `[TO BE FILLED]` |
| 9_16 precision | `[TO BE FILLED]` | — | `[TO BE FILLED]` |
| Overall recall | `[TO BE FILLED]` | — | `[TO BE FILLED]` |
| Overall precision | `[TO BE FILLED]` | — | `[TO BE FILLED]` |

### 3.3 Correction Statistics

| Metric | Value |
|--------|-------|
| Total corrections applied | `[TO BE FILLED]` |
| Mean lift | `[TO BE FILLED]` |
| Std lift | `[TO BE FILLED]` |
| % positive lift | `[TO BE FILLED]` |
| Guardrail triggers | `[TO BE FILLED]` |

---

## 4. Period Analysis

Results split by the three standard trading periods.

### 4.1 Period 1_8 (Valley, hours 1-8)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 4.2 Period 9_16 (Solar, hours 9-16)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 4.3 Period 17_24 (Peak, hours 17-24)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 4.4 Key observation

`[TO BE FILLED — e.g. "Correction primarily impacts 9_16 and 17_24; 1_8 is largely unchanged"]`

---

## 5. Regime Analysis

### 5.1 High Spike Regime

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| N hours | `[TO BE FILLED]` | — | — |
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 5.2 Severe Underestimate Regime

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| N hours | `[TO BE FILLED]` | — | — |
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 5.3 9_16 Regime

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| N hours | `[TO BE FILLED]` | — | — |
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 5.4 Normal Regime

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| N hours | `[TO BE FILLED]` | — | — |
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| MAE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 5.5 Key observation

`[TO BE FILLED — e.g. "High spike MAE improved by X%, confirming correction targets correctly. Normal regime sMAPE changed by < 1%, indicating no collateral damage."]`

---

## 6. Failure Case Analysis

### 6.1 Correction outcomes

| Category | Count | % of total | Notes |
|----------|-------|-----------|-------|
| Correction helped | `[TO BE FILLED]` | `[TO BE FILLED]` | Correction improved accuracy |
| Correction hurt | `[TO BE FILLED]` | `[TO BE FILLED]` | Correction worsened accuracy |
| False lift (non-spike) | `[TO BE FILLED]` | `[TO BE FILLED]` | Correction applied but was not a spike |
| Still missed (spike) | `[TO BE FILLED]` | `[TO BE FILLED]` | Spike detected but correction skipped |

### 6.2 Correction helped — Average effect

| Measure | Before | After |
|---------|--------|-------|
| Avg absolute error | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 6.3 Correction hurt — Average effect

| Measure | Before | After |
|---------|--------|-------|
| Avg absolute error | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 6.4 Root cause analysis

`[TO BE FILLED — e.g. "Most correction_hurt cases occur at regime transition hours (e.g. hour 8/9 boundary) where the risk model overestimates spike probability."]`

---

## 7. Risk Model Performance (Agent C)

### 7.1 Discrimination

| Metric | Value | Interpretation |
|--------|-------|---------------|
| AUC-ROC | `[TO BE FILLED]` | `[TO BE FILLED]` |
| AUC-PR | `[TO BE FILLED]` | More relevant when spikes are rare |

### 7.2 Calibration

| Metric | Value |
|--------|-------|
| Brier score | `[TO BE FILLED]` |
| Expected calibration error | `[TO BE FILLED]` |

### 7.3 Threshold analysis

| Threshold | Recall | Precision | F1 | False positive rate |
|-----------|--------|-----------|----|---------------------|
| 0.3 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| 0.5 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| 0.7 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 7.4 Key observation

`[TO BE FILLED]`

---

## 8. Monthly Comparison

### 8.1 November 2025

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| High spike sMAPE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 8.2 December 2025

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| sMAPE_floor50 | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |
| High spike sMAPE | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 8.3 Key observation

`[TO BE FILLED — e.g. "December shows larger improvement consistent with more spike events."]`

---

## 9. Day-level Analysis

`[TO BE FILLED — key days with largest improvement and largest regression]`

### 9.1 Top-5 days by sMAPE improvement

| Day | Base sMAPE | Final sMAPE | Improvement |
|-----|-----------|-------------|-------------|
| `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

### 9.2 Bottom-5 days by sMAPE regression

| Day | Base sMAPE | Final sMAPE | Regression |
|-----|-----------|-------------|------------|
| `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` | `[TO BE FILLED]` |

---

## 10. Key Findings

`[TO BE FILLED — summary of what the data shows]`

1. **Spike detection accuracy**: `[TO BE FILLED]`
   - How many spikes were correctly identified?
   - What was the false positive rate?

2. **Risk model effectiveness**: `[TO BE FILLED]`
   - Does the risk model add value beyond the detector alone?
   - Is there a correlation between risk probability and correction magnitude?

3. **Correction impact**: `[TO BE FILLED]`
   - Did corrections meaningfully improve spike-hour predictions?
   - Was there collateral damage to normal hours?

4. **Period dependence**: `[TO BE FILLED]`
   - Which periods benefited most? Benefited least?
   - Are there systematic biases?

5. **Model interaction**: `[TO BE FILLED]`
   - Do corrections interact well with the fusion weights?
   - Any unexpected behavior at period boundaries?

6. **Guardrail effectiveness**: `[TO BE FILLED]`
   - Did guardrails prevent negative price corruption?
   - How often did guardrails trigger?

---

## 11. Limitations

1. **Two-month scope**: Results are based on 2025-11/12 only. Generalisation to other
   months (e.g. summer with different price patterns) should be verified.
2. **Label definition dependency**: Spike labels are based on a specific definition
   (Agent B). A different definition may yield different conclusions.
3. **Single correction strategy**: Only one correction approach (Agent D) was tested.
4. **No cross-validation**: The evaluation is on the same period used for tuning,
   which may overstate in-period gains.
5. **Price regime shift**: If market conditions change, the risk model and correction
   thresholds may need recalibration.

---

## 12. Conclusion

**Verdict**: `[TO BE FILLED — GO / NO-GO / CONDITIONAL]`

### GO criteria (all must pass)

- [ ] Risk model AUC > 0.70
- [ ] High spike recall > 60%
- [ ] 9_16 recall > 70%
- [ ] Overall sMAPE_floor50 improves
- [ ] Normal regime sMAPE worsens by < 1%
- [ ] False lift rate < 20%
- [ ] Guardrail prevents negative price corruption

### NO-GO criteria (any single trigger)

- [ ] Overall sMAPE_floor50 regresses
- [ ] Normal regime sMAPE worsens by > 2%
- [ ] False lift rate > 30%
- [ ] Guardrail failures on negative prices

### Conditional acceptance

- Some criteria pass but not all: accept with conditions
- Required fixes before full production:
  1. `[TO BE FILLED]`
  2. `[TO BE FILLED]`
  3. `[TO BE FILLED]`

---

## 13. Next Steps

1. **Fill results**: Run the evaluation script after Agents B, C, D complete
2. **Review thresholds**: Adjust risk model threshold and correction magnitude
   based on actual results
3. **Cross-month validation**: Test on additional months (e.g. 2026-01, 2026-02)
4. **Production integration**: If verdict is GO, add toggle flag to production config
5. **Monitoring**: Add spike correction rate to production dashboard
6. **Iterate**: If verdict is CONDITIONAL, address specific failures and re-run

---

*Report generated by Agent E (evaluation framework)*
