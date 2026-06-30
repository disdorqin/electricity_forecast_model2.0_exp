# P3b Rolling 30D Fusion Report

**Generated**: 2026-06-30
**Branch**: `agent/p3-rolling-fusion-sota`
**Prediction pack**: `lightgbm_anchor_90` multi-candidate (4 models: lightgbm, dayahead_proxy, naive_lag1, naive_lag7)
**Date range**: 2025-11-01 ~ 2026-02-28 (121 business days, 2880 timestamps)

---

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Weight modes | anchor (lightgbm@0.9), softmax (T=0.1), convex (scipy SLSQP) |
| Lookback days | 30 |
| Min history days | 10 |
| Ridge alpha | 1.0 |
| Metric level | Timestamp-level deduplicated (1 row per business_day + hour_business) |

## 2. Results

| Metric | anchor_90 | softmax | convex |
|--------|-----------|---------|--------|
| sMAPE (floor50) | 20.61 | **19.86** | 23.70 |
| sMAPE 9-16 | 25.46 | 26.31 | 28.96 |
| Severe underestimates | 82 | 83 | 152 |
| Timestamps | 2880 | 2880 | 2880 |

## 3. Comparison vs Phase 2

| Candidate | sMAPE | Severe | Δ sMAPE | Δ Severe |
|-----------|-------|--------|---------|----------|
| LightGBM-only | 22.02 | 80 | - | - |
| Phase 2 best (anchor_90 + medium) | 20.86 | 63 | -1.16 | -17 |
| P3 rolling anchor_90 | 20.61 | 82 | -0.25 | +19 |
| P3 rolling softmax | **19.86** | 83 | **-1.00** | +20 |
| P3 rolling convex | 23.70 | 152 | +2.84 | +89 |

## 4. Analysis

**sMAPE improvement**: Both anchor and softmax rolling modes improve sMAPE vs Phase 2 best (−0.25 and −1.00 respectively). Softmax achieves the best overall sMAPE at 19.86.

**Severe underestimate regression**: All rolling modes increase severe underestimates vs Phase 2 best (+19 to +89). The gap from 63 to 83 represents ~20 more hours where predictions miss actual by >200 MW.

**Root cause**: The rolling weight optimizers minimize RMSE/sMAPE, which doesn't directly penalize severe underestimates. LightGBM dominates the weights (0.9 in anchor, 0.4–0.8 in softmax), but the rolling re-weighting introduces day-level variance that occasionally shifts weight away from the most conservative model on high-spike days.

**Convex mode instability**: Convex weights frequently fall back to equal weights (0.25 each), suggesting the SLSQP optimizer struggles with 4-model colinearity on 30-day windows.

## 5. Verdict

**NO-GO** — Rolling 30D fusion does not meet Phase 2 GO thresholds:

| Criterion | Threshold | Best P3 | Met? |
|-----------|-----------|---------|------|
| sMAPE ≤ 20.86 | 20.86 | 19.86 | ✅ |
| Severe underestimates ≤ 63 | 63 | 82 | ❌ |
| False lift rate ≤ 10% | 10% | N/A | ⚠️ Not evaluated |
| Normal hours degradation ≤ 0.5 | 0.5 | N/A | ⚠️ Not evaluated |

## 6. Next Steps

1. **Rolling fusion fix**: Modify optimizer to include severe underestimate penalty (weighted loss)
2. **Alternative**: Increase anchor weight from 0.9 to 0.95 to reduce day-level variance
3. **SOTA lab**: LightGBM hyperparameter tuning may recover severe estimate performance
4. **Scoping decision**: If rolling fusion cannot beat Phase 2 static anchor on severe counts, consider deploying Phase 2 as-is and deferring rolling fusion to Phase 4
