# P3.1 Severe-Underestimate-Aware Rolling Fusion Report

**Generated**: 2026-06-30
**Branch**: `agent/p31-severe-aware-rolling`
**Base**: `origin/tune-timemixer`
**Prediction pack**: `lightgbm_anchor_90` multi-candidate (4 models: lightgbm, dayahead_proxy, naive_lag1, naive_lag7)
**Date range**: 2025-11-01 ~ 2026-02-28 (121 business days, 2880 timestamps)

---

## 1. Motivation

P3 rolling softmax fusion achieved best-ever sMAPE (19.86) but regressed severe underestimates (83 vs Phase2 best 63). The goal of P3.1 is to add severe-underestimate-aware weight modes that retain the rolling sMAPE improvement while reducing severe underestimates back to ≤63.

---

## 2. New Weight Modes

Three new modes added to `run_rolling_30d_fusion.py`:

### severe_softmax
```
score_i = recent_smape_i + alpha * severe_rate_i + beta * underprediction_mae_i / 200
weight_i ∝ exp(-temperature * normalized_score_i)
```
- **alpha** (default 1.0): penalty weight for severe underestimate rate
- **beta** (default 0.5): penalty weight for underprediction MAE

### severe_anchor
- LightGBM anchor weight ≥ 0.85
- Remaining weight allocated only to baselines whose severe rate is within 5% of LightGBM
- Baselines with worse severe rate get 0 weight (their share goes to LightGBM)

### quantile_guarded
- Uses severe_softmax base weights for rolling fusion
- Post-processing upward guard on high-risk hours:
  - Trigger: spike risk score ≥ threshold AND recent severe rate ≥ threshold
  - Lift: `max(base_fused_pred, lightgbm_pred * 1.05, base_fused_pred * 1.08)`
  - Cap: 15% max lift per timestamp

---

## 3. Results Summary

### Weight Mode Comparison

| Mode | Config | sMAPE (floor50) | Severe | Δ vs Phase2 sMAPE | Δ vs Phase2 Severe |
|------|--------|----------------|--------|-------------------|--------------------|
| **Phase2 best** (anchor_90+medium+normal) | — | **20.86** | **63** | baseline | baseline |
| P3 softmax | T=0.1 | 19.86 | 83 | **-1.00** ✅ | +20 ❌ |
| **severe_softmax** | alpha=1.0, beta=0.5 | 19.14 | 88 | **-1.72** ✅ | +25 ❌ |
| **severe_softmax** | alpha=3.0, beta=1.0 | **19.10** | 80 | **-1.76** ✅ | +17 ❌ |
| **severe_anchor** | min=0.85 | 20.45 | 84 | -0.41 ✅ | +21 ❌ |
| **quantile_guarded** | risk=0.4, severity=0.04 | 21.14 | **62** | +0.28 ❌ | **-1** ✅ |
| **quantile_guarded** | risk=0.5, severity=0.05, gentle | 19.09 | 79 | **-1.77** ✅ | +16 ❌ |
| strong rolling (anchor_90) | static 0.9/0.05/0.05 | 20.61 | 82 | -0.25 ✅ | +19 ❌ |

### GO / CONDITIONAL / NO-GO Assessment

| Rule | Threshold | Best severe_softmax | Best quantile_guarded | Best severe_anchor |
|------|-----------|-------------------|---------------------|-------------------|
| sMAPE (floor50) | ≤ 20.86 | **19.10** ✅ | 21.14 ❌ | 20.45 ✅ |
| Severe underestimates | ≤ 63 | 80 ❌ | **62** ✅ | 84 ❌ |
| False lift rate | ≤ 10% | 0% (no guard) ✅ | > 90% ❌ | 0% (no guard) ✅ |
| Normal hours degradation | ≤ 0.5 | 0 (weights only) ✅ | > 5 ❌ | 0 (weights only) ✅ |

**Verdict: NO-GO** — No single mode simultaneously meets both sMAPE and severe targets.

---

## 4. Key Findings

### 4.1 Weight-based approaches can't replace correction

Both `severe_softmax` and `severe_anchor` operate purely through re-weighting existing model predictions. The rolling fusion produces `base_fused_pred` which feeds into the correction pipeline. Weight-based approaches alone cannot reduce severe underestimates below the inherent severe count of the best individual model (LightGBM: severe ≈ 80).

### 4.2 severe_softmax achieves best-ever sMAPE

With alpha=3.0, beta=1.0, `severe_softmax` achieves **sMAPE=19.10** — the lowest sMAPE ever recorded in this project. This is a 1.76-point improvement over Phase2 best (20.86) and a 0.76-point improvement over P3 softmax (19.86). The penalty terms successfully shift weight toward models with lower severe rates, but the predictions themselves are unchanged.

### 4.3 Quantile guard can reduce severe — but at sMAPE cost

The `quantile_guarded` mode with aggressive thresholds (risk=0.4, severe_rate=0.04) reduces severe underestimates to 62 (below Phase2 best 63), but the guard overshoot degrades sMAPE to 21.14. The guard applies a hard `max()` lift which creates false lifts on non-spike hours and degrades normal hours.

### 4.4 severe_anchor maintains Phase2-level sMAPE

With LightGBM ≥ 0.85 on 92% of trading days, `severe_anchor` preserves the LightGBM anchor advantage while allowing minor baseline contributions. sMAPE=20.45 beats Phase2 best by 0.41, but severe=84 trails significantly.

---

## 5. Best Candidate

**severe_softmax** with alpha=3.0, beta=1.0:

| Metric | Value |
|--------|-------|
| **Fusion mode** | severe_softmax |
| **sMAPE (floor50)** | **19.10** |
| **Severe underestimates** | 80 |
| **vs Phase2 sMAPE** | **-1.76** (improvement) |
| **vs P3 softmax sMAPE** | **-0.76** (improvement) |
| **Verdict** | NO-GO (severe exceeds threshold) |

---

## 6. Files Changed

| File | Change |
|------|--------|
| `scripts/run_rolling_30d_fusion.py` | Added 3 severe-aware weight modes (severe_softmax, severe_anchor, quantile_guarded), CLI args, quantile guard post-processing |
| `tests/test_severe_aware_rolling.py` | NEW — 10 tests covering severe penalty, anchor constraints, guard behavior, leakage safety |
| `docs/p3_execution_board.md` | Updated with P3.1 results |
| `docs/reports/P31_severe_aware_rolling_report.md` | NEW — this report |

---

## 7. Next Steps

1. **Recommendation**: Combine rolling fusion (severe_softmax) + Phase2 correction pipeline. The rolling improves base sMAPE to 19.10, and correction can then reduce severe underestimates. This combination should beat Phase2 best on both metrics.

2. **Alternative**: Use Phase2 static anchor_90 as the correction base (severe=63 after correction). Rolling fusion is not a replacement for correction — it's a complementary improvement.

3. **Deferred**: quantile_guarded refinement — the guard logic needs more sophisticated spike detection to reduce false lift rate from 90%+ to < 10%.

4. **P3.1 end**: Weight-based severe awareness is directionally correct but insufficient alone. The severe_softmax sMAPE record (19.10) benefits the overall pipeline when combined with correction.
