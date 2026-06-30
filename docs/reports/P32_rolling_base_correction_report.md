# P3.2 Rolling Base + Spike Correction Report

**Generated**: 2026-06-30
**Branch**: `agent/p3-rolling-fusion-sota`
**Script**: `scripts/evaluate_p32_rolling_base_correction.py`

---

## 1. Overview

P3.2 combines P3.1 rolling severe_softmax base predictions with Phase2 spike correction. The goal is to retain rolling's sMAPE improvement while reducing severe underestimates to ≤ 63 (Phase2 best).

### Pipeline

1. Read P3.1 rolling predictions (severe_softmax) → 2880 timestamps
2. Build prediction pack with rolling `base_fused_pred` + `y_true`
3. Merge with Phase2 risk predictions (`high_spike_prob`)
4. Apply correction (medium / conservative / aggressive profiles, normal mode only)
5. Compute metrics vs Phase2 and P3.1 baselines

---

## 2. Results

| Metric | Phase2 Best | P3.1 Rolling | P3.2 Medium | P3.2 Cons. | P3.2 Aggr. |
|--------|:-----------:|:------------:|:-----------:|:----------:|:----------:|
| sMAPE (floor50) | 20.86 | 21.00 | **20.74** | 20.99 | 23.84 |
| Severe underestimates | **63** | 88 | **73** | 86 | 64 |
| 9_16 sMAPE (floor50) | 25.46 | 29.58 | 29.11 | 29.53 | 31.50 |
| High-spike MAE | — | — | 253.80 | 274.42 | 233.90 |
| False lift rate | — | — | **0.076** | 0.004 | 0.749 |
| Normal hours degradation | — | — | **-0.16** | 0.00 | 3.30 |
| Lift applied count | — | — | 219 | 11 | 2158 |
| Total hours | 2880 | 2880 | 2880 | 2880 | 2880 |

### Profiles defined

| Profile | spike_prob_threshold | max_lift_ratio | max_absolute_lift | protect_normal_hours | period_9_16_boost |
|---------|:-------------------:|:--------------:|:-----------------:|:-------------------:|:-----------------:|
| Conservative | 0.75 | 0.20 | 200 | true | 1.00 |
| Medium | 0.60 | 0.35 | 350 | true | 1.15 |
| Aggressive | 0.45 | 0.60 | 600 | true | 1.30 |

---

## 3. Analysis

### sMAPE

- **P3.2 Medium** achieves the best sMAPE at 20.74, slightly better than Phase2 best (20.86) and P3.1 rolling base (21.00).
- Correction improves sMAPE in the medium profile (20.74 vs 21.00 base) — unusual, but reflects that the correction lift is modest and reduces large errors.
- Aggressive profile destroys sMAPE (23.84) due to overcorrection (+3.30 normal hours degradation).

### Severe Underestimates

- Correction monotonically reduces severe counts: 88 (base) → 86 (conservative) → 73 (medium) → 64 (aggressive).
- **Medium** reduces severe from 88 to 73 (−15, −17%) but remains 10 above the 63 threshold.
- **Aggressive** nearly hits 63 (achieved 64) but at unacceptable cost to sMAPE and false lift rate.
- The Phase2 correction pipeline works on rolling base predictions: lift targets the right hours.

### False Lift Rate

- **Medium** achieves 7.6% false lift rate (within 10% threshold ✅).
- **Conservative** is extremely safe (0.4% false lift) but barely activates (only 11 lifts).
- **Aggressive** is unusable (75% false lift — lifts on most hours regardless of true spike).

### Key Finding: P3.1 sMAPE Re-evaluation

The P3.1 severe_softmax rolling base was originally reported at sMAPE=19.10 based on per-timestamp sMAPE averaging. When re-evaluated using the standard Phase2 pipeline's overall sMAPE computation (which matches the GO decision metric), the correct value is **21.00** — slightly worse than Phase2 best (20.86). The rolling predictions improve model-level sMAPE but the overall metric is not better than the Phase2 static anchor.

---

## 4. GO / NO-GO Assessment

| Criterion | Threshold | P3.2 Medium | Met? |
|-----------|-----------|:-----------:|:----:|
| sMAPE ≤ 19.50 | 19.50 | 20.74 | ❌ |
| Severe underestimates ≤ 63 | 63 | 73 | ❌ |
| False lift rate ≤ 10% | 0.10 | 0.076 | ✅ |
| Normal hours degradation ≤ 0.5 | 0.50 | -0.16 | ✅ |

**Verdict: NO-GO** — No P3.2 profile achieves both sMAPE ≤ 19.50 and severe ≤ 63.

| Profile | sMAPE | Severe | Criteria Met | Verdict |
|---------|:----:|:------:|:------------:|:-------:|
| Medium | 20.74 | 73 | 2/4 | NO-GO |
| Conservative | 20.99 | 86 | 2/4 | NO-GO |
| Aggressive | 23.84 | 64 | 0/4 | NO-GO |

---

## 5. Comparison to Previous Phases

| Phase | Candidate | sMAPE | Severe | Δ sMAPE (vs P2) | Δ Severe (vs P2) |
|-------|-----------|:-----:|:------:|:---------------:|:----------------:|
| Phase 2 | lightgbm_anchor_90 + medium | 20.86 | 63 | — | — |
| P3.1 | Rolling severe_softmax (base only) | 21.00 | 88 | +0.14 | +25 |
| P3.2 | Rolling + medium correction | **20.74** | 73 | **−0.12** | +10 |
| P3.2 | Rolling + aggressive correction | 23.84 | **64** | +2.98 | +1 |

P3.2 medium achieves the **best overall sMAPE** across all experiments (20.74) but cannot match Phase2 severe count (63).

---

## 6. Conclusions

1. **Combining rolling base + correction is directionally correct**: medium correction reduces severe from 88→73 while improving sMAPE. The pipeline works.

2. **Weight-only approaches are insufficient**: P3.1 showed that re-weighting existing model predictions can't reduce severe below the best individual model's inherent severe count (~80). Correction is necessary.

3. **Correction on rolling base ≠ Phase2 on anchor**: When correction is applied to the rolling base (sMAPE=21.00), it achieves sMAPE=20.74 and severe=73. When applied to Phase2 anchor (sMAPE=22.02 base), it achieved sMAPE=20.86 and severe=63. The rolling base responds differently to correction.

4. **The severe gap is structural**: To reach severe ≤ 63 while keeping sMAPE ≤ 19.50, the base predictions themselves need improvement (not just correction). Getting sMAPE ≤ 19.50 from a corrected base of 21.00 is unlikely — correction typically adds 0.5-1.5 sMAPE points at most.

5. **Next best option**: Phase 2 lightgbm_anchor_90 + medium correction (sMAPE=20.86, severe=63) remains the best known candidate.

---

## 7. Next Steps

1. ❌ P3 combined rolling + correction: NO-GO — does not meet either primary threshold
2. Consider: Focus on base model improvements (LightGBM tuning) rather than fusion/correction combinations
3. Consider: Accept Phase 2 as best known candidate and move to deployment readiness
4. Alternative: Modify correction profile specifically tuned for rolling base (e.g., lower spike_prob_threshold with higher max_lift_ratio)
