# P3.3 Severe-Constrained Correction Search — Report

> **Author**: Window A (automated grid search)
> **Date**: 2026-06-30
> **Branch**: `agent/p33-severe-constrained-correction`
> **Script**: `scripts/search_p33_severe_constrained_correction.py`

---

## Summary

Grid search over 192 correction parameter combos (5 profile families). **No candidate meets either DEPLOY GO or RESEARCH GO thresholds.**

The fundamental bottleneck is **risk model calibration** — not correction parameters.

---

## Grid Configuration

| Family | Combos | Params Searched |
|--------|--------|-----------------|
| `medium_plus` | 144 | All 4 params full grid |
| `medium_spike_only` | 4 | spike_prob_threshold only |
| `medium_916_boost` | 4 | period_9_16_boost only |
| `high_risk_only` | 24 | spike ≥ 0.55, lift_ratio ≥ 0.35, boost ∈ [1.0, 1.3] |
| `asymmetric_lift` | 16 | low threshold + capped lift |
| **Total** | **192** | |

### Param Range

| Param | Values |
|-------|--------|
| spike_prob_threshold | 0.45, 0.50, 0.55, 0.60 |
| max_lift_ratio | 0.25, 0.35, 0.45 |
| max_absolute_lift | 250, 350, 500 |
| period_9_16_boost | 1.0, 1.15, 1.30, 1.50 |
| protect_normal_hours | true (fixed) |
| correction_mode | normal (fixed, no relaxed) |

---

## Baseline Metrics

| Metric | Phase2 Best | P3.2 Medium | P3.3 Base (no correction) |
|--------|:-----------:|:-----------:|:-------------------------:|
| sMAPE | 20.86 | 20.74 | 21.00 |
| Severe | 63 | 73 | 88 |
| False lift | 0% | 7.6% | 0% |

---

## Results

### Overall Best Candidate (Combo 136)

| Metric | Value | vs Phase2 | vs P3.2 Medium |
|--------|:-----:|:---------:|:--------------:|
| sMAPE | **20.92** | +0.06 (worse) | +0.18 (worse) |
| Severe | **69** | +6 (worse) | -4 (better) |
| False lift | **7.6%** | +7.6% | 0% |
| Normal degradation | **-0.16** | N/A | 0% |
| Lift applied | **219/2880** | N/A | same |

**Best params**: spike_prob_threshold=0.60, max_lift_ratio=0.45, max_absolute_lift=250, period_9_16_boost=1.50

### Best sMAPE Candidate (Combo 133)

| Metric | Value |
|--------|:-----:|
| sMAPE | **20.73** |
| Severe | **72** |
| False lift | 7.6% |

**Params**: spike_prob_threshold=0.60, max_lift_ratio=0.45, max_absolute_lift=250, period_9_16_boost=1.0

### Per-Family Best

| Family | sMAPE | Severe | False Lift |
|--------|:-----:|:------:|:----------:|
| medium_plus | 20.92 | 69 | 7.6% |
| medium_spike_only | 20.74 | 73 | 7.6% |
| medium_916_boost | 20.92 | 71 | 7.6% |
| high_risk_only | 20.74 | 71 | 7.6% |
| asymmetric_lift | *all eliminated* | — | — |

### Hard Elimination Breakdown

- **139/192** eliminated by false_lift > 12%
- **45/192** eliminated by normal_hours_degradation > 0.5
- **53/192** eligible (both constraints met)
- **0/192** meet DEPLOY GO (sMAPE ≤ 20.50 AND severe ≤ 63)
- **0/192** meet RESEARCH GO (sMAPE ≤ 20.00 AND severe ≤ 70)

---

## GO Criteria Assessment

| Criterion | DEPLOY GO | Best Candidate | RESEARCH GO |
|-----------|:---------:|:--------------:|:-----------:|
| sMAPE | ≤ 20.50 | 20.92 ❌ | ≤ 20.00 |
| Severe | ≤ 63 | 69 ❌ | ≤ 70 |
| False lift | ≤ 10% | 7.6% ✅ | ≤ 12% |
| Normal degrad. | ≤ 0.5 | -0.16 ✅ | ≤ 1.0 |

**Verdict: NO-GO** — severe cannot be reduced below 69 with acceptable false_lift.

---

## Root Cause Analysis

### Risk Model Calibration Issue

The spike risk model (`high_spike_prob`) is poorly calibrated for this task:

| Threshold | Non-spike hours above | True spike hours above | Precision |
|:---------:|:---------------------:|:---------------------:|:---------:|
| 0.45 | 95.0% | 100.0% | 4.0% |
| 0.50 | 77.0% | 100.0% | 4.7% |
| 0.55 | 26.5% | 94.5% | 12.2% |
| 0.60 | 15.7% | 51.4% | 11.4% |

At threshold=0.60 (required for false_lift ≤ 12%), only **51.4%** of true spike hours are captured. This means:
- 56/109 true spikes detected
- 53 severe underestimates missed (severe drops from 88 to 69, not enough)
- Even with aggressive lift, can't catch what the risk model misses

### False Lift Cliff

```
threshold=0.55 → false_lift=18.89% (ELIMINATED)
threshold=0.60 → false_lift=7.60%  (eligible)
```

The 0.05 difference creates a 11.3pp jump in false_lift. This means ~300 non-spike rows have `high_spike_prob` between 0.55-0.60, and applying lift to any of them creates false lift.

### sMAPE-Severe Tradeoff

At threshold=0.60 (eligible range):
- Higher boost = lower severe but higher sMAPE
- Boost 1.0: severe=72, sMAPE=20.73
- Boost 1.15: severe=71, sMAPE=20.74
- Boost 1.50: severe=69, sMAPE=20.92

The marginal return of additional lift diminishes rapidly: capturing 3 more severe rows costs 0.19 sMAPE points.

---

## Output Files

| File | Description |
|------|-------------|
| `reports/local/p33_severe_constrained_correction/grid_results.csv` | All 192 combos with metrics |
| `reports/local/p33_severe_constrained_correction/best_candidates.json` | Best per family + overall |
| `reports/local/p33_severe_constrained_correction/top_candidates_converged.csv` | Top 30 eligible candidates |
| `reports/local/p33_severe_constrained_correction/grid_manifest.json` | Run config |
| `scripts/search_p33_severe_constrained_correction.py` | Grid search script |

---

## Recommendation

**P3.3 Line A — STOP. Do not merge for production.**

The correction pipeline alone cannot bridge severe from 73 → ≤ 63 given the current risk model calibration. To achieve the GO threshold, one of these upstream changes is needed:

1. **Prioritize P3.3 Line D**: Spike-weighted LightGBM retraining to improve base prediction quality on spike hours. Better base → fewer rows needing correction → less false lift.

2. **Prioritize P3.3 Line B**: Spike-gated uplift with a separate model trained specifically on spike hours to improve the precision-recall tradeoff.

3. **Deploy Phase 2 champion** (sMAPE=20.86, severe=63) as the production candidate. P3.3 makes no improvement over this baseline.

**Decision**: CONDITIONAL — P3.3 Line A search is conclusive. No viable DEPLOY GO or RESEARCH GO candidate exists via correction params alone.
