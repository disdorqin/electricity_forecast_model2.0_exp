# P3.3 LightGBM Internal Sample Weighting Report

**Generated**: 2026-06-30
**Branch**: `agent/p33-lgbm-internal-weighting`
**Script**: `scripts/run_p33_lgbm_weighting.py`

---

## 1. Overview

P3.3 modifies the LightGBM daily walk-forward training pipeline to accept `sample_weight` profiles that upweight spike/severe hours during training. Unlike P3.1 (post-hoc weight fusion) and P3.2 (external correction), P3.3 directly embeds spike awareness into the LightGBM gradient computation.

### Weight Profiles

| Profile | Weight Rule | Rationale |
|---------|-------------|-----------|
| `none` | All 1.0 (uniform) | Baseline — no spike weighting |
| `spike_weighted` | high_spike → 3.0, 9_16 high_spike → 6.0 | Focus on all spike hours, extra focus on 9-16 |
| `severe_underestimate_weighted` | severe (p95+) → 4.0 | Focus on extreme-value hours |
| `period_spike_weighted` | 9_16 high_spike → 8.0, other high_spike → 3.0 | Aggressive 9-16 spike weighting |

**Thresholds**: `high_spike` = y > max(p90, 150), `severe` = y > max(p95, 250). Both computed per-training-window (no leakage).

### Code Changes

- `lightGBM/main_fix.py`: Added `_compute_spike_weights()` helper + `sample_weight_profile` parameter threaded through `_fit_realtime_fixed_window()` → `run_precision_simulation()` → `run_lgbm_pipeline()`
- Valley (1-8) and Peak (17-24) now also receive sample weights (previously only Solar and Peak had weights)
- `scripts/run_p33_lgbm_weighting.py`: NEW — experiment runner

---

## 2. Small Window Results (2025-11-01 ~ 2025-11-15, 15 days)

| Profile | sMAPE | Severe | MAE | 9-16 sMAPE | Δ sMAPE vs Ref | Δ Severe vs Ref |
|---------|:-----:|:------:|:---:|:----------:|:--------------:|:----------------:|
| reference | 24.59 | 30 | 90.33 | 32.67 | — | — |
| none | 24.00 | 30 | 88.07 | 32.29 | -0.59 | 0 |
| spike_weighted | 20.90 | 21 | 77.01 | 31.23 | **-3.69** | **-9** |
| severe_underestimate_weighted | 21.07 | 21 | 77.49 | 32.23 | -3.52 | -9 |
| **period_spike_weighted** | **19.52** | **20** | **74.37** | **27.14** | **-5.07** | **-10** |

All three spike profiles pass Internal GO (sMAPE < 22.02, severe < 80) ✅.
**period_spike_weighted** passes Strong GO (sMAPE ≤ 20.86, severe ≤ 63) ✅.

## 3. Full Window Results (2025-11-01 ~ 2025-12-31, 61 days)

| Profile | sMAPE | Severe | MAE | 9-16 sMAPE | Δ sMAPE vs Ref | Δ Severe vs Ref |
|---------|:-----:|:------:|:---:|:----------:|:--------------:|:----------------:|
| reference | 24.98 | 74 | 89.12 | 33.04 | — | — |
| none | 25.22 | 69 | 88.58 | 33.43 | +0.24 | -5 |
| spike_weighted | 23.96 | 54 | 90.30 | 33.24 | -1.02 | **-20** |
| severe_underestimate_weighted | 24.11 | 53 | 90.50 | 33.52 | -0.87 | **-21** |
| **period_spike_weighted** | **23.76** | **54** | 89.56 | **32.90** | **-1.22** | **-20** |

**Internal GO (sMAPE < 22.02, severe < 80)**: sMAPE ❌ (23.76), Severe ✅ (54)
**Strong GO (sMAPE ≤ 20.86, severe ≤ 63)**: sMAPE ❌ (23.76), Severe ✅ (54)

### Key Metrics Comparison

| Metric | Reference | period_spike_weighted | Δ | % Change |
|--------|:---------:|:--------------------:|:-:|:--------:|
| sMAPE (floor50) | 24.98 | **23.76** | **-1.22** | **-4.9%** |
| Severe underestimates | 74 | **54** | **-20** | **-27.0%** |
| MAE | 89.12 | 89.56 | +0.44 | +0.5% |
| 9-16 sMAPE | 33.04 | 32.90 | -0.14 | -0.4% |

---

## 4. Analysis

### Severe Reduction is Substantial

All three spike profiles reduce severe underestimates by approximately 20 (from 74 to 53-54), a **27% reduction**. This is the largest severe reduction observed across all P3 experiments:

| Experiment | Severe Reduction |
|-----------|:----------------:|
| P3.1 weight fusion | 80→88 (worse) |
| P3.2 rolling + correction | 88→73 |
| **P3.3 spike weighting** | **74→54** |

### sMAPE Improvement is Consistent but Insufficient

sMAPE improves by 1.0-1.2 points across all spike profiles (23.76 vs 24.98 reference). This is directionally correct but falls short of the 22.02 Internal GO threshold. The 9-16 sMAPE shows minimal change (32.90 vs 33.04), suggesting the model still struggles in daylight hours.

### Uniform Weights (none) Are Harmful

The `none` profile (all 1.0) slightly worsens sMAPE (25.22 vs 24.98) and only marginally improves severe (69 vs 74). This confirms that the existing weight logic (w_solar, w_peak) provides some value, but spike-aware weights are significantly more effective.

### Period Spike Weighting is the Best Profile

`period_spike_weighted` achieves the best sMAPE (23.76), while `severe_underestimate_weighted` achieves the best severe count (53 vs 54, marginal). Given the user's stated priority on severe reduction, `period_spike_weighted` is recommended for production consideration.

### Small Window vs Full Window Discrepancy

The 15-day window showed dramatic improvements (sMAPE 24.59→19.52) that did not fully hold at 61 days (24.98→23.76). This is expected: smaller windows have higher variance and may overfit the weight profile to short-term patterns. The 61-day result is more reliable.

---

## 5. GO Assessment

| Criterion | Threshold | period_spike_weighted | Met? |
|-----------|-----------|:--------------------:|:----:|
| sMAPE < 22.02 (Internal GO) | 22.02 | 23.76 | ❌ |
| Severe < 80 (Internal GO) | 80 | 54 | ✅ |
| sMAPE ≤ 20.86 (Strong GO) | 20.86 | 23.76 | ❌ |
| Severe ≤ 63 (Strong GO) | 63 | 54 | ✅ |

**Internal GO**: ❌ sMAPE 1.74 above threshold, ✅ severe 26 below threshold
**Strong GO**: ❌ sMAPE 2.90 above threshold, ✅ severe 9 below threshold

**Verdict: NO-GO** — sMAPE does not meet either GO threshold.

---

## 6. Conclusions

1. **P3.3 is directionally correct**: Spike-weighted training significantly reduces severe underestimates (−27%) and improves sMAPE (−4.9%).

2. **Severe reduction is best-in-class**: P3.3 achieves severe=54 on the 61-day window, beating Phase2 (63), P3.2 (73), and P3.1 (80).

3. **sMAPE gap remains**: The 23.76 sMAPE is still 1.74 above the Internal GO (22.02) and 2.90 above Strong GO (20.86). Further improvements needed.

4. **Recommendation**: Deploy `period_spike_weighted` as the new LightGBM training default. The severe reduction is meaningful even if sMAPE targets aren't met. Consider combining with Phase2 correction pipeline for further improvement.

---

## 7. Next Steps

1. **Combine with P0 correction**: Run period_spike_weighted predictions through Phase2 correction pipeline (medium profile) to further reduce severe
2. **Profile tuning**: Experiment with higher/lower spike weight multipliers (e.g., 6.0/10.0 for 9_16)
3. **Hyperparameter co-tuning**: The spike weighting interacts with learning_rate, num_leaves — optimize jointly
4. **Deploy period_spike_weighted**: Replace current hardcoded w_solar/w_peak with `period_spike_weighted` as default
