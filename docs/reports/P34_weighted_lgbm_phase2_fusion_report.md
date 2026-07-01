# P3.4 Weighted-LightGBM + Phase2 Fusion/Correction Report

**Generated**: 2026-07-01
**Branch**: `agent/p34-weighted-lgbm-phase2-fusion`
**Base**: `origin/tune-timemixer`

---

## Verdict: **NO-GO** ❌

Weighted LightGBM (period_spike_weighted) degrades prediction quality on the full
120-day window. Fusion with Phase2 baselines and correction cannot recover the loss.

---

## Summary

| Field | Value |
|-------|-------|
| Weight profile | `period_spike_weighted` |
| Date range | 2025-11-01 → 2026-02-28 (120 days) |
| Fusion modes | weighted_lgbm_anchor_90, weighted_lgbm_anchor_80, custom |
| Correction profiles | conservative, medium, aggressive (all normal mode) |
| Configurations evaluated | 9 (3 fusion × 3 correction) |
| All verdicts | **NO-GO** (9/9) |

## Actual Standalone Metrics

The weighted LGBM predictions (full 120-day daily walk-forward):

| Model | sMAPE | Severe | MAE |
|-------|-------|--------|-----|
| **Standard LightGBM** (level1 pack) | **20.73** | **80** | 73.54 |
| **Weighted LGBM** (period_spike_weighted) | **24.34** | **146** | 94.74 |
| Weighted LGBM (first 15 days only) | 18.82 | 19 | — |

**Critical finding**: Weighted LGBM is **worse** than standard LGBM on the full 120-day window.
The spike weighting degrades sMAPE by +3.61 points and severe by +66 events.

The earlier p33 result (sMAPE 19.5, severe 20) was on an **easy 15-day window** (Nov 1-15).
On the full 120-day window, the weighting scheme does not generalize:
- High weight (8x) on 9_16 high-spike rows skews the model toward overfitting
- Per-day weight computation from volatile training windows creates inconsistency
- The model predicts conservatively (y_pred max 594 vs y_true max 1408), causing more severe underestimates

## All Results

| # | Fusion | Correction | sMAPE | Base | 9_16 | Severe | ΔSev | Degrad | False Lift | Lift | Verdict |
|---|--------|------------|-------|------|------|--------|------|--------|------------|------|---------|
| 1 | custom | conservative | 23.63 | 24.00 | 31.02 | 116 | -15 | -0.39 | 6.1% | 214 | NO-GO |
| 2 | custom | medium | 23.63 | 24.00 | 31.02 | 116 | -15 | -0.39 | 6.1% | 214 | NO-GO |
| 3 | custom | aggressive | 23.63 | 24.00 | 31.02 | 116 | -15 | -0.39 | 6.1% | 214 | NO-GO |
| 4 | weighted_lgbm_anchor_80 | conservative | 23.73 | 24.13 | 31.23 | 112 | -15 | -0.42 | 6.0% | 215 | NO-GO |
| 5 | weighted_lgbm_anchor_80 | medium | 23.73 | 24.13 | 31.23 | 112 | -15 | -0.42 | 6.0% | 215 | NO-GO |
| 6 | weighted_lgbm_anchor_80 | aggressive | 23.73 | 24.13 | 31.23 | 112 | -15 | -0.42 | 6.0% | 215 | NO-GO |
| 7 | weighted_lgbm_anchor_90 | conservative | 23.92 | 24.31 | 31.06 | 118 | -17 | -0.40 | 6.0% | 214 | NO-GO |
| 8 | weighted_lgbm_anchor_90 | medium | 23.92 | 24.31 | 31.06 | 118 | -17 | -0.40 | 6.0% | 214 | NO-GO |
| 9 | weighted_lgbm_anchor_90 | aggressive | 23.92 | 24.31 | 31.06 | 118 | -17 | -0.40 | 6.0% | 214 | NO-GO |

### Key Observations

1. **Correction barely helps**: severe reduces by only 15-17 events (146→130 range at timestamp level).
   Only 9/2880 timestamps have spike_prob > 0.8, so correction has few opportunities to activate.

2. **All correction profiles identical**: conservative/medium/aggressive produce identical sMAPE and severe
   because the guardrail blocks most corrections. The risk model's `spike_risk_score` does not align with
   the weighted LGBM's error pattern.

3. **Lower anchor weight helps slightly**: custom (85% weighted) > anchor_80 > anchor_90.
   This is because the standard baselines (dayahead_proxy, naive_lag7) outperform the weighted LGBM,
   so lower anchor weight improves results. This confirms weighted LGBM is the weak link.

## Comparison vs Baselines

| Candidate | sMAPE | Severe | Δ sMAPE vs Phase2 | Δ Severe vs Phase2 |
|-----------|-------|--------|-------------------|--------------------|
| Phase2 champion (anchor_90 + medium normal) | 20.86 | 63 | — | — |
| Standard LightGBM (no weighting) | 20.73 | 80 | -0.13 ✅ | +17 ❌ |
| Weighted LGBM standalone (period_spike_weighted) | 24.34 | 146 | +3.48 ❌ | +83 ❌ |
| **P3.4 best** (custom + conservative) | 23.63 | 116 | +2.77 ❌ | +53 ❌ |

## DEPLOY GO / RESEARCH GO Assessment

| Criterion | DEPLOY GO | RESEARCH GO | Best P3.4 | Met? |
|-----------|-----------|-------------|-----------|------|
| sMAPE | ≤ 20.50 | ≤ 20.00 | 23.63 | ❌ |
| Severe | ≤ 63 | ≤ 70 | 116 | ❌ |
| False Lift | ≤ 10% | ≤ 12% | 6.1% | ✅ |
| Normal Degradation | ≤ 0.5 | ≤ 1.0 | -0.39 | ✅ |

## Root Cause Analysis

### Why weighted LGBM fails on the full window

1. **p33 result was misleading**: The 15-day window (Nov 1-15) has fewer spike events and lower price
   volatility. Weighted LGBM achieved sMAPE 18.82 on this window, but degrades to 24.34 on the full
   120-day window. The remaining 105 days contain many more challenging spike events.

2. **Spike weighting skews optimization**: The 8x weight on 9_16 high-spike rows (period_spike_weighted
   profile) dominates the training loss. The model optimizes for these specific rows at the expense of
   overall prediction quality. Standard LGBM without weighting achieves better aggregate performance.

3. **Inconsistent per-day weights**: The weight thresholds (p90/p95 of each day's training window)
   vary across days. A value considered "high spike" on one day may not be on another, creating
   inconsistent training signal across the 120-day walk-forward.

4. **Prediction ceiling**: Weighted LGBM's max prediction is 594 vs y_true's 1408. The model
   under-predicts extreme spikes, causing severe=146 vs standard LGBM's 80.

### Why correction doesn't help

1. **Risk model misalignment**: Only 9/2880 timestamps have spike_prob > 0.8. The risk model was
   trained on standard LGBM errors, not weighted LGBM errors. Weighted LGBM's error pattern is
   different (more frequent, higher magnitude), so the risk score doesn't identify the right hours.

2. **Lift correction is too weak**: Even when correction activates, the max lift (350 for medium)
   is insufficient for the severe errors (often 400+ on spike days).

3. **Guardrail blocks aggressively**: Most correction attempts are blocked by the guardrail because
   the spike probability doesn't exceed the threshold (0.45-0.75 depending on profile).

## Conclusion

**P3.4 Line E is NO-GO.** The spike-weighted LightGBM approach does not improve on the standard
LightGBM baseline. On the contrary, it degrades performance significantly.

### Implications for P3

1. **Spike weighting as a standalone improvement path is not viable** on the full test window.
2. **Standard LightGBM (no weighting) is the correct base model** for Phase2 fusion.
3. **Phase2 champion (anchor_90 + medium correction, sMAPE=20.86, severe=63) remains the best
   known production candidate.**
4. **P3 has exhausted all candidate combination paths** — all P3.3 and P3.4 lines produce NO-GO.
5. **Recommendation**: Proceed to P3/Paper Decision Report. Deploy Phase2 champion.

## Files Changed

| File | Change |
|------|--------|
| `scripts/evaluate_p34_weighted_lgbm_phase2_fusion.py` | **New** — Full evaluation pipeline |
| `tests/test_p34_weighted_lgbm_fusion.py` | **New** — 28 unit tests |
| `docs/reports/P34_weighted_lgbm_phase2_fusion_report.md` | **New** — This report |
| `docs/p3_execution_board.md` | **Updated** — P3.4 NO-GO status |
