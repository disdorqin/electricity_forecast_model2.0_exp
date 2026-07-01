# P4 Final Fusion + Correction Report

**Date:** 2026-07-01
**Branch:** agent/p4-fusion-correction-finalizer
**Status:** COMPLETE — NO-GO

---

## Objective

Combine Window 2 (best LightGBM quantile candidate) and Window 3 (best ML gate risk) with Phase2 fusion + correction to achieve DEPLOY GO.

| Criterion | DEPLOY GO Threshold | Phase2 Champion |
|-----------|--------------------|------------------|
| sMAPE | <= 20.50 | 20.86 |
| Severe underestimates | <= 63 | 63 |
| False lift rate | <= 10% | — |
| Normal hours degradation | <= 0.5 | — |

## Inputs

| Input | Source | Path |
|-------|--------|------|
| W1 Canonical Pack | Phase2 canonical evaluation pack | `reports/local/p4_canonical/canonical_prediction_pack.csv` |
| W2 Best Candidate | LightGBM quantile alpha=0.8 (full) | `reports/local/p4_lgbm_sota_tuning/full_obj_quantile_0p8/predictions.csv` |
| Phase2 Risk | Canonical risk predictions | `reports/local/p4_canonical/canonical_risk_predictions.csv` |
| W3 Risk | ML gate risk predictions | `reports/local/p4_hybrid_gate/ml_gate/risk_predictions_gate.csv` |

Note: W2 covers 62 days (2025-11-01 to 2026-01-01, 1464 timestamps). Full period is 120 days (2025-11-01 to 2026-02-28, 2879 timestamps).

## Combinations Evaluated

| # | Combo | Base | Risk | Period | sMAPE | Severe | False Lift | Normal Degrad | Verdict |
|---|-------|------|------|--------|:-----:|:------:|:----------:|:-------------:|:-------:|
| A | phase2_baseline | Canonical `base_fused_pred` | Phase2 | full | **22.34** | **63** | 0.08 | -0.20 | NO-GO |
| B | w2_phase2_risk | W2 anchor fusion (0.9*W2 + 0.1*baselines) | Phase2 | W2 | **26.23** | **22** | 0.06 | 0.08 | NO-GO |
| C | phase2_w3_risk | Canonical `base_fused_pred` | W3 ML gate | full | **22.60** | **73** | 0.10 | -0.07 | NO-GO |
| D | w2_w3_risk | W2 anchor fusion | W3 ML gate | W2 | **26.14** | **27** | 0.07 | 0.05 | NO-GO |

Reference: Phase2 champion sMAPE=20.86, severe=63.

## Analysis

### Finding 1: Phase2 baseline re-evaluates at sMAPE=22.34 (vs 20.86 champion)

The canonical pack's `base_fused_pred` uses the same Phase2 anchor_90 fusion formula (0.9*lightgbm + 0.1*mean of baselines), but the corrected sMAPE measures at 22.34. This suggests the Phase2 champion's 20.86 was from a different evaluation setup (e.g., different risk file, different dedup, or a slightly different date range). On this canonical pack, the correction-only improvement is marginal (22.68 base -> 22.34 corrected = -0.34 sMAPE improvement).

### Finding 2: W2 quantile 0.8 degrades sMAPE severely (26.23 vs 20.86)

The LightGBM quantile alpha=0.8 model trades sMAPE for severe reduction. Severe drops from 63 to 22 (good), but sMAPE balloons to 26.23 (bad). The quantile model systematically over-predicts, causing high error across all hours. This tradeoff is unacceptable for DEPLOY GO.

### Finding 3: W3 ML gate risk does not improve over Phase2 risk

Phase2 base + W3 risk: sMAPE=22.60, severe=73 (worse than Phase2 risk: 22.34, 63). The ML gate's `high_spike_prob` values do not provide better correction targeting than the Phase2 risk model. Both metrics degrade across the board.

### Finding 4: W2 + W3 combined is the worst of both worlds

sMAPE=26.14, severe=27. The quantile model dominates the base_fused_pred (high sMAPE) and W3 risk doesn't help correct effectively.

## Verdict

**NO-GO** — No combination meets DEPLOY GO. All 4 combos fail on sMAPE (none < 20.50).

| Combo | sMAPE <= 20.50 | Severe <= 63 | False lift <= 10% | Normal degrad <= 0.5 |
|-------|:--------------:|:------------:|:-----------------:|:--------------------:|
| A | ❌ 22.34 | ✅ 63 | ✅ 0.08 | ✅ -0.20 |
| B | ❌ 26.23 | ✅ 22 | ✅ 0.06 | ✅ 0.08 |
| C | ❌ 22.60 | ❌ 73 | ✅ 0.10 | ✅ -0.07 |
| D | ❌ 26.14 | ✅ 27 | ✅ 0.07 | ✅ 0.05 |

## Recommendations

1. **Phase 2 champion remains best candidate** — Phase2 anchor_90 + medium correction (normal mode) is still the best known configuration.
2. **W2 quantile model unsuitable** — The sMAPE cost of quantile-based predictions is too high. If severe reduction is needed, it must come from correction, not from biased base predictions.
3. **W3 ML gate needs improvement** — Risk predictions that degrade both sMAPE and severe vs Phase2 risk need retraining or different features.
4. **Deploy Phase2 champion as production candidate** — Since no P4 candidate beats Phase2, fall back to Phase2 champion.
