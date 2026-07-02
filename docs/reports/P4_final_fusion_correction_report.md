# P4 Final Fusion + Correction Report

**Date:** 2026-07-01
**Branch:** agent/p4-fusion-correction-finalizer
**Status:** COMPLETE — ALL NO-GO

---

## Objective

Combine Window 2 (best LightGBM quantile candidate) and Window 3 (best ML gate risk) with Phase2 fusion + correction to achieve DEPLOY GO.

| Criterion | DEPLOY GO Threshold | Phase2 Champion (Full) | Phase2 Champion (Overlap) |
|-----------|--------------------|------------------------|---------------------------|
| sMAPE floor50 | <= 20.50 | 20.8675 | 19.11 |
| Severe underestimates | <= 63 | 63 | 18 |
| False lift rate | <= 10% | 6.64% | 5.27% |
| Normal hours degradation | <= 0.5 | -0.19 | -0.20 |

**Note on sMAPE formula**: Uses canonical floor50 sMAPE: `max(|x|, 50)` on both y_true and y_pred in denominator, with floored values in numerator. Matches `canonical_metrics_baseline.json`.

## Inputs

| Input | Source | Path |
|-------|--------|------|
| W1 Canonical Pack | Phase2 canonical evaluation pack | `reports/local/p4_canonical/canonical_prediction_pack.csv` |
| W2 Best Candidate | LightGBM quantile alpha=0.8 (full) | `reports/local/p4_lgbm_sota_tuning/full_obj_quantile_0p8/predictions.csv` |
| Phase2 Risk | Canonical risk predictions | `reports/local/p4_canonical/canonical_risk_predictions.csv` |
| W3 Risk | ML gate risk predictions | `reports/local/p4_hybrid_gate/ml_gate/risk_predictions_gate.csv` |

W2 covers 61 days (2025-11-01 ~ 2025-12-31, 1464 timestamps). Full period is 120 days (2025-11-01 ~ 2026-02-28, 2879 timestamps).

## Combos

| # | Combo | Base | Risk | Window |
|---|-------|------|------|--------|
| A | Phase2 canonical baseline | Canonical `base_fused_pred` | Phase2 (via `final_pred_reference`) | Full + Overlap |
| B | W2 + Phase2 corr | Replace lightgbm → W2 quantile, recompute anchor_90 | Phase2 canonical | Overlap only |
| C | Phase2 + W3 gate | Canonical `base_fused_pred` | W3 ML gate | Full + Overlap |
| D | W2 + W3 gate | Replace lightgbm → W2 quantile, recompute anchor_90 | W3 ML gate | Overlap only |

**Key fix**: Combo A uses `final_pred_reference` directly from canonical pack (no re-correction). sMAPE uses the exact canonical floor50 formula.

## Full-Window Results (2025-11-01 ~ 2026-02-28, 2879 timestamps)

| Combo | sMAPE | Base sMAPE | Severe | False Lift | Normal Degrad | Verdict |
|-------|:-----:|:----------:|:------:|:----------:|:-------------:|:-------:|
| **A — Phase2 baseline** | **20.87** | **21.21** | **63** | **0.07** | **-0.19** | **NO-GO** |
| C — Phase2 + W3 gate | 21.13 | 21.21 | 73 | 0.10 | -0.07 | NO-GO |

Phase2 champion reference: sMAPE=20.8675, severe=63, false_lift=0.0664, normal_degrad=-0.1929.

**A matches champion exactly ✅** — sanity check passed.

C shows slight degradation: sMAPE +0.26, severe +10 vs champion. W3 ML gate risk does not improve over Phase2 risk on the full window.

## Overlap-Window Results (2025-11-01 ~ 2025-12-31, 1464 timestamps)

| Combo | sMAPE | Base sMAPE | Severe | False Lift | Normal Degrad | Verdict |
|-------|:-----:|:----------:|:------:|:----------:|:-------------:|:-------:|
| **A — Phase2 baseline** | **19.11** | **19.37** | **18** | **0.05** | **-0.20** | **DEPLOY GO*** |
| B — W2 + Phase2 corr | 24.84 | 24.73 | 21 | 0.05 | 0.18 | NO-GO |
| C — Phase2 + W3 gate | 19.33 | 19.37 | 19 | 0.07 | -0.02 | DEPLOY GO* |
| D — W2 + W3 gate | 24.77 | 24.73 | 26 | 0.07 | 0.06 | NO-GO |

**DEPLOY GO* = DEPLOY GO on overlap-window only.** Per protocol, DEPLOY GO is only assessed on full-window. Overlap-window verdicts are directional.

### Analysis

**Phase2 baseline (A) on overlap**: sMAPE=19.11, severe=18. The Nov-Dec period has inherently lower error than the full window (which includes Jan-Feb winter peak). This is the fair baseline for B/C/D comparison on overlap.

**W2 quantile (B, D)**: sMAPE ~24.8 vs A_overlap 19.11 — **severe degradation of ~5.7 sMAPE points**. The quantile alpha=0.8 model systematically over-predicts, matching the standalone W2 evaluation (full W2 run: sMAPE=27.17). The quantile approach is not viable for production fusion.

**W3 ML gate (C)**: sMAPE=19.33, severe=19 vs A 19.11, 18 — slightly worse on both metrics. On the overlap window, the W3 gate provides no improvement over Phase2 risk. On the full window, C is strictly worse (21.13/73 vs 20.87/63).

**W2 + W3 combined (D)**: Picks up the worst of both — high sMAPE from W2 quantile (24.77) with no compensating benefit from W3 gate.

## Verdict

**NO-GO** — No combination meets DEPLOY GO thresholds on the full-window.

| Combo | sMAPE <= 20.50 | Severe <= 63 | False lift <= 10% | Normal degrad <= 0.5 | Overall |
|-------|:--------------:|:------------:|:-----------------:|:--------------------:|:-------:|
| A | ❌ 20.87 | ✅ 63 | ✅ 6.6% | ✅ -0.19 | **NO-GO** |
| B | ❌ 24.84 | ✅ 21 | ✅ 5.3% | ✅ 0.18 | **NO-GO** |
| C | ❌ 21.13 | ❌ 73 | ✅ 10.1% | ✅ -0.07 | **NO-GO** |
| D | ❌ 24.77 | ✅ 26 | ✅ 7.0% | ✅ 0.06 | **NO-GO** |

## Recommendations

1. **Phase2 champion remains best candidate** — sMAPE=20.87, severe=63. No P4 combo beats it.
2. **W2 quantile model not suitable for fusion** — The sMAPE cost (~5.7 points on overlap) is too high. Severe reduction (18→21) does not compensate. Quantile approach abandoned for fusion.
3. **W3 ML gate needs retraining** — No improvement over Phase2 risk on either full or overlap window. If pursuing further, need to investigate feature set or training target mismatch.
4. **Phase2 champion should be deployment candidate** — Since no P4 candidate exceeds it, phase2 anchor_90 + medium correction (normal mode) remains production choice.

## Comparison Summary

| Aspect | Full-Window | Overlap-Window |
|--------|:-----------:|:--------------:|
| A — Phase2 baseline | 20.87 / 63 | 19.11 / 18 |
| B — W2 + Phase2 corr | — | 24.84 / 21 |
| C — Phase2 + W3 gate | 21.13 / 73 | 19.33 / 19 |
| D — W2 + W3 gate | — | 24.77 / 26 |

Format: sMAPE / severe. Full-window: 120 days. Overlap-window: 61 days (W2 coverage).
