# P3–P4 Execution Board

> **Purpose**: Track all tasks across P3 (complete) and P4 (active) execution pipeline.
> **Status**: `[P3a–P3.4 COMPLETE (all NO-GO) — P4 FIVE-WINDOW FOCUS SPRINT ACTIVE]`
> **Branch**: `tune-timemixer`
> **Deployment champion**: Phase 2 lightgbm_anchor_90 + medium + normal (sMAPE=20.86, severe=63)
> **PR #10**: ✅ MERGED — P3 tooling/leakage-fix/rolling framework
> **PR #11**: OPEN — P3.1 severe-aware rolling (NO-GO, low-priority tooling merge)
> **PR #12**: ✅ MERGED — LightGBM internal spike-weighting tooling
> **PR #13**: OPEN — P3.3 spike-gated uplift (needs sync + merge)

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **W0: Runner / 总控** | `tune-timemixer` | P4 five-window focus sprint coordinator | `ACTIVE` |
| **W1: Data + Pack Auditor** | `tune-timemixer` | Audit feature leakage, pack quality, data integrity | `ACTIVE` |
| **W2: SOTA Model Tuning** | `agent/p4-lgbm-sota-tuning` | LightGBM hyperparameter grid search, RT916/TimesFM eval | `ACTIVE` |
| **W3: Spike Module / Risk Gate** | `agent/p4-canonical-eval-pack` | ML + Rule + Hybrid spike gate evaluation | `COMPLETE — RESEARCH GO` |
| **W4: Fusion + Correction Finalizer** | `agent/p4-fusion-correction-finalizer` | Final fusion + correction pipeline tuning, pending W2/W3 inputs | `ACTIVE` |
| **P3.1 (Severe-Aware)** | `agent/p31-severe-aware-rolling` | 3 severe-aware modes | `NO-GO — PR #11 OPEN` |
| **P3.2 (Rolling+Correction)** | `tune-timemixer` | rolling + correction combo | `✅ MERGED via PR #10 (NO-GO)` |
| **P3.4 Line G (TimesFM Smoke)** | `agent/p34-timesfm-diversity-smoke` | TimesFM diversity test | `COMPLETE — NO-GO` |

---

## Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| P3a–P3.4 (all P3) | COMPLETE | All NO-GO — see history below |
| P4 W3: Hybrid Spike Gate | RESEARCH GO | severe=56, false_lift=9.06%, sMAPE=22.43 |
| P4 W2: Quantile LGBM | Single-model GO | sMAPE=18.61, severe=12 (small window). Full window: sMAPE=27.17 |
| P4 W1: Data + Pack Audit | PENDING | — |
| P4 W4: Fusion + Correction | PENDING | Awaiting W2/W3 inputs |

---

## P4 Five-Window Focus Sprint

> **Goal**: Systematic parallel investigation across 5 windows to find a configuration that beats Phase 2 champion.
> **Champion baseline**: lightgbm_anchor_90 + medium + normal — sMAPE=20.86, severe=63, false_lift≤10%.

### P4 DEPLOY GO

| Criteria | Threshold |
|----------|-----------|
| sMAPE | ≤ 20.50 |
| severe | ≤ 63 |
| false_lift | ≤ 10% |
| normal_degradation | ≤ 0.5 |

### P4 PAPER GO

Single-model or core-module improves sMAPE by ≥ 1.0 vs its fair baseline AND has clear technical novelty.

### Window Assignments

| Window | Role | Branch | Scope | Status |
|--------|------|--------|-------|--------|
| **W0** | Runner / 总控 | `tune-timemixer` | Maintain board, receive results, adjudicate GO/NO-GO | `ACTIVE` |
| **W1** | Data + Pack Auditor | `tune-timemixer` | Audit feature leakage, pack quality, data integrity issues | `ACTIVE` |
| **W2** | SOTA Model Tuning | `agent/p4-lgbm-sota-tuning` | LightGBM hyperparameter grid search, RT916/TimesFM eval | `ACTIVE` |
| **W3** | Spike Module / Risk Gate | `agent/p4-canonical-eval-pack` | ML + Rule + Hybrid spike gate evaluation | `COMPLETE — RESEARCH GO` |
| **W4** | Fusion + Correction Finalizer | `agent/p4-fusion-correction-finalizer` | Final fusion + correction pipeline tuning | `ACTIVE — PENDING INPUTS (W2, W3)` |

### Results Log

| Date | Window | Result | Verdict |
|------|--------|--------|---------|
| 2026-07-01 | **W2** | Quantile α=0.8 LGBM: Single-model sMAPE=18.61, severe=12 (small window). Full window: sMAPE=27.17, severe=25. | **Single-model GO** — base model improvement. Needs fusion + correction for DEPLOY GO. |
| 2026-06-30 | **W3** | ml_gate+aggressive → severe=56 ✅, false_lift=9.06% ✅, sMAPE=22.43. Reduces severe by 7 vs baseline medium (63→56). | **RESEARCH GO** — severe + false_lift met, sMAPE limited by base model. |

---

### P4 W3: Hybrid Spike Gate — Detailed Results

> **Goal**: Replace inflated old risk model with a hybrid ML+Rule gate to improve severe recall while maintaining false_lift ≤ 10%.
> **Three gates**: ml_gate (RF), rule_gate (heuristic), hybrid_gate (0.6×ML + 0.4×rule).

| Config | sMAPE | Severe | False Lift | Recall | Lifted |
|--------|:-----:|:------:|:----------:|:------:|:------:|
| **ml_gate + aggressive** | **22.43** | **56** | **0.0906** | **0.80** | **466** |
| hybrid_gate + aggressive | 22.38 | 61 | 0.0829 | 0.76 | 434 |
| ml_gate + medium | 22.60 | 73 | 0.0378 | 0.64 | 283 |
| baseline old_risk + medium | 22.34 | 63 | 0.0702 | 0.15 | 225 |

**Key insight**: ML gate probabilities better calibrated (mean 0.21 vs old risk 0.53). Need aggressive profile (threshold 0.40).

**Verdict**: RESEARCH GO — severe=56 beats target, false_lift under 10%, sMAPE limited by base model accuracy.

---

### P4 W2: Quantile LightGBM Results

| Date | Combo | sMAPE | Severe | Verdict |
|------|-------|:-----:|:------:|:-------:|
| 2025-11-01~2025-11-15 (small) | obj_quantile_0p8 | 18.6124 | 12 | GO ✅ |
| 2025-11-01~2025-12-31 (full) | obj_quantile_0p8 | 27.1655 | 25 | see report |

**Note**: Single-model quantile LGBM outperforms baseline on sub-period. Full-window regresses. W2 output → W4 finalizer.

---

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | INACTIVE |
| B21 | SOTA model experiments | DEFERRED — now P4 W2 |
| B22 | P3.2 rolling + correction NO-GO | CLOSED |
| B23 | P3.1 sMAPE re-evaluation | CLOSED |
| B24 | No approach beats Phase 2 simultaneously on sMAPE + severe | **OPEN — P4 sprint objective** |

## Next Actions

1. ✅ P3 tooling merged: PR #10, PR #12
2. ⏳ PR #11: P3.1 severe-aware rolling — low-priority merge
3. ⏳ PR #13: spike-gated uplift — needs sync + merge
4. ✅ P4 W3: Hybrid Spike Gate complete — RESEARCH GO
5. 🏃 **P4 W2**: Quantile LGBM tuning — single-model GO, awaiting full pipeline eval
6. 🏃 **P4 W1**: Data + Pack audit
7. 🏃 **P4 W4**: Await W2/W3 inputs for final fusion + correction
8. 🎯 **Decision gate**: First DEPLOY GO candidate → new champion
9. 🎯 **Fallback**: If no P4 candidate beats Phase 2 → deploy Phase 2 as production
