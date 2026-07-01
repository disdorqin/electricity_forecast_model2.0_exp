# P3 Rolling Fusion + SOTA Lab — Execution Board

> **Purpose**: Track all tasks across the P3 execution pipeline.
> **Status**: `[P3a–P3.3 COMPLETE (all NO-GO) — P3.4 CANDIDATE COMBINATION ACTIVE]`
> **Branch**: `tune-timemixer`
> **Deployment champion**: Phase 2 lightgbm_anchor_90 + medium + normal (sMAPE=20.86, severe=63)
> **PR #10**: ✅ MERGED — tooling/leakage-fix/experiment framework
> **PR #11**: ✅ MERGED — P3.1 severe-aware rolling (NO-GO, research tooling)

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `tune-timemixer` | P3.4 candidate combination sprint | `ACTIVE` |
| **E: Weighted-LGBM + Fusion** | `agent/p34-weighted-lgbm-phase2-fusion` | Weighted-LGBM base → Phase2 fusion/correction | `EVALUATING` |
| **F: PR Cleanup + Tooling Merge** | `tune-timemixer` | PR #11/#12/#13 merge, NO-GO verdicts | `ACTIVE` |
| **G: TimesFM Diversity Smoke** | `agent/p34-timesfm-diversity-smoke` | TimesFM extra model column fusion smoke test | `ACTIVE` |
| **H: P3/Paper Decision Report** | — | Document P3 outcome, lessons, paper viability | `PLANNED` |
| **SA1 (Leakage Fix)** | `tune-timemixer` | FIX-01–FIX-04 + schema.py | `✅ MERGED via PR #10` |
| **P3.1 (Severe-Aware)** | `agent/p31-severe-aware-rolling` | 3 severe-aware modes | `NO-GO — PR #11 MERGED` |
| **P3.2 (Rolling+Correction)** | `tune-timemixer` | rolling + correction combo | `✅ MERGED via PR #10 (NO-GO)` |

---

## Status Overview

| Component | Status | Notes |
|-----------|--------|-------|
| P3a: Risk leakage fixes | Done | FIX-01–FIX-04 applied, schema.py created, all tests pass |
| P3b: Rolling 30D fusion script | Done | `run_rolling_30d_fusion.py` — 4 weight modes, CLI aliases, leakage-safe |
| P3b: Rolling 30D actual run | DONE | Softmax: sMAPE 19.86 (GO), severe 82 (NO-GO). All 3 modes: NO-GO |
| P3c: SOTA lab scaffolding | Done | `docs/P3_single_model_sota_lab.md` — experiment plan + feasibility matrix |
| P3c: SOTA actual experiments | DEFERRED | LightGBM tuning deferred — rolling fusion did not meet GO thresholds |
| P3d: Unified evaluation script | Done | `evaluate_p3_rolling_sota_summary.py` — GO/CONDITIONAL/NO-GO |
| P3d: Actual evaluation run | DONE | VERDICT: NO-GO — severe underestimate threshold not met |
| P3.1: Severe-aware rolling fusion | DONE | 3 new modes: severe_softmax, severe_anchor, quantile_guarded. All NO-GO |
| P3.2: Rolling base + spike correction | DONE | Combined rolling severe_softmax + Phase2 correction. VERDICT: NO-GO |
| P3.3 Sprint A: Correction grid | DONE | 192 combos searched, 0 meet GO. NO-GO |
| P3.3 Sprint B: Spike-gated uplift | DONE | sMAPE=22.68, severe=76. NO-GO |
| P3.3 Sprint C: Extra prediction signal | DONE | TimesFM diversity smoke. NO-GO |
| P3.3 Sprint D: LightGBM weighting | DONE | sMAPE=23.76, severe=54. NO-GO |
| P3.4 Line E: Weighted LGBM + Phase2 fusion | DONE | sMAPE=23.63, severe=116. Weighted LGBM degrades on full window. **NO-GO** |
| P3.4 Line G: TimesFM diversity smoke | DONE | No diversity benefit from TimesFM as extra model column |

---

## P3a — Leakage Fixes

| Fix | File | Status | Test |
|-----|------|--------|------|
| FIX-01 | `train_realtime_spike_risk.py` — exclude ACTUAL_COLS via ALL_EXCLUDED_COLS | Done | test_actual_cols_excluded |
| FIX-02 | `predict_realtime_spike_risk.py` — remove y_true placeholder, leakage-safe fallback | Done | test_predict_no_y_true |
| FIX-03 | `build_realtime_spike_dataset.py` — whitelist after merge | Done | test_whitelist_drops |
| FIX-04 | `evaluate_realtime_spike_correction.py` — timestamp-level dedup | Done | (manual) |

## P3b — Rolling 30D Fusion

| Config | Value |
|--------|-------|
| Script | `scripts/run_rolling_30d_fusion.py` |
| Weight modes | convex, ridge, softmax, anchor |
| Default lookback | 30 days |
| Min history | 10 days |
| CLI aliases | `--weight-mode` / `--fusion-mode`, `--lookback-days` / `--train-window-days` |
| Status | Script complete — ACTUAL RUN COMPLETE — see `docs/reports/P3_rolling_30d_fusion_report.md` |

### P3b Rolling Fusion Results

| Mode | sMAPE | Severe | Verdict |
|------|-------|--------|---------|
| anchor_90 | 20.61 | 82 | sMAPE ✅ (≤20.86), Severe ❌ (>63) |
| softmax | **19.86** | 83 | sMAPE ✅, Severe ❌ |
| convex | 23.70 | 152 | Both ❌ |

## P3c — Single-Model SOTA Lab

| Model | Priority | Plan |
|-------|----------|------|
| LightGBM | High | Hyperparameter grid search + CV |
| RT916 | Low | Feasibility diagnosis only |
| SGDFNet | Low | Feasibility diagnosis only |
| TimeMixer | Low | Feasibility diagnosis only |

## P3d — Unified Evaluation

| Script | Status |
|--------|--------|
| `scripts/evaluate_p3_rolling_sota_summary.py` | Done — GO/CONDITIONAL/NO-GO rules |
| **Actual evaluation** | **DONE — verdict: NO-GO** |

### P3d Evaluation Verdict

| Criterion | Threshold | Best P3 (softmax) | Met? |
|-----------|-----------|-------------------|------|
| sMAPE ≤ 20.86 | 20.86 | 19.86 | ✅ |
| Severe underestimates ≤ 63 | 63 | 83 | ❌ |
| **Verdict** | | | **NO-GO** |

## P3.1 — Severe-Aware Rolling Fusion

| Weight mode | Description | sMAPE | Severe | Verdict |
|-------------|-------------|:-----:|:------:|:-------:|
| severe_softmax | sMAPE + α·severe_rate + β·underprediction_MAE | 21.00 | 88 | NO-GO |
| severe_anchor | LightGBM ≥ 0.85, filter baselines by severe rate | 21.12 | 86 | NO-GO |
| quantile_guarded | severe_softmax + p75 guard on high-risk hours | 21.14 | 62 | NO-GO (sMAPE) |

Note: sMAPE re-evaluated with standard pipeline (overall, not per-timestamp avg).

## P3.2 — Rolling Base + Spike Correction

Combines P3.1 rolling severe_softmax base with Phase2 correction pipeline.

| Profile | sMAPE | Severe | False Lift | Normal Degrad. | Verdict |
|---------|:-----:|:------:|:----------:|:--------------:|:-------:|
| Medium | 20.74 | 73 | 0.076 | -0.16 | NO-GO |
| Conservative | 20.99 | 86 | 0.004 | 0.00 | NO-GO |
| Aggressive | 23.84 | 64 | 0.749 | 3.30 | NO-GO |

**Key finding**: Correction reduces severe from 88→73 (medium) and improves sMAPE to 20.74 (best-ever), but no profile meets both thresholds.

**Verdict: NO-GO** — Phase 2 best (static anchor_90 + medium correction) remains best known candidate.

---

## P3.3 Parallel Sprint

> **Goal**: Find ANY configuration that simultaneously beats sMAPE ≤ 20.50 AND severe ≤ 63.
> **Baseline**: Phase 2 champion — sMAPE=20.86, severe=63.

### Unified Evaluation Criteria

| Tier | sMAPE | Severe | False Lift | Normal Degrad. | Action |
|------|-------|--------|------------|----------------|--------|
| **DEPLOY GO** | ≤ 20.50 | ≤ 63 | ≤ 10% | ≤ 0.5 | Merge as deployment candidate |
| **RESEARCH GO** | ≤ 20.00 | ≤ 70 | ≤ 12% | ≤ 1.0 | Promising — iterate |
| **NO-GO** | > 20.86 | > 70 | > 15% | > 1.0 | Stop — do not merge |

### Sprint A: Severe-Constrained Correction (COMPLETE)

| Field | Value |
|-------|-------|
| Branch | `agent/p33-severe-constrained-correction` |
| Approach | Grid search over 192 correction param combos (5 profile families) |
| Best result | severe=69, sMAPE=20.92, false_lift=7.6% (spike_prob_threshold=0.60, boost=1.50) |
| Status | `COMPLETE — NO-GO` |

**Result**: 192 combos searched, 53 eligible, **0 meet GO**. Root cause: risk model calibration. Minimum achievable severe is 69.

### Sprint B: Spike-Gated Uplift (COMPLETE)

| Field | Value |
|-------|-------|
| Branch | `agent/p33-spike-gated-uplift` |
| Result | sMAPE=22.68, severe=76 — **NO-GO**. Research architecture only. PR #13. |

### Sprint C: Extra Prediction Signal / TimesFM (COMPLETE)

| Field | Value |
|-------|-------|
| Branch | `agent/p33-extra-prediction-signal` |
| Result | See `docs/reports/P33_extra_prediction_signal_report.md`. |

### Sprint D: LightGBM Internal Weighting (COMPLETE)

| Field | Value |
|-------|-------|
| Branch | `agent/p33-lgbm-internal-weighting` |
| Result | sMAPE=23.76, severe=54 — **NO-GO** (sMAPE regression). PR #12. |

**P3.3 VERDICT**: All four lines NO-GO. No configuration beats Phase 2 on sMAPE AND severe simultaneously.

---

## P3.4 — Candidate Combination Sprint

> **Goal**: Converge. Combine best P3 signals into single system beating Phase 2 champion.
> **Baseline**: sMAPE=20.86, severe=63.
> **Threshold**: DEPLOY GO (sMAPE ≤ 20.50, severe ≤ 63, false lift ≤ 10%, normal degrad ≤ 0.5).

### Line E: Weighted-LightGBM + Phase2 Fusion/Correction

| Field | Value |
|-------|-------|
| Branch | `agent/p34-weighted-lgbm-phase2-fusion` |
| Approach | Weighted-LightGBM (Line D) output → Phase2 fusion + correction pipeline |
| Rationale | D achieved severe=54 (best-ever) but sMAPE=23.76. Fix sMAPE by fusing with strong baselines. |
| Fusion modes | weighted_lgbm_anchor_90, weighted_lgbm_anchor_80, custom |
| Correction profiles | medium, conservative, aggressive (all normal mode) |
| Status | `EVALUATED — NO-GO` |
| Best result | sMAPE=23.63, severe=116 — both worse than Phase2 champion |
| Finding | Weighted LGBM degrades on full window (severe 146 vs standard LGBM 80). Weighting does not generalize beyond the 15-day p33 validation window. Correction cannot recover the loss. |

### Line F: PR Cleanup + Research Tooling Merge

| Field | Value |
|-------|-------|
| Scope | PR #11 (P3.1), PR #12 (LGBM weighting), PR #13 (spike-gated uplift) |
| Status | PR #11: Merged. PR #12/#13: Mergeable, need manual merge via web UI. |

### Line G: TimesFM Diversity Fusion Smoke Test

| Field | Value |
|-------|-------|
| Branch | `agent/p34-timesfm-diversity-smoke` |
| Result | TimesFM predictions do not improve multi-candidate fusion. No diversity benefit. |
| Status | `COMPLETE — NO-GO` |

### Line H: P3/Paper Decision Report

| Field | Value |
|-------|-------|
| Deliverable | Document P3 outcome analysis, root causes, lessons learned, paper viability assessment |
| Status | `PLANNED` |

---

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | INACTIVE — all rolling approaches produce severe >63 |
| B21 | SOTA model experiments | DEFERRED — LightGBM tuning scoped but not started |
| B22 | P3.2 rolling + correction NO-GO | CLOSED — best at severe=73, above threshold |
| B23 | P3.1 sMAPE re-evaluation | CLOSED — P3.1 rolling base sMAPE is 21.00 (not 19.10) |
| B24 | No approach beats Phase 2 simultaneously on sMAPE + severe | **CLOSED — P3.3 all lines NO-GO. P3.4 opened** |

## Next Actions

1. ✅ PR #10 merged — tooling/leakage-fix/experiment framework in `tune-timemixer`
2. ✅ PR #11 merged — P3.1 severe-aware rolling (NO-GO, research tooling)
3. ✅ P3.2 rolling + correction evaluation complete (NO-GO)
4. ✅ P3.3 all four lines complete (all NO-GO)
5. ✅ **P3.4 Line E**: Complete — sMAPE=23.63, severe=116. **NO-GO**
6. 📋 **P3.4 Line F**: Clean up and merge PR #12, PR #13 (manual web UI merge needed)
7. 🏃 **P3.4 Line H**: Draft P3/Paper decision report — Phase 2 champion is final production candidate
8. 🎯 **Decision gate**: P3.4 Line E NO-GO confirmed. All P3 paths exhausted. **Deploy Phase 2 champion.**
