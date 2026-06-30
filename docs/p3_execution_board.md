# P3 Rolling Fusion + SOTA Lab — Execution Board

> **Purpose**: Track all tasks across the P3 execution pipeline.
> **Status**: `[P3a–P3.2 COMPLETE (all NO-GO) — P3.3 Sprint A COMPLETE (NO-GO)]`
> **Branch**: `tune-timemixer`
> **Deployment champion**: Phase 2 lightgbm_anchor_90 + medium + normal (sMAPE=20.86, severe=63)
> **PR #10**: ✅ MERGED — tooling/leakage-fix/experiment framework
> **PR #11**: OPEN — P3.1 severe-aware rolling (NO-GO, but mergeable as experiment tooling)

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `tune-timemixer` | P3.3 parallel sprint coordinator | `ACTIVE` |
| **A: Severe-Constrained Correction** | `agent/p33-severe-constrained-correction` | Correction → severe penalty optimization | `NO-GO — risk model bottleneck` |
| **B: Spike-Gated Uplift** | `agent/p33-spike-gated-uplift` | Separate uplift model for spike hours | `PLANNED` |
| **C: Extra Prediction Signal** | `agent/p33-extra-prediction-signal` | RT916 / TimesFM integration | `PLANNED` |
| **D: LightGBM Internal Weighting** | `agent/p33-lgbm-internal-weighting` | Sample-weighted LightGBM retrain | `PLANNED` |
| **SA1 (Leakage Fix)** | `tune-timemixer` | FIX-01–FIX-04 + schema.py | `✅ MERGED via PR #10` |
| **P3.1 (Severe-Aware)** | `agent/p31-severe-aware-rolling` | 3 severe-aware modes | `NO-GO — PR #11 pending` |
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

---

## Files Changed

| File | Change |
|------|--------|
| `extreme/realtime_high_spike/schema.py` | NEW — leakage-safe column schema |
| `scripts/train_realtime_spike_risk.py` | FIX-01 — import ALL_EXCLUDED_COLS |
| `scripts/predict_realtime_spike_risk.py` | FIX-02 — y_true-free heuristic + model loading |
| `scripts/build_realtime_spike_dataset.py` | FIX-03 — whitelist drop ACTUAL_VALUE_EXCLUDE_COLS |
| `scripts/evaluate_realtime_spike_correction.py` | FIX-04 — timestamp-level dedup arg |
| `scripts/run_rolling_30d_fusion.py` | NEW — rolling weight fusion with 4 modes |
| `scripts/evaluate_p3_rolling_sota_summary.py` | NEW — unified comparison + verdict |
| `tests/test_realtime_spike_no_leakage.py` | NEW — 5 leakage safety tests |
| `docs/P3_single_model_sota_lab.md` | NEW — SOTA experiment plan |
| `docs/p3_execution_board.md` | NEW — this board |
| `scripts/evaluate_p32_rolling_base_correction.py` | NEW — P3.2 rolling base + correction eval |
| `tests/test_p32_rolling_base_correction.py` | NEW — 8 P3.2 tests |
| `docs/reports/P32_rolling_base_correction_report.md` | NEW — P3.2 findings |

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

**Key finding**: Correction reduces severe from 88→73 (medium) and improves sMAPE to 20.74 (best-ever), but no profile achieves both sMAPE ≤ 19.50 AND severe ≤ 63. Rolling base re-evaluated at sMAPE=21.00 (not 19.10 as originally reported — per-timestamp avg was misleading).

**Verdict: NO-GO** — P3.2 medium is the best combined result (sMAPE=20.74, severe=73) but does not meet either primary GO threshold.

Phase 2 best (static anchor_90 + medium correction) remains the best known candidate.

---

## P3.3 Parallel Sprint

> **Goal**: Find ANY configuration that simultaneously beats sMAPE ≤ 20.50 AND severe ≤ 63.
> **Baseline**: Phase 2 champion — sMAPE=20.86, severe=63.
> **Evaluation**: All windows must report using the unified criteria below.

### Unified Evaluation Criteria

| Tier | sMAPE | Severe | False Lift | Normal Degrad. | Action |
|------|-------|--------|------------|----------------|--------|
| **DEPLOY GO** | ≤ 20.50 | ≤ 63 | ≤ 10% | ≤ 0.5 | Merge as deployment candidate |
| **RESEARCH GO** | ≤ 20.00 | ≤ 70 | ≤ 12% | ≤ 1.0 | Promising — iterate |
| **NO-GO** | > 20.86 | > 70 | > 15% | > 1.0 | Stop — do not merge |

Note: sMAPE beats Phase 2 alone is insufficient. Both sMAPE AND severe must beat Phase 2 simultaneously.

### Sprint Line A: Severe-Constrained Correction Search

| Field | Value |
|-------|-------|
| Branch | `agent/p33-severe-constrained-correction` |
| Approach | Grid search over 192 correction param combos (5 profile families) |
| Goal | severe ≤ 63 with sMAPE ≤ 20.50 |
| Best result | severe=69, sMAPE=20.92, false_lift=7.6% (spike_prob_threshold=0.60, boost=1.50) |
| Status | `COMPLETE — NO-GO` |

**Results**: 192 combos searched, 53 eligible (false_lift ≤ 12%, normal_degrad ≤ 0.5), **0 meet DEPLOY GO or RESEARCH GO**. Minimum achievable severe is 69 at threshold=0.60. Root cause: risk model calibration — 77% of non-spike hours have high_spike_prob ≥ 0.5, forcing threshold ≥ 0.60 for acceptable false_lift, which captures only 51.4% of true spikes.

**Recommendation**: De-prioritize correction tuning. Upstream improvements needed (P3.3 Line B/D). See `docs/reports/P33_severe_constrained_correction_report.md`.

### Sprint Line B: Spike-Gated Uplift Model

| Field | Value |
|-------|-------|
| Branch | `agent/p33-spike-gated-uplift` |
| Approach | Train separate uplift model for high-spike-risk hours; gate by spike probability |
| Goal | Apply extra lift only where spike probability > threshold, reducing false lift on normal hours |
| Status | `PLANNED` |

### Sprint Line C: Extra Prediction Signal (RT916 / TimesFM)

| Field | Value |
|-------|-------|
| Branch | `agent/p33-extra-prediction-signal` |
| Approach | Integrate RT916 or TimesFM predictions as additional model columns in multi-candidate pack |
| Goal | Add signal diversity to reduce systematic underprediction on spike hours |
| Status | `PLANNED` |

### Sprint Line D: LightGBM Internal Sample Weighting

| Field | Value |
|-------|-------|
| Branch | `agent/p33-lgbm-internal-weighting` |
| Approach | Retrain LightGBM with sample weights that penalize spike-hour underestimates |
| Goal | Improve base prediction quality on spike hours → less correction needed downstream |
| Status | `PLANNED` |

---

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | INACTIVE — all rolling approaches produce severe >63 |
| B21 | SOTA model experiments | DEFERRED — LightGBM tuning scoped but not started |
| B22 | P3.2 rolling + correction NO-GO | CLOSED — best at severe=73, above threshold |
| B23 | P3.1 sMAPE re-evaluation | CLOSED — P3.1 rolling base sMAPE is 21.00 (not 19.10) |
| B24 | **No approach beats Phase 2 simultaneously on sMAPE + severe** | **OPEN — P3.3 sprint objective** |

## Next Actions

1. ✅ PR #10 merged — tooling/leakage-fix/experiment framework in `tune-timemixer`
2. ⏳ PR #11 — P3.1 severe-aware rolling (4 files only). Merge as tooling if clean, or close as NO-GO
3. ✅ P3.2 rolling + correction evaluation complete (NO-GO)
4. 🏃 **P3.3 Sprint A**: Run severe-constrained correction search
5. 📋 **P3.3 Sprint B**: Scaffold spike-gated uplift model branch
6. 📋 **P3.3 Sprint C**: Assess RT916/TimesFM checkpoint availability
7. 📋 **P3.3 Sprint D**: Scaffold LightGBM internal weighting branch
8. 🎯 **Decision gate**: If no P3.3 line produces DEPLOY GO, deploy Phase 2 champion as production candidate
