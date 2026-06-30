# P3 Rolling Fusion + SOTA Lab — Execution Board

> **Purpose**: Track all tasks across the P3 execution pipeline.
> **Status**: `[P3a COMPLETE — P3b RUN COMPLETE (NO-GO) — P3c SCOPED — P3d EVALUATED (NO-GO)]`
> **Branch**: `agent/p3-rolling-fusion-sota`

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `agent/p3-rolling-fusion-sota` | P3 end-to-end | `ACTIVE` |
| **SA1 (Leakage Fix)** | `agent/p3-rolling-fusion-sota` | FIX-01–FIX-04 + schema.py | `DONE` |
| **SA3 (Rolling Fusion)** | `agent/p3-rolling-fusion-sota` | run_rolling_30d_fusion.py | `SCRIPT DONE` |
| **SA4 (Evaluation)** | `agent/p3-rolling-fusion-sota` | evaluate_p3_rolling_sota_summary.py | `SCRIPT DONE` |
| **P3.1 (Severe-Aware Rolling)** | `agent/p31-severe-aware-rolling` | 3 severe-aware weight modes | `DONE (NO-GO)` |
| **P3.2 (Rolling Base + Correction)** | `agent/p3-rolling-fusion-sota` | rolling + Phase2 correction combo | `DONE (NO-GO)` |

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

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | OPEN — all modes produce severe >63; optimizer needs severe penalty |
| B21 | SOTA model experiments | DEFERRED — LightGBM tuning scoped but not started |
| B22 | P3.2 rolling + correction NO-GO | OPEN — P3.2 medium best at severe=73, still 10 above threshold |
| B23 | P3.1 sMAPE re-evaluation | CLOSED — P3.1 rolling base sMAPE is 21.00 (not 19.10) |

## Next Actions

1. ✅ Run P3b rolling fusion with Phase 2 prediction pack
2. ✅ Run P3d unified evaluation
3. ❌ Rolling fusion NO-GO — severe underestimates exceed threshold
4. ✅ P3.1 severe-aware rolling — 3 new modes, all NO-GO
5. ✅ P3.2 rolling + correction — best combined result at sMAPE=20.74 / severe=73
6. ❌ **P3 combined verdict: NO-GO** — No rolling or correction approach achieves both sMAPE ≤ 19.50 and severe ≤ 63
7. **Recommendation**: Phase 2 lightgbm_anchor_90 + medium correction (sMAPE=20.86, severe=63) remains best known candidate
8. SOTA LightGBM experiments (deferred to next cycle)
