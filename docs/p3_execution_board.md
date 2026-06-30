# P3 Rolling Fusion + SOTA Lab — Execution Board

> **Purpose**: Track all tasks across the P3 execution pipeline.
> **Status**: `[P3a COMPLETE — P3b RUN COMPLETE (NO-GO) — P3c SCOPED — P3d EVALUATED (NO-GO)]`
> **Branch**: `agent/p3-rolling-fusion-sota`
> **Verdict**: PR #10 = tooling/leakage-fix/experiment framework merge, NOT a new deployment candidate.
>   Current deployment candidate remains Phase 2: `lightgbm_anchor_90` + `medium` + `normal`.

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `agent/p3-rolling-fusion-sota` | P3 end-to-end | `ACTIVE` |
| **SA1 (Leakage Fix)** | `agent/p3-rolling-fusion-sota` | FIX-01–FIX-04 + schema.py | `DONE` |
| **SA3 (Rolling Fusion)** | `agent/p3-rolling-fusion-sota` | run_rolling_30d_fusion.py | `SCRIPT DONE` |
| **SA4 (Evaluation)** | `agent/p3-rolling-fusion-sota` | evaluate_p3_rolling_sota_summary.py | `SCRIPT DONE` |

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

Phase 2 best (static anchor_90 + medium correction) remains the best known candidate.

---

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | OPEN — all modes produce severe >63; optimizer needs severe penalty |
| B21 | SOTA model experiments | DEFERRED — LightGBM tuning scoped but not started |

## Next Actions

1. ✅ Run P3b rolling fusion with Phase 2 prediction pack
2. ✅ Run P3d unified evaluation
3. ❌ Rolling fusion NO-GO — severe underestimates exceed threshold
4. **Diagnose**: Add severe underestimate penalty to weight optimizer
5. **Alternative**: Increase anchor weight to 0.95 or deploy Phase 2 as-is
6. SOTA LightGBM experiments (deferred to next cycle)
