# P0 Realtime High Spike — Execution Board

> **Purpose**: Track all tasks across the P0 full execution pipeline.
> **Status**: `[Phase 0.7 — PR #5 Conflict Resolution]`
> **Runner branch**: `agent/p0-threshold-tuning`

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `agent/p0-full-execution-runner-clean` | Orchestrate P0 full run | `READY` |
| **SA1 (Contract Fix)** | `origin/agent/p0-residual-lift-guardrail` | Contract unification | `✅ ALREADY ON tune-timemixer` |
| **SA2 (Path Compat)** | `origin/agent/p0-path-compat` | P0 scripts + path compat | `✅ MERGED to tune-timemixer` |
| **SA3 (Threshold Tuning)** | `origin/agent/p0-threshold-tuning` | Correction profile tuning | `🟡 CONFLICT RESOLVED — pending push` |
| **SA4 (Final Report)** | `TBD` | Consolidation report | `PENDING` |

---

## Status Overview

| Component | Status | Agent | Notes |
|---|---|---|---|
| SA1 — Contract Fix | ✅ Done | SA1 | Field contract unified, business_day/hour_business fix |
| SA2 — Path Compatibility | ✅ Done | SA2 | Unified CLI flags; path resolution from daily_runs/ with outputs/ fallback |
| SA3 — Residual Lift + Guardrail | ✅ Done | SA3 | Core correction pipeline built |
| SA3 — Threshold Tuning | ✅ Done | SA3 | Profiles + evaluation scripts complete |
| P0 Full Run (Offline Eval) | 🟡 Ready for Runner | — | Waiting for prediction pack and Runner launch |
| Extreme Diagnostics | ✅ Scripts ready | SA2 | diagnose_extreme_events.py, diagnose_model_regime.py |
| Spike Correction Pipeline | 🟡 Scripts ready | SA2/SA3 | build/train/predict/evaluate scripts |
| Window Coverage | ✅ Path manifest | SA2 | daily_runs/ + outputs/ fallback |

---

## Phase 0.5 — SA1 Integration

### SA1 Contract Fixes Verified

| Fix | Description | Status | Notes |
|-----|-------------|--------|-------|
| FIX-01 | sMAPE_floor50 canonical formula | ✅ PASS | Verified via test |
| FIX-02 | PREDICTION_PACK_COLUMNS update | ✅ PASS | schema.py confirmed on SA3 |
| FIX-03 | hour → hour_business | ✅ PASS | P0 eval scripts use hour_business |
| FIX-04 | is_spike → label_high_spike | ✅ PASS | Correct column name in use |
| FIX-05 | business_day mapping for hour=24 | ✅ PASS | Verified in data_contract.py + 3 synthetic cases |
| FIX-06 | p0_shared_contract.md | ✅ EXISTS ON SA3 | Unified contract doc created |
| FIX-07 | p0_execution_board.md | ✅ CREATED | This board |

### Synthetic Time Mapping Test

```
Timestamp -> business_day, hour_business:
  2026-01-02 00:00:00 -> 2026-01-01, hb=24  ✓
  2026-01-01 13:00:00 -> 2026-01-01, hb=13  ✓
  2026-01-01 08:00:00 -> 2026-01-01, hb=8   ✓
  2026-01-01 23:00:00 -> 2026-01-01, hb=23  ✓
  2026-01-02 01:00:00 -> 2026-01-02, hb=1   ✓

Round-trip (all 24 hours):                      ✓
```

---

## Phase 0.6 — Merge Audit & Cleanup

### Audit Results

| Metric | Value |
|--------|-------|
| Changed files vs `origin/tune-timemixer` | 8 (1 modified .gitignore, 7 deletions) |
| Forbidden files modified | 0 |
| Data artifacts added | 0 |
| SA1 merge pollution | 0 (already on tune-timemixer) |

### Cleanup Actions

- `.gitignore` refined: `reports/` → `reports/local/`, added `outputs/local/`, etc.
- Runner branch rebuilt: `agent/p0-full-execution-runner-clean` from `origin/tune-timemixer`
- Polluted runner backed up: `backup/p0-runner-polluted-20260629-210055`

---

## Phase 0.7 — PR #5 Conflict Resolution

- **Latest base**: `origin/tune-timemixer` (`2656ea7`)
- **PR #4 path compatibility**: ✅ Already merged
- **PR #6 main**: ✅ Already merged
- **PR #5 threshold tuning**: 🔄 Conflict resolved — pending push

### Files Conflicted & Resolved

| File | Type | Resolution |
|------|------|------------|
| `docs/p0_execution_board.md` | add/add | Merged: kept SA3 profile tables + SA2 runner status + Phase 0.5/0.6 audit records + Phase 0.7 section |
| `docs/p0_shared_contract.md` | add/add | Merged: unified header + path contract (SA2) + profile/metrics contract (SA3) |
| `scripts/evaluate_realtime_spike_correction.py` | add/add | Merged: SA3 correction pipeline base + SA2 unified CLI flags |
| `scripts/evaluate_p0_realtime_spike_full.py` | add/add | Merged: SA3 profile-based correction + SA2 orchestrator mode |

### Tests Run

- py_compile: ✅ All modified scripts pass
- pytest (guardrail): ✅ Passed

### Remaining Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B12 | SA2 ↔ SA3 file conflicts | ✅ RESOLVED in Phase 0.7 |

---

## Target Windows

| Window | start-date | end-date | Priority |
|--------|-----------|---------|----------|
| Nov–Dec 2025 | `2025-11-01` | `2025-12-31` | P0 (worst perf) |
| Jan–Feb 2026 | `2026-01-01` | `2026-02-28` | P0 |

## Threshold Tuning Profiles

| Profile | spike_prob_threshold | max_lift_ratio | max_absolute_lift | period_9_16_boost |
|---|---|---|---|---|
| conservative | 0.75 | 0.20 | 200 | 1.0 |
| medium | 0.60 | 0.35 | 350 | 1.15 |
| aggressive | 0.45 | 0.60 | 600 | 1.30 |

## Phase 1-4 Execution — PENDING (Wait for PR #5 mergeable)

| Phase | Script | Description | Status |
|-------|--------|-------------|--------|
| 1a | `build_realtime_spike_dataset.py` | Build spike labels + features | `PENDING` |
| 1b | — | Validate label distribution | `PENDING` |
| 2a | `train_realtime_spike_risk.py` | Train risk model | `PENDING` |
| 2b | — | Validate AUC/recall/calibration | `PENDING` |
| 3a | `evaluate_realtime_spike_correction.py` | Apply + evaluate correction | `PENDING` |
| 3b | — | Conservative/medium/aggressive pass | `PENDING` |
| 4a | `evaluate_p0_realtime_spike_full.py` | Unified evaluation | `PENDING` |
| 4b | SA4 report | GO/NO-GO decision | `PENDING` |
