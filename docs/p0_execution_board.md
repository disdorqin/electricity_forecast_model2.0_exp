# P0 Realtime High Spike — Execution Board

> **Purpose**: Track all tasks across the P0 full execution pipeline.
> **Purpose**: Track all tasks across the P0 full execution pipeline.
> **Status**: `[Phase 1 — BLOCKED: No model predictions for P0 window]`
> **Runner branch**: `agent/p0-full-execution-runner-clean`

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
| **B13** | **No model predictions for P0 window** | **OPEN** | No `daily_runs/` directory exists. Prediction pack has 0 rows, 120 missing dates. See Phase 1 Failure. |

---

---

## Prediction Source Inventory

| Field | Value |
|-------|-------|
| **Script** | `scripts/inventory_prediction_sources.py` |
| **Run by** | SA2 (Path Compatibility) |
| **Date** | 2026-06-29 |
| **Output** | `reports/local/p0_full_run/inventory/prediction_source_inventory.md` |
| **Output** | `reports/local/p0_full_run/inventory/prediction_source_inventory.json` |

### Key Findings

| Question | Answer |
|----------|--------|
| Which model is easiest for first prediction pack? | **LightGBM** — has checkpoint `models/LightGBM/best_model_实时电价.pkl`, CPU inference, ~5 min for 4 months |
| Which are inference-only? | **LightGBM** (has checkpoint), **TimesFM** (pre-trained, needs GPU) |
| Which must train? | **TimeMixer** (no checkpoint, no inference script), **SGDFNet** (no checkpoint), **RT916** (no checkpoint for P0 window) |
| RT916 checkpoint status? | Only outputs for May–Jun 2026 (outside P0 window Nov–Feb) |
| LightGBM baseline feasible? | **Yes** — has `infer_fix.py`, checkpoint exists, fastest path to first prediction pack |
| Naive pack approach viable? | **Yes** — build from raw xlsx features + LightGBM predictions; fusion can produce `base_fused_pred` from single-model pack |

### Model Readiness Summary

| Model | Checkpoint | Inference | GPU | Est. Time | P0 Ready? |
|-------|-----------|-----------|-----|-----------|-----------|
| lightgbm | ✅ `best_model_实时电价.pkl` | ✅ `infer_fix.py` | ❌ | ~5 min | **✅ YES** |
| timesfm | ❌ (pre-trained, needs legacy TF) | ✅ `infer.py` | ✅ | ~30 min | ❌ |
| timemixer | ❌ (no .pt found) | ❌ | ✅ | Unknown | ❌ |
| sgdfnet | ❌ (no .pth/.ckpt) | ✅ `pipeline.py` | ✅ | ~15 min | ❌ |
| rt916 | ❌ (only May–Jun 2026) | ✅ `model.py` | ✅ | ~2 hr | ❌ |
| fusion | ❌ (needs predictions first) | ❌ | ❌ | N/A | ❌ |

### Recommended Path

1. Run **LightGBM inference** (`lightGBM/infer_fix.py`) for Nov 2025 – Feb 2026
2. Build prediction pack from LightGBM output
3. Use fusion to produce `base_fused_pred` (single-model fused = LightGBM predictions)
4. Proceed with spike correction evaluation

---

## Phase 1 Failure — Prediction Pack Build

| Field | Value |
|-------|-------|
| **Command** | `python scripts/build_backtest_prediction_pack.py --data-path data/shandong_pmos_hourly.xlsx --runs-root daily_runs --runs-root-fallback outputs --target realtime --start-date 2025-11-01 --end-date 2026-02-28 --models all` |
| **Error** | Prediction pack is empty (0 rows). 120 missing dates (all dates in Nov 2025 – Feb 2026). |
| **Root Cause** | `daily_runs/` directory does not exist in project root. Available prediction CSVs (RT916, SGDFNet, TimesFM) only cover May–Jun 2026 (`oof_runs/`), not the P0 window (Nov 2025 – Feb 2026). Raw xlsx data exists at `data/shandong_pmos_hourly.xlsx` (2857 rows in P0 window) but contains only features and actual prices — no model predictions. |
| **Files Involved** | `scripts/build_backtest_prediction_pack.py`, `data/shandong_pmos_hourly.xlsx` |
| **Data Available** | Raw xlsx: 2857 rows in P0 window (Nov–Feb), realtime price + features. Model predictions: NONE for P0 window. |
| **Can continue?** | **NO** — see blocker B13. Cannot proceed to Phase 2-4 without model predictions for evaluation. |
| **Need support agent?** | **SA2/Path** (if daily_runs exists elsewhere), or **Other** (guidance on prediction source) |

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

## Phase 1-4 Execution — BLOCKED (No predictions for P0 window)

| Phase | Script | Description | Status |
|-------|--------|-------------|--------|
| 1a | `build_backtest_prediction_pack.py` | Build spike labels + features | ❌ **FAILED** — empty pack, 0 rows |
| 1b | — | Validate label distribution | `SKIPPED` |
| 2a | `diagnose_extreme_events.py` | Extreme event diagnostics | `SKIPPED` |
| 2b | `diagnose_model_regime.py` | Model regime analysis | `SKIPPED` |
| 3a | `build_realtime_spike_dataset.py` | Build spike training dataset | `SKIPPED` |
| 3b | `train_realtime_spike_risk.py` | Train risk model | `SKIPPED` |
| 4a | `evaluate_realtime_spike_correction.py` | Three-profile correction | `SKIPPED` |
| 4b | `evaluate_p0_realtime_spike_full.py` | Full evaluation / GO-NOGO | `SKIPPED` |
