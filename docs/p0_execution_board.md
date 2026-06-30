# P0 Realtime High Spike — Execution Board

> **Purpose**: Track all tasks across the P0 full execution pipeline.
> **Status**: `[Phase 1B — COMPLETE: Full pipeline end-to-end via LightGBM bootstrap]`
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
| **B13** | **No model predictions for P0 window** | **RESOLVED** | LightGBM inference + baseline pack bypasses `daily_runs/` dependency. See Phase 1B. |

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

## Phase 1B — Long-Run Bootstrap (Completed 2026-06-29)

> **Result**: Full pipeline executed end-to-end using Level 1 (LightGBM) predictions.
> **Status**: `✅ COMPLETE — No further action needed for Level 1`

### Loop 1: Prediction Source Inventory

| Script | Status | Output |
|--------|--------|--------|
| `scripts/inventory_prediction_sources.py` | ✅ Done | `reports/local/p0_full_run/inventory/` |

**Readiness**: LightGBM (✅ checkpoint, ~5 min CPU), TimesFM (❌ legacy TF), TimeMixer (❌ no checkpoint), SGDFNet (❌ no checkpoint), RT916 (❌ no P0 checkpoint), Fusion (❌ needs predictions)

### Loop 2: Level 0 Baseline Pack

| Script | Status | Output |
|--------|--------|--------|
| `scripts/build_baseline_prediction_pack.py` | ✅ Done | `reports/local/p0_full_run/prediction_pack_level0/` |

**Models**: naive_lag1, naive_lag7, dayahead_proxy, baseline_fusion
**Coverage**: 4 models × 120 dates (full P0 window)

### Loop 3: Level 1 LightGBM Predictions

| Script | Status | Output |
|--------|--------|--------|
| `scripts/run_p0_lightweight_predictions.py` | ✅ Done | `reports/local/p0_full_run/prediction_pack_level1/` |

**Model**: LightGBM ThreeStageLGBM (valley/solar/peak sub-models)
**Coverage**: 2,862 rows, 120/120 dates, 24h/day
**Avg sMAPE**: 17.76

### Loop 4: RT916 Selective Plan

| Plan | Status | Output |
|------|--------|--------|
| `scripts/rt916_selective_plan.md` | ✅ Created | `reports/local/p0_full_run/rt916_selective_plan.md` |

**Top 3 dates**: 2025-11-08 (1 spike, 1408), 2026-01-26 (4 spikes, 1291), 2026-01-18 (5 spikes, 1187)
**Recommendation**: DEFER until Level 1 evaluation reviewed (no available checkpoint)

### Loop 5: Full P0 Evaluation (Level 1)

| Phase | Script | Status | Key Metrics |
|-------|--------|--------|-------------|
| 2a | `diagnose_extreme_events.py` | ✅ Done | 144 high_spike, 523 total extreme events |
| 3a | `build_realtime_spike_dataset.py` | ✅ Done | 39,168 rows, 956 spike samples |
| 3b | `train_realtime_spike_risk.py` | ✅ Done | AUC 0.929, recall 0.83, RF 200 trees |
| 4a | `evaluate_realtime_spike_correction.py` | ✅ Done | All 3 profiles: 0 lift_applied |
| 4b | `evaluate_p0_realtime_spike_full.py` | ✅ Done | Full manifest written |

### Correction Results (all 3 profiles identical)

| Metric | Value |
|--------|-------|
| Overall sMAPE (floor50) | 22.02 |
| 9_16 sMAPE (floor50) | 28.16 |
| High Spike MAE | 260.56 |
| High Spike sMAPE | 46.95 |
| Severe Underestimates | 80 |
| Normal Hours sMAPE | 18.91 |
| Normal Hours Degradation | 0.0 |
| False Lift Rate | 0.0 |
| Lift Applied | **0** |
| Lift Rejected (low prob) | 2,463 |
| Lift Rejected (negative base) | 399 |

### Key Findings

1. **Pipeline works end-to-end**: All 5 phases executed successfully from raw data to correction manifests
2. **LightGBM baseline reasonable**: sMAPE 22 overall, but severe_underestimates=80 indicates significant spikes missed
3. **Correction not activating**: All 3 profiles produce 0 lift_applied — risk model flags too few rows at the configured thresholds, and among those flagged, guardrails reject due to negative base residual
4. **Bottleneck**: Single-model pack means `base_fused_pred` = LightGBM prediction; correction lifts only apply when `base_fused_pred > y_pred` (i.e., the fused prediction already exceeds the raw prediction), which never happens in a single-model pack
5. **RT916 still deferred**: No checkpoint available; would need full training run (~3-4 hours per model)

### Nested Output Path Issue

The correction evaluator appends profile name to `--out-dir`, producing double-nested paths:
`level0/correction/conservative/conservative/`. Output files are correct — cosmetic only.

### Blocker B13 Resolution

| Blocker | Status | Resolution |
|---------|--------|------------|
| **B13** — No model predictions for P0 window | ✅ **RESOLVED** | LightGBM inference + baseline pack approach bypasses need for `daily_runs/` |

### Full Report

See [docs/reports/P0_long_run_status.md](reports/P0_long_run_status.md) for detailed metrics, GO/CONDITIONAL/NO-GO assessment, and next steps.
