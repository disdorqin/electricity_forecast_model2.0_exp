# P0 Realtime High Spike — Execution Board

> **Purpose**: Track all tasks across the P0 full execution pipeline.
> **Status**: `[SA1 MERGED — Awaiting SA2/SA3]`
> **Runner branch**: `agent/p0-full-execution-runner`

---

## Roles

| Role | Branch | Scope | Status |
|------|--------|-------|--------|
| **Runner (orchestrator)** | `agent/p0-full-execution-runner` | Orchestrate P0 full run | `IN PROGRESS` |
| **SA1 (Contract Fix)** | `agent/p0-contract-fix` | Field/method/contract unification | **MERGED** |
| **SA2 (Path Compat)** | `agent/p0-path-compat` | Path/runs-root/out-dir/filename compat | `DONE` |
| **SA3 (Threshold Tuning)** | `agent/p0-threshold-tuning` | Correction profile tuning | `PENDING` |
| **SA4 (Final Report)** | `TBD` | Consolidation report & decision | `PENDING` |

---

## Phase 0.5 — SA1 Integration Results

### SA1 Merge Summary
- **Merge commit**: `c684881` (--no-ff merge from `7ee74b3`)
- **Files changed**: ~136 files (pipeline, OOF learner, adapters, RT916, SGDFNet, CLI, docs, tests)
- **SA1 branch tip**: `7ee74b3` (local branch, not pushed to origin)

### SA1 Contract Fixes Verified

| Fix | Description | Status | Notes |
|-----|-------------|--------|-------|
| FIX-01 | sMAPE_floor50 canonical formula | ✅ PASS | Canonical formula verified via test (individual floor at 50, denom/2, *100) |
| FIX-02 | PREDICTION_PACK_COLUMNS update | ⚠️ CANNOT VERIFY | `schema.py` lives on SA3's commit `48b69fd` — verify after SA3 merge |
| FIX-03 | hour → hour_business | ⚠️ CANNOT VERIFY | P0 eval scripts in SA3's commit `48b69fd` |
| FIX-04 | is_spike → label_high_spike | ⚠️ CANNOT VERIFY | P0 eval scripts in SA3's commit `48b69fd` |
| FIX-05 | business_day mapping for hour=24 | ✅ PASS | Verified in `data_contract.py` + 3 synthetic cases + 24h round-trip |
| FIX-06 | p0_shared_contract.md | ✅ EXISTS ON SA3 | SA1 did not commit; SA3's commit `48b69fd` includes it |
| FIX-07 | p0_execution_board.md | ✅ CREATED BY RUNNER | This board replaces SA1's missing doc |

### Synthetic Time Mapping Test

```
Timestamp -> business_day, hour_business:
  2026-01-02 00:00:00 -> 2026-01-01, hb=24  ✓
  2026-01-01 13:00:00 -> 2026-01-01, hb=13  ✓
  2026-01-01 08:00:00 -> 2026-01-01, hb=8   ✓
  2026-01-01 23:00:00 -> 2026-01-01, hb=23  ✓
  2026-01-02 01:00:00 -> 2026-01-02, hb=1   ✓

Round-trip (all 24 hours):                      ✓
Business_day -> Timestamp -> Business_day
```

### py_compile Results

All key SA1-affected modules compiled without error:
- `data_contract.py` ✓ | `parser.py` ✓ | `production_pipeline.py` ✓
- `validation_tap.py` ✓ | `r3d_output_validator.py` ✓
- `rolling_oof/adapters/base.py` ✓ | `rt916.py` ✓ | `sgdfnet.py` ✓
- `timemixer.py` ✓ | `timesfm.py` ✓ | `scheduler.py` ✓ | `contracts.py` ✓
- `fusion/learners/r3d_tap_gef.py` ✓ | `metrics.py` ✓
- `main.py` ✓

---

## Blocker Register

| ID | Blocker | Owner | Status | Resolution |
|----|---------|-------|--------|------------|
| B1-B7 | Original SA1 contract blockers | SA1 | `RESOLVED` | Merged via SA1 branch |
| **B8** | **SA1 contract docs not on runner branch** | **SA1** | **OPEN** | SA1 FIX-06/07 claimed `p0_shared_contract.md` and `p0_execution_board.md` but neither was committed to SA1 branch. `p0_shared_contract.md` exists on SA3's commit `48b69fd` and will arrive with SA3 merge. Runner created this `p0_execution_board.md`. |
| **B9** | **SA1 fixes reference scripts on SA3 branch** | **Runner** | **OPEN** | FIX-01/02/03/04 reference scripts that exist only on SA3's branch. Verification deferred until SA3 merge. |
| **B10** | **All P0 scripts untracked on runner branch** | **SA3** | **OPEN** | All P0 pipeline scripts exist only as untracked files. SA3 branch must commit and merge them (`48b69fd` has them). |
| **B11** | **SA1 merge brought large data artifacts** | **Runner** | **OPEN** | SA1 merge included ~259 CSV/XLSX files (TimesFM cache, oof_runs). Needs .gitignore audit. |

---

## Phase 1-4 Execution — PENDING (Wait for SA2 + SA3)

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

---

## Exchange Artifacts

All agent-to-agent exchange artifacts are written to `reports/local/p0_exchange/`.

| Artifact | Producer | Consumer | Description | Status |
|----------|----------|----------|-------------|--------|
| `contract_manifest.json` | SA1 | All | Canonical contract definitions | ⚠️ Untracked (local only) |
| `path_manifest.json` | SA2 | All | File/path conventions | `PENDING` |
| `tuning_manifest.json` | SA3 | All | Correction profile params | `PENDING` |
| `report_manifest.json` | SA4 | All | Final report metadata | `PENDING` |

---

## Notes

- SA1 merge completed successfully. Contract documentation files not committed — tracking as B8.
- FIX-02/03/04 require SA3 merge for full verification — tracking as B9.
- All P0 scripts are untracked — SA3 must commit before Phase 1-4.
- Large data artifacts from SA1 history need cleanup — tracking as B11.
- Do NOT modify `production_pipeline.py`, model training entries, or negative-price module.
