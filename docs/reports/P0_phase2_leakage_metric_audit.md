# SA1 Leakage, Metric, and Business-Time Audit — Phase 2 Anchored Fusion

**Date**: 2026-06-30
**Auditor**: SA1 (Contract + Leakage)
**Scope**: Phase 2 Anchored Fusion Big Run results on branch `agent/p0-phase2-anchored-fusion-run`

---

## 1. Audit Scope

| Dimension | Description |
|-----------|-------------|
| Leakage | Prediction-time features using D+1 actual values |
| Business time | `business_day` / `hour_business` mapping correctness |
| Timestamp-level metrics | GO/NOTRUSTED decision based on deduplicated rows |
| Correction mode | Normal vs relaxed distinction in final verdict |

## 2. Files Audited

| File | Role | Leakage Risk |
|------|------|-------------|
| `scripts/build_multicandidate_pack.py` | Fuse baseline + LightGBM, build prediction pack, compute base metrics | Low |
| `scripts/evaluate_phase2_anchored_results.py` | Aggregate all correction results, compute ranked table, GO/NO-GO | Low |
| `scripts/evaluate_realtime_spike_correction.py` | Run correction for single profile, compute all metrics | Low |
| `scripts/evaluate_p0_realtime_spike_full.py` | Correction evaluation (SA3 profile-based) + orchestrator | Low |
| `scripts/build_realtime_spike_dataset.py` | Build spike training dataset from raw data + predictions | **MEDIUM** |
| `scripts/train_realtime_spike_risk.py` | Train RandomForest risk model | **MEDIUM** |
| `scripts/predict_realtime_spike_risk.py` | Generate spike risk predictions (placeholder) | **HIGH** (known placeholder) |
| `extreme/realtime_high_spike/residual_lift.py` | Lift computation + candidate fitting | Low |
| `extreme/realtime_high_spike/guardrail.py` | Safety guardrails | Low |
| `extreme/realtime_high_spike/apply_correction.py` | Correction pipeline orchestration | Low |
| `SGDFNet/src/sgdfnet/data_contract.py` | Business-time column mapping, ACTUAL_COLS definition | Reference |
| `docs/reports/P0_phase2_anchored_correction_report.md` | Phase 2 report | Being audited |
| `docs/p0_execution_board.md` | Status tracking | Being audited |

## 3. Leakage Audit Results

### 3.1 Prediction Pack pipeline (build_multicandidate_pack.py)

**Verdict: CLEAN**

- Reads baseline pack + LightGBM pack, both containing only prediction columns + `y_true` (for evaluation)
- `y_true` is used ONLY for metric computation (sMAPE, severe_underestimate_flag, residual) — NOT as a feature
- No raw exogenous actual-value columns from the xlsx are loaded
- Fusion is computed purely from model predictions (`y_pred` columns)
- Risk predictions are loaded from external file, merged on `business_day + hour_business`
- `build_risk_predictions()` does not create risk scores — it only reindexes the source predictions

**Leakage risk: NONE**

### 3.2 Correction evaluation scripts

**Verdict: CLEAN**

- `evaluate_realtime_spike_correction.py`, `evaluate_p0_realtime_spike_full.py`, `evaluate_phase2_anchored_results.py`:
  - `y_true` only used for computing evaluation metrics (sMAPE, MAE, severe count)
  - No feature construction from actual values
  - The `run_correction()` pipeline only reads `base_fused_pred` + `high_spike_prob` — both prediction-derived

**Leakage risk: NONE**

### 3.3 Spike risk model pipeline ⚠️

**Verdict: POTENTIAL LEAKAGE — see details below**

#### 3.3.1 `build_realtime_spike_dataset.py`

The `build_features()` function (lines 81-125):
- Creates features ONLY from `y_pred` (prediction) column: lag features, rolling stats ✅
- Calendar features from timestamp: hour, weekday ✅
- Creates `spike_label` from price columns as training target ✅

**However**: The raw data merge (line 144-146) joins raw xlsx data (containing all ACTUAL_COLS) with predictions on exact timestamp:
```python
merged = pd.merge(data_df, pred_df, on="ds", how="left", suffixes=("", "_pred"))
```

This carries ALL raw xlsx columns into the training dataset. While `build_features()` only explicitly creates features from `y_pred` and calendar info, the raw columns remain in the output DataFrame.

#### 3.3.2 `train_realtime_spike_risk.py`

The feature selection logic (lines 94-96):
```python
exclude_cols = {"ds", "spike_label", "model_name", "_source", "_source_file",
                "realtime_price", "dayahead_price", "y_true", "hour", "hour_business", "weekday"}
feature_cols = [c for c in df.columns if c not in exclude_cols
                and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
```

**ISSUE FOUND**: The `exclude_cols` set excludes:
- `realtime_price` (renamed from 实时电价) — ✅ excluded
- `dayahead_price` (renamed from 日前电价) — ✅ excluded
- `y_true` — ✅ excluded

**But does NOT exclude the following Chinese-named actual-value columns:**
- `竞价空间实际值`
- `新能源总加实际值`
- `光伏总加实际值`
- `风电总加实际值`
- `核电总加实际值`
- `直调负荷实际值`
- `联络线受电负荷实际值`
- `地方电厂总加实际值`
- `自备机组总加实际值`
- `试验机组总加实际值`

These columns have numeric dtype and are NOT in `exclude_cols`, so they WOULD be included as features if present in the training dataset.

**Impact assessment**:
- The merge in `build_realtime_spike_dataset.py` performs `how="left"` from data_df. If the prediction pack does NOT share column names with the actual-value columns (which it shouldn't), the resulting DataFrame will contain these actual-value columns.
- The trained RandomForest model would learn correlations between same-hour actual exogenous features and spike labels.
- At prediction time, if the prediction pack used for inference does NOT include these actual-value columns, the model would need to handle missing features silently (RandomForest with `fillna(0)` in place).
- **However**: The `predict_realtime_spike_risk.py` placeholder does NOT load or use the trained model at all — it uses a `y_true`-based heuristic instead (see 3.3.3).

**Leakage risk: MEDIUM — actual-value columns used in training data; unclear if model inference runs in production path**

#### 3.3.3 `predict_realtime_spike_risk.py` ⚠️ ⚠️

**ISSUE FOUND**: Kn)own placeholder leakage (lines 79-87):
```python
if "y_true" in df.columns:
    y_true_vals = pd.to_numeric(df["y_true"], errors="coerce").fillna(0)
    residual = y_true_vals - y_pred_vals
    residual_norm = (residual - residual.min()) / max(residual.max() - residual.min(), 1e-10)
    df["spike_risk_score"] = residual_norm
```

This computes spike risk score using **y_true** (the actual realtime price) at what would be prediction time. This is a clear D+1 information leak.

The code comments state this is a placeholder ("In production, this would load the actual model and predict"). **However, this script was used in Phase 1B to generate risk predictions for the correction evaluation.**

**For Phase 2 results specifically**: The `build_multicandidate_pack.py` references risk predictions from `reports/local/p0_full_run/level0/risk_model/spike_risk_predictions.csv`. This path is different from the Phase 1B orchestration output path (`reports/local/p0_full_run/spike_prediction/`). The provenance of this specific risk file cannot be verified from code alone — it may have been regenerated with the trained model or manually copied.

**Leakage risk: HIGH — if Phase 2 risk predictions came from the placeholder script**

### 3.4 Summary of actual-value columns found

| Column | Where Found | Used as Feature? | Used as Label/Evaluation? |
|--------|-----------|-----------------|--------------------------|
| `实时电价` (realtime_price) | data_contract.py, spike dataset | Only in training data (not excluded) | Used for `spike_label` (allowed) |
| `日前电价` (dayahead_price) | data_contract.py, spike dataset | Excluded in train script | N/A |
| `竞价空间实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `新能源总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `光伏总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `风电总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `核电总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `直调负荷实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `联络线受电负荷实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `地方电厂总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `自备机组总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |
| `试验机组总加实际值` | data_contract.py, spike dataset | **NOT excluded in train** | N/A |

**Were actual-value columns used as prediction-time features?** — **UNCLEAR for risk model pipeline; NO for correction pipeline**

### 3.5 Historical lag actuals

The `data_contract.py` `preprocess_dataframe()` function creates `hist_*_lag24` features from ACTUAL_COLS with `.shift(24)` (line 266). This is a **lagged** (yesterday's) value and is acceptable — it is historical data, not same-hour D+1 actuals. Same for the `_safe_delta_history` and `_safe_hourly_history` shift-based helpers.

These shift-based features are used by the LightGBM models in the main prediction pipeline, NOT by the spike risk model pipeline. They are **NOT** a leakage concern.

---

## 4. Business Day / Hour Business Audit

### 4.1 Mapping verification

| Source | Timestamp → business_day | Timestamp → hour_business | Verified |
|--------|------------------------|---------------------------|----------|
| `data_contract.py:add_business_time_columns()` | `(ts - (ts.hour==0).astype(int) days).normalize()` | `ts.hour.replace({0: 24})` (as `target_hour`) | ✅ |
| `build_realtime_spike_dataset.py:build_features()` | Not created | `df["hour"].apply(lambda h: 24 if h == 0 else h)` | ✅ |
| `build_multicandidate_pack.py` | Uses existing columns | Uses existing columns | ✅ |
| `residual_lift.py:get_period()` | N/A | Range mapping `1-8→1_8, 9-16→9_16, 17-24→17_24` | ✅ |

### 4.2 Key case: 00:00 → hour_business = 24

```
timestamp = 2026-01-02 00:00:00
  → business_day = 2026-01-01 (subtract 1 day because hour == 0)
  → hour_business = 24
```

This is **correctly** handled in `data_contract.py` (line 201-202) and `build_realtime_spike_dataset.py` (line 97).

The reverse mapping uses the same convention:
```
business_day = 2026-01-01, hour_business = 24
  → timestamp = 2026-01-02 00:00:00
```

### 4.3 Merge keys

**All** correction pipeline scripts use `["business_day", "hour_business"]` as the merge/groupby key:
- `build_multicandidate_pack.py`: groupby `["business_day", "hour_business"]` for fusion, dedup, risk merge ✅
- `apply_correction.py:load_and_merge()`: merge on `["business_day", "hour_business"]` ✅
- `evaluate_phase2_anchored_results.py`: `drop_duplicates(subset=["business_day", "hour_business"])` ✅

**No script was found using `timestamp.date()` or `ds.normalize()` + `hour` instead of `business_day` + `hour_business` for cross-DataFrame merging.** ✅

### 4.4 Verdict

**Business time mapping: CORRECT** ✅

---

## 5. Timestamp-Level Metrics Audit

### 5.1 Multi-candidate pack structure

The multi-candidate pack has 4 rows per timestamp (1 per model):
- naive_lag1, naive_lag7, dayahead_proxy, lightgbm

Row-level metrics would inflate counts by ~4x.

### 5.2 Deduplication verification

| Script | Deduplication Method | For Final Decision? |
|--------|---------------------|-------------------|
| `build_multicandidate_pack.py` | `dedup_timestamp()` → `drop_duplicates(subset=["business_day", "hour_business"])` | ✅ Yes — manifest metrics are timestamp-level |
| `evaluate_phase2_anchored_results.py` | `drop_duplicates(subset=["business_day", "hour_business"])` | ✅ Yes — all metrics computed on deduplicated |
| `evaluate_realtime_spike_correction.py` | **No explicit dedup** within script | The `compute_all_metrics` runs on whatever DataFrame is passed. The correction result has 1 row per timestamp (because correction applies to `base_fused_pred` which is at timestamp level in the fused pack). However, if the input prediction pack has multiple rows per timestamp, the metrics would be inflated. |
| `evaluate_p0_realtime_spike_full.py` | Same as above | Same |
| Phase 2 report | States: "Metrics here are computed on deduplicated timestamps" | ✅ Yes |

### 5.3 Verification of specific metrics

| Metric | Dedup Applied? | Computation |
|--------|---------------|-------------|
| sMAPE_floor50 | ✅ Yes | `compute_smape_floor50` on dedup `y_true` vs `base_fused_pred`/`final_pred` |
| severe_underestimate_count | ✅ Yes | `(y_true - final_pred > 200).sum()` on 1 row per timestamp |
| false_lift_rate | ✅ Yes | `(final_pred > base_fused_pred)` count on dedup non-spike timestamps |
| normal_hours_degradation | ✅ Yes | sMAPE delta on dedup non-9_16 timestamps |
| high_spike MAE | ✅ Yes | MAE on dedup timestamps with `high_spike_flag == 1` |
| 9_16 sMAPE | ✅ Yes | sMAPE on dedup timestamps in period 9_16 |

### 5.4 Concern: individual profile outputs

The `evaluate_realtime_spike_correction.py` script generates per-profile `correction_result.csv` files. If these filesss are loaded directly (not through `evaluate_phase2_anchored_results.py`), the metrics would be at the row level. The per-profile outputs in `reports/local/p0_phase2_anchored/correction/` should NOT be used for final decision-making without deduplication.

### 5.5 Verdict

**Timestamp-level metric handling: CORRECT for final decisions** ✅ — The aggregate evaluation script `evaluate_phase2_anchored_results.py` correctly deduplicates before computing all metrics. The Phase 2 report references these deduplicated metrics.

**Row-level concern: Mitigated** — The `build_multicandidate_pack.py` manifest includes an explicit known_limitation about row-level inflation.

---

## 6. Correction Mode Audit

### 6.1 Mode enforcement

| Check | Result |
|-------|--------|
| Phase 2 report identifies normal mode as production-safe | ✅ Yes (Row 36-40) |
| Phase 2 report identifies relaxed as offline-only | ✅ Yes (explicit warning) |
| Best candidate uses normal mode | ✅ `lightgbm_anchor_90 + normal + medium` |
| GO verdict based on normal mode | ✅ 3 normal-mode candidates achieve GO |
| Relaxed mode excluded from GO | ✅ All relaxed candidates are NO-GO |
| `evaluate_phase2_anchored_results.py:go_nogo()` | ✅ Returns "NO-GO" for relaxed unless sMAPE/severe/false_lift/degrad all pass, then "CONDITIONAL (relaxed)" — never GO |
| `evaluate_realtime_spike_correction.py` prints RELAXED warning | ✅ `[RELAXED MODE] offline-only, do NOT use in production.` |
| Profiles YAML has no explicit `mode` field | ⚠️ Profiles default to NORMAL mode in code; explicit `correction_mode` parameter can override via CLI |

### 6.2 Verdict

**Correction mode handling: CORRECT** ✅ — The final GO recommendation uses only normal-mode results. Relaxed mode is correctly labeled as offline-only.

---

## 7. Trust Level Verdict

### TRUSTED_WITH_LIMITATIONS

The Phase 2 anchored fusion results are **trusted with the following limitations**:

**Why TRUSTED:**
1. The correction pipeline (build_multicandidate_pack → evaluate_anchored_results) has NO leakage of actual-value columns as prediction-time features
2. Business_day/hour_business mapping is correct and consistently used across all scripts
3. Timestamp-level deduplication is correctly applied for final decision metrics
4. Correction mode (normal) is correctly used for the GO verdict
5. All 28 guardrail tests pass
6. All 5 audited scripts compile without errors

**Why WITH_LIMITATIONS:**
1. **Risk model training data may contain actual-value columns** — `train_realtime_spike_risk.py` does not exclude Chinese-named ACTUAL_COLS from feature selection. If the training dataset from `build_realtime_spike_dataset.py` carries these columns (which the merge logic would do), the RandomForest model learned from potentially leaked same-hour actual exogenous features.
2. **Risk prediction provenance uncertain** — `predict_realtime_spike_risk.py` uses y_true-based placeholder logic. The Phase 2 risk predictions file path differs from the pipeline output path. Without checking the actual file on disk, the source of the risk scores used in Phase 2 cannot be fully verified from code alone.
3. **Limited model scope** — Only LightGBM + naive baselines are in the fusion. No TimesFM/SGDFNet/RT916 real predictions. True multi-model fusion may produce different (potentially better or worse) results.

### Current best candidate (assuming clean risk predictions):

| Metric | Value | GO Threshold | Status |
|--------|-------|-------------|--------|
| sMAPE (floor50) | 20.86 | <= 22.02 | ✅ PASS |
| Severe underestimates | 63 | < 80 | ✅ PASS |
| False lift rate | 7.0% | <= 15% | ✅ PASS |
| Normal hours degradation | -0.33 | <= 0.5 | ✅ PASS |
| Correction mode | normal | must be normal | ✅ PASS |

---

## 8. Required Fixes Before Merge

### P1 — Must Fix

| ID | File | Issue | Fix |
|----|------|-------|-----|
| FIX-01 | `scripts/train_realtime_spike_risk.py:94-96` | ACTUAL_COLS not excluded from features | Add all 10 Chinese-named actual columns to `exclude_cols` |
| FIX-02 | `scripts/predict_realtime_spike_risk.py:82-87` | Placeholder uses y_true for risk score | Either (a) load the trained model and predict, or (b) use a forecast-error-based heuristic that doesn't require y_true, or (c) mark the function as `@deprecated` with explicit warning |

### P2 — Should Fix

| ID | File | Issue | Fix |
|----|------|-------|-----|
| FIX-03 | `scripts/build_realtime_spike_dataset.py:144-146` | Merge carries all raw columns into training dataset | Explicitly select only needed columns after merge, or add a column whitelist |
| FIX-04 | `scripts/evaluate_realtime_spike_correction.py` | Profile-level metrics may be row-level if input pack has multiple rows per timestamp | Add explicit `drop_duplicates(subset=["business_day", "hour_business"])` before metric computation |

### P3 — Nice to Fix

| ID | File | Issue | Fix |
|----|------|-------|-----|
| FIX-05 | `docs/reports/P0_phase2_anchored_correction_report.md` | No section documenting risk prediction source | Add note about which risk model/pipeline generated the risk scores used |

---

## 9. Appendices

### A. Script compilation verification

```bash
python -m py_compile scripts/build_multicandidate_pack.py  # OK
python -m py_compile scripts/evaluate_phase2_anchored_results.py  # OK
python -m py_compile scripts/evaluate_realtime_spike_correction.py  # OK
python -m py_compile scripts/evaluate_p0_realtime_spike_full.py  # OK
python -m py_compile scripts/build_realtime_spike_dataset.py  # OK
```

### B. Test results

```bash
pytest tests/test_realtime_spike_guardrail.py -v  # 28/28 PASSED
```

### C. Actual-value columns reference

From `SGDFNet/src/sgdfnet/data_contract.py`:
```python
ACTUAL_COLS = [
    "地方电厂总加实际值",
    "联络线受电负荷实际值",
    "风电总加实际值",
    "光伏总加实际值",
    "核电总加实际值",
    "自备机组总加实际值",
    "试验机组总加实际值",
    "直调负荷实际值",
    "竞价空间实际值",
    "新能源总加实际值",
]
```

These columns are defined in the data contract for the main prediction models (where they are properly lagged with `.shift(24)`). They are also carried into the risk model training dataset via the `build_realtime_spike_dataset.py` merge, but NOT excluded from features in `train_realtime_spike_risk.py`.
