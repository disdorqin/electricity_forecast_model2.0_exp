# P5 Model-Zoo Dataset Report

> **Purpose**: Unified training/validation/test dataset for ALL P5 model windows (W2/W3).
> **Status**: ✅ COMPLETE — 2880 timestamps, 28 features, 4 model predictions, no leakage.
> **Branch**: `agent/p5-canonical-model-zoo-pack`
> **Generated**: 2026-07-02

---

## 1. Data Sources

| Source | Rows | Date Range | Usage |
|--------|------|------------|-------|
| `data/shandong_pmos_hourly.csv` | 39,168 | 2022-01-01 ~ 2026-06-09 | Raw features (12 forecast cols, calendar, derived) |
| `reports/local/p4_canonical/canonical_prediction_pack.csv` | 2,879 | 2025-11-01 ~ 2026-02-28 | y_true, model predictions (4 models) |
| `reports/local/p4_canonical/canonical_risk_predictions.csv` | 2,879 | 2025-11-01 ~ 2026-02-28 | Risk scores (high_spike_prob, spike_risk_score) |

---

## 2. Output Files

All under `reports/local/p5_model_zoo/` (gitignored):

| File | Rows | Days | Description |
|------|------|------|-------------|
| `train_panel.csv` | 2,208 | 92 | Training features + y_true + model predictions |
| `valid_panel.csv` | 360 | 15 | Validation features + y_true + model predictions |
| `test_panel.csv` | 312 | 13 | Test features + y_true + model predictions |
| `feature_manifest.json` | — | — | Column-level metadata with roles and leakage checks |
| `prediction_schema.json` | — | — | Unified model output schema for W2/W3 |

---

## 3. Date Coverage

| Split | Start | End | Days | Rows |
|-------|-------|-----|:----:|:----:|
| Train | 2025-11-01 | 2026-01-31 | 92 | 2,208 |
| Valid | 2026-02-01 | 2026-02-15 | 15 | 360 |
| Test | 2026-02-16 | 2026-02-28 | 13 | 312 |
| **Total** | **2025-11-01** | **2026-02-28** | **120** | **2,880** |

Partitions are disjoint with no overlap.

---

## 4. Feature Summary

### Source Forecast Columns (12 → English)
| Chinese | English | Description |
|---------|---------|-------------|
| 日前电价 | dayahead_price | Day-ahead electricity price |
| 地方电厂总加预测值 | local_plant_forecast | Local plant total forecast |
| 联络线受电负荷预测值 | interconnect_forecast | Interconnection load forecast |
| 风电总加预测值 | wind_forecast | Wind total forecast |
| 光伏总加预测值 | solar_forecast | Solar total forecast |
| 核电总加预测值 | nuclear_forecast | Nuclear total forecast |
| 自备机组总加预测值 | self_owned_forecast | Self-owned units forecast |
| 试验机组总加预测值 | test_units_forecast | Test units forecast |
| 直调负荷预测值 | load_forecast | Direct dispatch load forecast |
| 竞价空间预测值 | bidding_space_forecast | Bidding space forecast |
| 新能源总加预测值 | renewable_forecast | Renewable total forecast |

### Calendar Features (6)
hour_business, weekday, day_of_week, month, day_of_month, is_weekend, period

### Derived Features (10)
net_load, solar_ratio, net_load_sq, bidding_space, space_ratio, wind_ratio, renew_penetration, ramp_load, ramp_solar, lag_price_target, lag_price_week

### Model Predictions (4)
y_pred_lightgbm, y_pred_dayahead_proxy, y_pred_naive_lag1, y_pred_naive_lag7

### Risk Scores (3)
high_spike_prob, spike_risk_score, spike_risk_flag

### Target (evaluation only)
y_true — real-time electricity price

---

## 5. Leakage Checks

| Check | Status | Detail |
|-------|--------|--------|
| 实际值 columns dropped | ✅ | 10 columns removed |
| No forbidden features | ✅ | y_true kept only as evaluation target |
| No realtime_price as feature | ✅ | Used only for lag computation, then dropped |
| No residual/abs_error/smape | ✅ | No eval metrics in feature set |
| business_day+hour_business key | ✅ | 2880 unique timestamps across all panels |
| 00:00 → previous day 24:00 | ✅ | 1-second offset applied |

---

## 6. Prediction Schema

All W2/W3 model outputs MUST conform to this schema:

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| model_name | string | ✅ | Model identifier |
| business_day | string (YYYY-MM-DD) | ✅ | Business date |
| hour_business | int (1-24) | ✅ | Business hour |
| timestamp | string (ISO) | ✅ | Physical timestamp |
| y_pred | float | ✅ | Model prediction |
| source_file | string | ✅ | Generating script path |
| prediction_mode | string | ✅ | eval/live/backfill |
| leakage_safe | boolean | ✅ | Must be True |

Schema file: `reports/local/p5_model_zoo/prediction_schema.json`

---

## 7. Known Issues

- **1 NaN in test panel**: `business_day=2026-02-28, hour_business=24` maps to `2026-03-01 00:00` which is outside the canonical date range. This is documented and expected.
- **13 NaN in y_pred_lightgbm**: The canonical pack has 17 lightgbm prediction gaps; 13 fall in the training split. Models should handle missing predictions appropriately (e.g., impute from other models).
- **Source data discrepancy**: Source CSV has `realtime_price` as the target column. This is dropped from features and only used for lag feature computation.

---

## 8. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `scripts/build_p5_model_zoo_dataset.py` | **NEW** | Dataset builder script |
| `tests/test_p5_model_zoo_dataset.py` | **NEW** | 15 tests for dataset quality |
| `docs/reports/P5_model_zoo_dataset_report.md` | **NEW** | This report |
| `docs/p3_execution_board.md` | MODIFIED | Added P5 Model-Zoo entry |
| `.gitignore` | MODIFIED | Added test file negation |
| `reports/local/p5_model_zoo/*` | NEW (gitignored) | Dataset outputs |
