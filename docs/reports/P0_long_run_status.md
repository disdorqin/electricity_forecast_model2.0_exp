# P0 Long-Run Bootstrap — Status Report

**Generated**: 2026-06-29
**Pipeline**: Phase 1B (6-loop long-run bootstrap)
**Objective**: Generate model predictions for P0 window (2025-11-01 ~ 2026-02-28) and run full spike correction pipeline end-to-end.

---

## 1. Pack Level

| Level | Model | Source | Rows | Status |
|-------|-------|--------|------|--------|
| **Level 1** (primary) | LightGBM ThreeStageLGBM | `run_p0_lightweight_predictions.py` | 2,862 | ✅ Available |
| Level 0 (baseline) | naive_lag1/lag7/dayahead_proxy/baseline_fusion | `build_baseline_prediction_pack.py` | 13,632 | ✅ Available |

The Level 1 pack is the primary evaluation target. Level 0 exists as a fallback validation.

## 2. Date Coverage

| Window | Dates | Coverage | LightGBM Rows |
|--------|-------|----------|---------------|
| 2025-11-01 ~ 2025-11-30 | 30 | 30/30 (100%) | 720 |
| 2025-12-01 ~ 2025-12-31 | 31 | 31/31 (100%) | 744 |
| 2026-01-01 ~ 2026-01-31 | 31 | 31/31 (100%) | 744 |
| 2026-02-01 ~ 2026-02-28 | 28 | 28/28 (100%) | 654 |
| **Total** | **120** | **120/120 (100%)** | **2,862** |

Coverage is 24 hours/day × 120 days = 2,880 expected; 2,862 actual (18 gap hours due to pipeline boundary trimming).

## 3. Model Coverage

| Model | P0 Ready? | Used in Pack? | Checkpoint | Runtime |
|-------|-----------|---------------|------------|---------|
| **LightGBM** | ✅ Yes | ✅ Level 1 | `models/LightGBM/best_model_实时电价.pkl` | ~5 min CPU |
| TimesFM | ❌ Legacy TF | ❌ | Pre-trained only | ~30 min GPU |
| TimeMixer | ❌ No checkpoint | ❌ | None | Unknown |
| SGDFNet | ❌ No checkpoint | ❌ | None | ~15 min GPU |
| RT916 | ❌ No P0 checkpoint | ❌ | Only May-Jun 2026 | ~2 hr GPU |
| Fusion | ❌ Needs predictions | ❌ | N/A | N/A |

Limitation: The Level 1 pack is single-model (LightGBM only). Multi-model fusion is not available for the P0 window.

## 4. Performance Metrics (Level 1 LightGBM)

### Overall

| Metric | Value |
|--------|-------|
| Overall sMAPE (floor50) | 22.02 |
| 9_16 sMAPE (floor50) | 28.16 |
| Normal hours sMAPE (floor50) | 18.91 |
| High Spike MAE | 260.56 |
| High Spike sMAPE (floor50) | 46.95 |

### Error Distribution

| Category | Count |
|----------|-------|
| Severe Underestimates (y_true - y_pred > 200) | **80** |
| High Spike Events (y_true > P95=478.5) | 144 |
| High Spike Events in 9_16 | 48 |
| Hours evaluated | 2,862 |

### Period Breakdown

sMAPE is highest in 9_16 (peak solar hours), where the model struggles with the rapid ramp from solar generation to load demand.

## 5. Extreme Event Diagnostics

### High Spike Events by Month

| Month | Events | Avg Price | Max Price | 9_16 % |
|-------|--------|-----------|-----------|--------|
| 2025-11 | 36 | 532 | 1,408 | 13.9% |
| 2025-12 | 6 | 586 | 660 | 100.0% |
| 2026-01 | 73 | 612 | 1,291 | 47.9% |
| 2026-02 | 29 | 532 | 701 | 6.9% |

### Top Spike Hours

| Hour | Spike Count | Avg Price | Max Price |
|------|------------|-----------|-----------|
| 12 | 7 | 867 | 1,408 |
| 11 | 6 | 789 | 1,275 |
| 10 | 5 | 755 | 1,291 |

## 6. Risk Model

| Attribute | Value |
|-----------|-------|
| Algorithm | RandomForestClassifier |
| Estimators | 200 |
| Max Depth | 10 |
| Training Samples | 31,334 |
| Test Samples | 7,834 |
| **ROC AUC** | **0.929** |
| Recall (class 1) | 0.83 |
| Precision (class 1) | 0.50 |

### Top 5 Features

| Feature | Importance |
|---------|-----------|
| 竞价空间实际值 | 0.203 |
| 竞价空间预测值 | 0.095 |
| 新能源总加实际值 | 0.091 |
| 光伏总加实际值 | 0.081 |
| 核电总加实际值 | 0.062 |

## 7. Correction Results

### Per-Profile Comparison

| Profile | Threshold | Max Lift | 9_16 Boost | Lift Applied | sMAPE | 9_16 sMAPE | Spike MAE |
|---------|-----------|----------|------------|-------------|-------|------------|-----------|
| Conservative | 0.75 | 0.20 / 200 | 1.0 | **0** | 22.02 | 28.16 | 260.56 |
| Medium | 0.60 | 0.35 / 350 | 1.15 | **0** | 22.02 | 28.16 | 260.56 |
| Aggressive | 0.45 | 0.60 / 600 | 1.30 | **0** | 22.02 | 28.16 | 260.56 |

### Lift Rejection Breakdown (same for all profiles)

| Rejection Reason | Count |
|-----------------|-------|
| Low probability (< threshold) | 2,463 |
| Negative base residual | 399 |
| Normal hour protection | 0 |
| **Lift applied** | **0** |

### Why No Correction Was Applied

1. **Single-model pack**: `base_fused_pred` = LightGBM `y_pred`. The guardrail `base_fused_pred > y_pred` (fused prediction must already exceed the raw prediction) is never true for any row.

2. **Risk model probability distribution**: The RandomForest produces probabilities heavily skewed toward 0. Even at the aggressive threshold (0.45), very few rows are flagged for correction.

3. **Guardrail interaction**: Even if the probability threshold were crossed, the negative base residual check (`base_fused_pred - y_pred <= 0`) would block all corrections in a single-model pack.

## 8. GO / CONDITIONAL / NO-GO Assessment

### **CONDITIONAL — Pipeline Operational, Correction Not Triggering**

| Criterion | Assessment | Detail |
|-----------|-----------|--------|
| Prediction available? | ✅ PASS | Level 1 LightGBM, full 120-day coverage |
| Pipeline executes? | ✅ PASS | All 5 phases completed without errors |
| Risk model trains? | ✅ PASS | AUC 0.929, recall 0.83 |
| Correction applied? | ❌ FAIL | 0 lift_applied across all profiles |
| Metrics acceptable? | ⚠️ BORDERLINE | sMAPE 22 overall but 80 severe underestimates |

### Conditions to reach GO

1. **Multi-model fusion pack**: Add at least one more model (TimesFM or SGDFNet) so `base_fused_pred` differs from individual `y_pred`, enabling the guardrail to pass.
2. **Risk model threshold tuning**: Lower spike_prob_threshold (e.g., 0.30 or 0.20) to increase flagging rate, or use the probability as a continuous weight instead of a binary gate.
3. **RT916 for spike days**: Run RT916 inference on the top 3 spike dates (2025-11-08, 2026-01-26, 2026-01-18) to test whether a deep-learning model captures spikes better.

### Conditions to reach NO-GO (stop)

- Complete inability to improve correction metrics after multi-model + threshold tuning
- Pipeline remains single-model only and severe underestimates exceed 100

## 9. RT916 Recommendation

**DEFER** — No checkpoint exists for the P0 window. Full RT916 training (~3-4 hours GPU) is not justified until:
1. Level 1 evaluation is reviewed and deemed insufficient
2. Multi-model fusion is confirmed as the path forward
3. The top 3 spike dates are selected for selective inference

## 10. Next Steps

| Priority | Action | Owner |
|----------|--------|-------|
| P0 | Review this report and approve/reject CONDITIONAL status | User |
| P1 | Diagnose why lift_applied=0 and propose threshold adjustments | Dev |
| P2 | Evaluate RT916 selective inference for top 3 spike dates | Dev (after approval) |
| P3 | Add TimesFM inference for P0 window (requires legacy TF fix) | Dev |
| P4 | Multi-model fusion pack construction | Dev |

## 11. Pipeline Outputs

```
reports/local/p0_full_run/
├── inventory/
│   ├── prediction_source_inventory.json
│   └── prediction_source_inventory.md
├── prediction_pack_level0/
│   ├── prediction_pack_realtime_level0_2025_11_01_2026_02_28.csv
│   ├── baseline_pack_manifest.json
│   └── baseline_coverage_report.md
├── prediction_pack_level1/
│   ├── prediction_pack_realtime_level1_2025_11_01_2026_02_28.csv
│   ├── level1_manifest.json
│   └── level1_model_coverage_report.md
├── final/
│   ├── diagnostics/extreme_events/
│   │   ├── extreme_event_report.md
│   │   ├── extreme_event_report.csv
│   │   ├── bad_case_samples.csv
│   │   ├── monthly_extreme_summary.csv
│   │   └── period_extreme_summary.csv
│   ├── spike_dataset/
│   │   └── spike_training_dataset.csv
│   ├── risk_model/
│   │   ├── spike_model_info.json
│   │   └── spike_risk_predictions.csv
│   └── correction/
│       ├── conservative/  (metrics_summary.json, correction_manifest.json, correction_result.csv)
│       ├── medium/        (same structure)
│       └── aggressive/    (same structure)
├── level0/
│   └── (same structure as final/, from Level 0 pack evaluation)
├── prediction_pack/
│   └── (initial build_backtest attempts — 0-row packs)
└── rt916_selective_plan.md
```

## 12. Known Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Correction lift not applied — single-model guardrail | High | Open — architecture limitation |
| Output path double-nesting (profile/profiles/profile/) | Low | Cosmetic — files written correctly |
| Risk model logger TypeError (%d with list) | Low | Non-fatal, model trains fine |
| Python UnicodeEncodeError on Windows (Chinese chars) | Low | Terminal-only, files written with correct encoding |
