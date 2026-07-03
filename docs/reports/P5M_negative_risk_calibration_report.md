# P5M Negative Risk Calibration Report

## Overview

This report documents the calibration of negative/low-valley risk scoring
for the P5M negative price residual correction module.

## Problem

The original heuristic risk model used a hard percentile threshold
(`base_fused_pred <= p5` / `p10`), producing binary 0/1 risk scores.
This was too conservative: for many real-world prediction packs the
prediction distribution did not overlap with the label distribution,
resulting in **zero corrections** across all profiles.

## Solution: Two Risk Scorers

### 1. `heuristic_v2` (rule-based, no training)

Computes continuous 0-1 risk scores by combining multiple prediction-time
safe signals:

| Signal | Weight (Neg) | Weight (LV) | Rationale |
|--------|-------------|-------------|-----------|
| Hour factor | 10% | 10% | Early morning / late night = higher risk |
| Pred level | 20% | 25% | Lower pred → higher risk |
| Recent neg rate | 25% | 10% | Historical negative rate in same hour |
| Recent low rate | — | 20% | Historical low-valley rate in same hour |
| Residual factor | 15% | 15% | Mean historical overprediction in low regime |
| Renewable ratio | 10% | 10% | High renewable → low price risk |
| Prediction spread | 5% | 5% | Model disagreement → uncertainty |

All features are **leakage-safe**:
- No y_true, residuals, or actual columns at prediction time
- `recent_mean_low_residual_by_hour` computed from historical data only

### 2. `rolling_ml_low_valley` (walk-forward ML)

For each day D:
- Train on `[D - train_window, D - 1]`
- Predict D
- Uses RandomForest with balanced class weights
- Falls back to heuristic_v2 when training fails

**Target labels:** `combined` (low_valley OR negative_price)

**Features:** Same prediction-time-safe set as heuristic_v2, plus
forecast exogenous columns (风电, 光伏, 负荷, 竞价空间).

## Risk CSV Schema

All scorers output:

```csv
business_day,hour_business,negative_prob,low_valley_prob,risk_source,leakage_safe
```

## Calibration Results

Run via:

```bash
python scripts/calibrate_p5m_negative_risk.py \
    --canonical-pack reports/local/canonical_eval_pack.csv \
    --out-dir reports/local/p5m_calibration
```

## Monitor Dashboard

The health monitor tracks operational metrics:

```bash
python scripts/monitor_p5m_residual_health.py \
    --canonical-pack reports/local/canonical_eval_pack.csv \
    --profile conservative
```

### Monitored Metrics

| Metric | Source | Target |
|--------|--------|--------|
| `negative_count` | Label | Count > 0 |
| `low_valley_count` | Label | Count > 0 |
| `negative_trigger_rate` | heuristic_v2 | > 0 |
| `low_valley_trigger_rate` | heuristic_v2 | > 0 |
| `high_spike_overlap_count` | Data | Minimize |
| `downward_correction_count` | Pipeline | Count > 0 |
| `normal_degradation` | Metrics | <= 0.5 |
| `high_spike_MAE_delta` | Metrics | <= 3% |
| `DATA_LIMITED` | Label | False if negative_count=0 |

## File Structure

```
extreme/negative_price/
  risk_model.py           — heuristic_v2 + RollingLowValleyScorer

scripts/
  calibrate_p5m_negative_risk.py     — Calibration runner
  monitor_p5m_residual_health.py     — Health monitor

tests/
  test_p5m_negative_risk_calibration.py  — 12+ tests

docs/reports/
  P5M_negative_risk_calibration_report.md — This report
```
