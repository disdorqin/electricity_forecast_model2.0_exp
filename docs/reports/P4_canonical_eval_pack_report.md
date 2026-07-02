# P4 Window 1: Canonical Evaluation Pack Report

> **Purpose**: Locked-down evaluation pack that ALL P4 windows must use for consistent metric computation.
> **Status**: ✅ COMPLETE — Phase2 champion metrics reproduce within tolerance.
> **Branch**: `agent/p4-canonical-eval-pack`
> **Generated**: 2026-07-01 16:54

---

## 1. Overview

The canonical evaluation pack provides a single source of truth for all P4 window experiments. It locks:

- **Date range**: 2025-11-01 ~ 2026-02-28 (120 business days)
- **Metric level**: Timestamp-level `(business_day, hour_business)` — no row-level inflation
- **Fusion method**: `lightgbm_anchor_90` (0.9 × lightgbm + 0.1 × mean of naive/dayahead)
- **Correction profile**: medium + normal (spike_prob_threshold=0.60, max_lift_ratio=0.35, max_absolute_lift=350, period_9_16_boost=1.15)
- **Base models**: dayahead_proxy, naive_lag1, naive_lag7, lightgbm
- **sMAPE formula**: floor50 (`max(|y|, 50)` on both y_true and y_pred) — matches Phase2 champion evaluation

---

## 2. Output Files

All under `reports/local/p4_canonical/` (gitignored):

| File | Rows | Description |
|------|------|-------------|
| `canonical_prediction_pack.csv` | 2879 | 1 row/timestamp with wide model columns, reference correction |
| `canonical_risk_predictions.csv` | 2879 | high_spike_prob aligned to same timestamps |
| `canonical_metrics_baseline.json` | — | Phase2 champion metric reproduction |
| `canonical_manifest.json` | — | Full metadata, completeness, anomalies |

---

## 3. Completeness

| Metric | Value |
|--------|-------|
| Expected timestamps | 2880 (120 days × 24h) |
| Actual timestamps | 2879 |
| Missing | 1 — `(2026-02-28, hour_business=24)` |
| Business days | 120 |

### Missing Timestamp Detail

- **2026-02-28 hour_business=24**: This maps to 2026-03-01 00:00 which is outside the source data range. Not a data integrity issue.

### Model Coverage

| Model | Present | Missing | % Missing |
|-------|---------|---------|-----------|
| dayahead_proxy | 2856 | 23 | 0.80% |
| naive_lag1 | 2856 | 23 | 0.80% |
| naive_lag7 | 2856 | 23 | 0.80% |
| lightgbm | 2862 | 17 | 0.59% |

Missing model rows are primarily at hour_business=24 on Fridays (which map to Saturday 00:00).

---

## 4. Phase2 Champion Reproduction

| Metric | Expected | Actual | Δ | Result |
|--------|----------|--------|---|--------|
| sMAPE_floor50 | 20.86 | 20.8675 | +0.0075 | ✅ PASS (tolerance ±0.05) |
| Severe underestimates | 63 | 63 | 0 | ✅ PASS (exact) |
| Base sMAPE_floor50 (uncorrected) | — | 21.2093 | — | — |

### Detailed Baseline Metrics

| Metric | Value |
|--------|-------|
| n_timestamps | 2879 |
| sMAPE_floor50 | 20.8675 |
| base_sMAPE_floor50 | 21.2093 |
| severe_underestimate_count | 63 |
| severe_underestimate_base | 81 |
| sMAPE_9_16_floor50 | 25.5349 |
| base_sMAPE_9_16_floor50 | 26.1598 |
| false_lift_rate | 0.0664 (6.64%) |
| normal_hours_degradation | -0.1929 (improvement) |
| n_spike_hours | 109 |
| n_non_spike_hours | 2770 |

---

## 5. Leakage Safety

✅ **No leakage columns detected.** The pack contains only:

- Calendar/timestamp fields (`business_day`, `hour_business`, `timestamp`, `period`)
- Prediction-time model outputs (`base_fused_pred`, `y_pred_*`)
- Risk model outputs (`high_spike_prob`, `spike_risk_score`, `spike_risk_flag`)
- Evaluation-only target (`y_true` — not used at prediction time)
- Derived correction fields (`final_pred_reference`, `lift_applied`, `reason_code`)

No D+1 actual features, no "实" (Chinese actual) columns, no future-looking fields.

---

## 6. Leakage Analysis

- **Dedup ratio**: 74.8% of source-pack rows removed by timestamp-level dedup (11430 → 2879). The source pack has 4 rows per timestamp (one per model). After pivoting to wide format, 1 row per timestamp.
- **Missing timestamps**: 1 (hour_business=24 on 2026-02-28, which maps to 2026-03-01 00:00 — outside the date range).
- **Model gaps**: Small number of missing model predictions (0.59–0.80%), primarily at hour_business=24 boundary.
- **False lift rate**: 6.64%, below the 10% threshold.
- **Normal hours degradation**: -0.19 (negative = correction improves normal hours too).

---

## 7. How P4 Windows Use This Pack

Each P4 window:

1. Reads `canonical_prediction_pack.csv` as the base
2. Modifies `base_fused_pred` using their window-specific strategy
3. Feeds through the correction pipeline (or their own variant)
4. Computes metrics using the canonical `compute_baseline_metrics()` function (floor50 sMAPE)
5. Compares against the `canonical_metrics_baseline.json` Phase2 champion reference

---

## 8. Files Changed

| File | Action | Description |
|------|--------|-------------|
| `scripts/build_p4_canonical_eval_pack.py` | **NEW** | Pack builder script |
| `tests/test_p4_canonical_eval_pack.py` | **NEW** | 10 tests for pack quality |
| `docs/reports/P4_canonical_eval_pack_report.md` | **NEW** | This report |
| `docs/p3_execution_board.md` | MODIFIED | Added P4 Line I entry |
| `reports/local/p4_canonical/*` | NEW (gitignored) | Pack outputs |
