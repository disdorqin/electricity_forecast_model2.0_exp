# P5M Final Release Report

> Generated: 2026-07-03
> Branch: `tune-timemixer` @ d146b0e
> Status: All 3 PRs merged

---

## 1. What Was Merged

| PR | Branch | Content |
|---|---|---|
| **PR #17** | `agent/p5m-plugin-interface` | Plugin/interface module (`plugin/`), negative residual base (`extreme/negative_price/`), P5 schema contract |
| **PR #19** | `agent/p5m-negative-risk-calibration` | Heuristic V2 + Rolling ML risk scorers, calibration script, health monitor, improvement convention unification |
| **PR #18** | `agent/p5m-unified-residual-stack` | Unified residual stack orchestrator, risk source classification, evaluation script, report with `[official]`/`[dry-run]`/`[data-missing]` policy |

### Files Added (net +3894 lines)
- `plugin/` — Plugin interface, schema, pipeline adapter, external loader
- `extreme/negative_price/` — Risk models (heuristic V2, rolling ML scorer), residual correction, guardrail
- `residual_stack/` — Orchestrator, priority, metrics, report, risk source, schema
- `scripts/` — `calibrate_p5m_negative_risk.py`, `evaluate_p5m_residual_stack.py`, `monitor_p5m_residual_health.py`
- `tests/` — 4 test suites (154 total tests)
- `docs/reports/` — P5M unified residual stack report, negative risk calibration report

---

## 2. Current Champion

Phase2 (`lightgbm_anchor_90` + medium profile + normal mode) remains the **deployment champion**:

| Metric | Value |
|---|---|
| sMAPE | 20.8675 |
| severe | 63 |

The residual stack operates as a **post-processing layer on top of the champion** — it does not replace the base model.

---

## 3. Residual Module Status

| Module | Ready | Status |
|---|---|---|
| Plugin/Interface | ✓ | P5 schema contract enforced, 70 tests |
| Negative Residual Correction | ✓ | Downward correction, guardrail, mutual exclusion with high-spike |
| Negative Risk Calibration | ✓ | Heuristic V2 + Rolling window RF, leakage-safe, calibrated_prob |
| Unified Residual Stack | ✓ | High-spike → negative → guardrail pipeline, risk source policy |
| Health Monitor | ✓ | Continuous monitoring script, GO/NO-GO verdict |

---

## 4. Official E2E Result

Canonical pack: `reports/local/p4_canonical/canonical_prediction_pack.csv` (2879 rows, 120 days)

| Config | Verdict | Detail |
|---|---|---|
| **A** baseline | `[official] GO` | Phase2 baseline (no corrections) |
| **B** high_spike only | `[data-missing] DATA-MISSING` | No real `high_spike_prob` data available — canonical pack has only binary `high_spike_flag` |
| **C** negative-only | `[official] GO` | Calibrated negative risk used — all GO conditions met |
| **D** unified stack | `[data-missing] DATA-MISSING` | Unified stack needs spike risk data for high_spike step |

**Overall Verdict: GO**

---

## 5. Negative/Low-Valley Improvement (Config C, aggressive profile)

| Metric | Before | After | Improvement | GO? |
|---|---|---|---|---|
| negative_MAE | 108.56 | 104.95 | **+3.32%** | ✓ (>= 0) |
| low_valley_MAE | 112.64 | 108.78 | **+3.42%** | ✓ (>= 0) |
| overall_sMAPE | 43.7227 | 43.7082 | **+0.0146** | ✓ (>= -0.3) |
| high_spike_MAE | 59.74 | 60.06 | **-0.54%** | ✓ (>= -3%) |
| normal_degradation | 36.73 | 36.66 | **-0.0733** | ✓ (<= 0.5) |
| severe_underestimate | 299 | 299 | 0 | unchanged |

**Interpretation:**
- Negative correction improves negative price MAE by **3.32%** and low-valley MAE by **3.42%**
- Overall sMAPE is essentially flat (+0.01 improvement)
- High-spike MAE degrades by only 0.54% (well within the -3% limit)
- Normal hours see slight improvement (degradation = -0.07, well below 0.5 limit)
- The correction is **safe and effective** with aggressive profile

---

## 6. Risk-Source Policy

| Risk Source | Description | Policy |
|---|---|---|
| `real_prob` | Explicit spike risk CSV path | → `official` |
| `calibrated_prob` | `high_spike_prob` column in pack | → `official` |
| `synthetic_flag` | Binary `high_spike_flag` only | → `dry-run` (with `--allow-synthetic-risk`) |
| | | → `data-missing` (without flag) |
| `missing` | No spike risk data at all | → `data-missing` |

The negative risk source (negative_prob, low_valley_prob) is always **calibrated_prob** when loaded via `--negative-risk-path`.

---

## 7. Next Integration Point

1. **External model branches** should produce prediction CSV via the `plugin/` schema:
   - Required columns: `model_name`, `business_day`, `hour_business`, `timestamp`, `y_pred`, `source_file`, `prediction_mode`, `leakage_safe`
   - Optional: `y_true`, `target_day`, `ds`, `train_end_date`, `model_version`, `prediction_spread`, `high_spike_prob`, `min_pred_last_24h`, `max_pred_last_24h`

2. **Spike risk branch** should produce `high_spike_prob` CSV:
   - When real spike risk data becomes available, Config B and D will produce `[official]` verdicts
   - Columns: `business_day`, `hour_business`, `high_spike_prob` (continuous 0-1)

3. **Monitoring** can be scheduled via:
   ```bash
   python scripts/monitor_p5m_residual_health.py \
     --canonical-pack <latest_pack> \
     --risk-path <latest_risk> \
     --out-dir <monitor_dir>
   ```
