# P5M Unified Residual Stack Report

## 1. Stack Architecture

```
base_pred (base_fused_pred)
    │
    ▼
┌──────────────────────────────────────┐
│  Module 1: High-spike correction     │
│  Source: extreme.realtime_high_spike │
│  Direction: upward lift only         │
│  Period-aware quantile fitting       │
└──────────────┬───────────────────────┘
               │ after_high_spike_pred
               ▼
┌──────────────────────────────────────┐
│  Module 2: Negative/low-valley       │
│  Source: extreme.negative_price      │
│  Direction: downward only            │
│  Mutual exclusion with high_spike    │
└──────────────┬───────────────────────┘
               │ after_negative_pred
               ▼
┌──────────────────────────────────────┐
│  Module 3: Final guardrail           │
│  - high_spike hours never harmed     │
│  - normal-hour degradation ≤ 20%     │
└──────────────┬───────────────────────┘
               │ final_pred
               ▼
          correction_reason
          module_sequence
```

### Output columns

| Column | Description |
|--------|-------------|
| `base_pred` | Input prediction (base_fused_pred) |
| `after_high_spike_pred` | After Module 1 (upward lift) |
| `after_negative_pred` | After Module 2 (downward correction) |
| `final_pred` | After Module 3 (deployable prediction) |
| `high_spike_applied` | bool: lift was > 0 |
| `negative_applied` | bool: downward correction was > 0 |
| `correction_reason` | Human-readable summary |
| `module_sequence` | e.g. "high_spike→negative→guardrail" |

## 2. High-spike / Negative Priority

Core rule: **high_spike takes unconditional priority over negative**.

```
IF high_spike_applied OR high_spike_prob >= 0.5:
    negative module MUST NOT apply downward correction
    correction_reason → "spike_blocks_negative"
```

This is enforced at the row level in `residual_stack/priority.py`:

- `check_high_spike_priority()` — stateless check
- `should_apply_negative()` — decides with explanation string
- The orchestrator calls these before every negative correction

## 3. Correction Fields

Each run of `ResidualStackOrchestrator` produces all `STACK_OUTPUT_COLUMNS`.
The correction reason follows deterministic rules:

| Condition | correction_reason |
|-----------|-------------------|
| No correction needed | `no_correction` |
| Spike lift applied | `high_spike_lifted` |
| Spike clipped by guardrail | `high_spike_guardrail_clipped` |
| Negative downward applied | `negative_downward` |
| Negative blocked by guardrail | `negative_blocked_by_guardrail` |
| Spike blocks negative | `spike_blocks_negative` |
| Both apply (rare) | `spike_lifted_then_negative` |
| Insufficient data | `data_limited` |

## 4. Metrics

Computed by `residual_stack/metrics.py` `compute_stack_metrics()`:

| Metric | Description |
|--------|-------------|
| `negative_count` | Rows with y_true < 0 |
| `low_valley_count` | Rows with y_true <= 50 |
| `high_spike_count` | Rows with y_true > 150 |
| `negative_MAE_before/after` | MAE on negative-price rows |
| `low_valley_MAE_before/after` | MAE on low-valley rows |
| `negative_miss_before/after` | y_true < 0 but y_pred >= 0 |
| `low_valley_overestimate_before/after` | y_pred - y_true >= 30 on low-valley rows |
| `overall_sMAPE_before/after/delta` | sMAPE with floor 50 |
| `high_spike_MAE_before/after/delta_pct` | MAE on high-spike rows |
| `false_lift_rate` | Fraction of lifts that overshoot y_true |
| `normal_sMAPE_before/after` | sMAPE on non-9_16 hours |
| `normal_degradation` | sMAPE delta on normal hours |
| `data_limited` | True if negative_count < 5 |

## 5. GO / NO-GO / DATA-LIMITED

GO conditions (all must pass):

```
1. overall_sMAPE_delta <= 0.3
2. severe_underestimate <= 63 or not worsened
3. high_spike_MAE_delta_pct <= 3%
4. low_valley_MAE_delta < 0 (improved)
5. normal_degradation <= 0.5
```

If `negative_count < 5`: verdict = **DATA-LIMITED** (low-valley still evaluated).

If any condition fails: verdict = **NO-GO** with reasons.

If all pass: verdict = **GO**.

## 6. Comparison Configurations

The evaluation script (`scripts/evaluate_p5m_residual_stack.py`) compares:

| Config | Corrections | Description |
|--------|------------|-------------|
| A | None | Phase2 champion baseline |
| B | high_spike only | Phase2 + plugin dry-run |
| C | negative only | Phase2 + negative residual |
| D | high_spike + negative | Full unified stack |
## 6. Risk Source Policy

Each config that uses spike risk data (B, D) is classified by its **risk
source**. This determines whether the verdict is official, dry-run, or
DATA-MISSING.

### Hierarchy

| RiskSource | Detection | Verdict Eligibility |
|---|---|---|
| `REAL_PROB` | Explicit `spike_risk_path` file | Official GO / NO-GO |
| `CALIBRATED_PROB` | `high_spike_prob` column in pack | Official GO / NO-GO |
| `SYNTHETIC_FLAG` | Only `high_spike_flag` column | Dry-run only |
| `MISSING` | No risk data at all | DATA-MISSING |

### Policy Resolution

| Source | `--allow-synthetic-risk` | Result |
|---|---|---|
| REAL / CALIBRATED | ignored | `can_run=True`, `status=official` |
| SYNTHETIC_FLAG | `False` (default) | `can_run=False`, `status=data_missing` |
| SYNTHETIC_FLAG | `True` | `can_run=True`, `status=dry_run` |
| MISSING | ignored | `can_run=False`, `status=data_missing` |

### Report Verdict Prefixes

| Prefix | Meaning |
|---|---|
| `[official] GO` | Real/calibrated spike risk, all GO conditions met |
| `[official] NO-GO` | Real/calibrated spike risk, condition(s) failed |
| `[official] DATA-LIMITED` | Too few negative samples (still official) |
| `[dry-run] ...` | Synthetic risk data, informative only |
| `[data-missing] DATA-MISSING` | No spike risk data (configs B/D skip) |

### Overall Verdict

The overall verdict in the report is computed from **official** results only;
dry-run and data-missing configs are excluded. If no config has official
results, the overall verdict is `NO-OFFICIAL-RESULTS`.

### Orchestrator Integration

`ResidualStackOrchestrator.run()` detects and stores the risk source
automatically. The result's `risk_source` and `run_status` fields are
available for callers to inspect.

### `--allow-synthetic-risk`

```bash
python scripts/evaluate_p5m_residual_stack.py      \
    --canonical-pack outputs/.../prediction_pack.csv \
    --allow-synthetic-risk
```

Without this flag, configs B and D produce `DATA-MISSING` when only a
binary `high_spike_flag` is available.

**Important**: Previously reported low_valley_MAE degradation (e.g., +12.69%
in earlier E2E runs) was based on synthetic risk data and is considered a
dry-run result, not an official regression.

## 7. Comparison Configurations

The evaluation script (``scripts/evaluate_p5m_residual_stack.py``) compares:

| Config | Corrections | Risk Dependency | Description |
|---|---|---|---|
| A | None | None | Phase2 baseline (always official) |
| B | high_spike only | Spike risk data | Phase2 + plugin |
| C | negative only | None | Phase2 + negative residual (always official) |
| D | high_spike + negative | Spike risk data | Full unified stack |

## 8. Future Production Integration Path

1. **Smoke test**: Run `evaluate_p5m_residual_stack.py` on canonical pack.
2. **Profile tuning**: Adjust `--high-spike-profile` and `--negative-profile`.
3. **Integration**: Add `ResidualStackOrchestrator` call in `production_pipeline.py`
   after Step 4 (Fusion) and before Step 5 (Classifier):

```python
from residual_stack.orchestrator import ResidualStackOrchestrator

orch = ResidualStackOrchestrator()
result = orch.run(
    prediction_pack_path=output_path,
    spike_risk_path=spike_risk_path,
    profile=StackProfile(
        spike_profile_name="medium",
        negative_profile_name="conservative",
    ),
)
result.df.to_csv(final_output_path, index=False)
```

## File Inventory

```text
residual_stack/
  __init__.py        — Public API
  schema.py          — STACK_OUTPUT_COLUMNS, REASON_CODES
  priority.py        — High_spike > negative priority rules
  orchestrator.py    — ResidualStackOrchestrator
  metrics.py         — compute_stack_metrics, compare_configs
  report.py          — generate_verdict, write_report
  risk_source.py     — RiskSource enum, detection, policy

scripts/
  evaluate_p5m_residual_stack.py — Evaluation script (A/B/C/D comparison)

tests/
  test_p5m_residual_stack.py     — 46+ tests

docs/reports/
  P5M_unified_residual_stack_report.md — This document
```
