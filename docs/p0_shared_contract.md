# P0 Shared Contract — Realtime High Spike Correction

## Purpose

Unify CLI interface and metric definitions across all P0 offline-evaluation scripts,
and define the tuning profile contract for spike correction.

## Universal CLI Flags

| Flag | Required | Default | Description |
|---|---|---|---|
| `--data-path` | no | `data/shandong_pmos_hourly.xlsx` | Input data file |
| `--runs-root` | no | `daily_runs` | Prediction run root (primary scan target) |
| `--prediction-pack` | no | *(see prediction_pack)* | Pre-built prediction pack CSV |
| `--target` | no | `realtime` | `realtime` / `dayahead` / `both` |
| `--start-date` | no | `2025-11-01` | Start of evaluation window |
| `--end-date` | no | `2025-12-31` | End of evaluation window |
| `--out-dir` | no | `reports/local/p0_full_run` | Output directory for reports |
| `--profile` | no | `medium` | Correction profile: `conservative`, `medium`, `aggressive`, or `all` |
| `--profile-config` | no | `config/p0_spike_correction_profiles.yaml` | Profile configuration file (YAML or JSON) |

Explicit overrides (take precedence over profile):
- `--spike-prob-threshold`
- `--max-lift-ratio`
- `--max-absolute-lift`
- `--protect-normal-hours`

## Prediction Pack File Name

```
prediction_pack_realtime_{start_YYYY}_{MM}_{end_YYYY}_{MM}.csv
```

Example for the full P0 window:

```
prediction_pack_realtime_2025_11_2026_02.csv
```

## Path Resolution Order

1. `--prediction-pack` (explicit file, highest priority)
2. `--runs-root/{date}/realtime/model_outputs/{model}/*.csv`
3. `--runs-root/{date}/realtime/real/all_model_forecasts_long.csv`
4. `--runs-root/{date}/realtime/fused/fused_predictions.csv`
5. `--runs-root/{date}/realtime/final/realtime_final_predictions.csv`
6. `--runs-root/{date}/final/realtime_final_predictions.csv`
7. `--runs-root/{date}/final/realtime_final_predictions_corrected.csv`
8. `--runs-root/{date}/compat_fusion/realtime/fused_predictions_corrected.csv`
9. `outputs/{date}/...` (legacy fallback, try same patterns)

## Coverage / Gap Report

Every pack-building run must emit two artifacts:

| File | Content |
|---|---|
| `coverage_report.csv` | Per-date, per-model: files_found, rows_found, date_range |
| `gap_report.csv` | Dates where expected predictions are missing |

## Output Directories

| Directory | Purpose |
|---|---|
| `reports/local/p0_full_run/` | Full evaluation outputs aggregated over the whole window |
| `reports/local/p0_exchange/` | Cross-agent exchange artifacts (manifest, notes) |

## Data Loading Contract

- Prefer `--data-path` when provided
- Fallback: `data/shandong_pmos_hourly.xlsx` then `data/shandong_pmos_hourly.csv`
- Support GBK / UTF‑8 / UTF‑8‑SIG encoding
- Never hardcode local absolute paths

## Profiles

Three tuning profiles defined in `config/p0_spike_correction_profiles.yaml`:

| Parameter | conservative | medium | aggressive |
|---|---|---|---|
| spike_prob_threshold | 0.75 | 0.60 | 0.45 |
| max_lift_ratio | 0.20 | 0.35 | 0.60 |
| max_absolute_lift | 200 | 350 | 600 |
| protect_normal_hours | true | true | true |
| period_9_16_boost | 1.0 | 1.15 | 1.30 |

## Primary Metric

- **realtime overall sMAPE_floor50**: sMAPE(floor=50%) over all hours after correction

## Secondary Metrics

| Metric | Definition |
|---|---|
| `9_16 sMAPE_floor50` | sMAPE(floor=50%) limited to hours 9-16 |
| `high_spike MAE` | Mean Absolute Error on high-spike hours only |
| `high_spike sMAPE_floor50` | sMAPE(floor=50%) on high-spike hours only |
| `severe_underestimate_count` | Count of hours where `y_true - final_pred > 200` |
| `normal_hours_degradation` | `normal_after_smape_floor50 - normal_before_smape_floor50` |
| `false_lift_rate` | Proportion of non-high-spike hours with `final_pred > base_fused_pred AND lift > 0` |

## Data Leakage Rules

1. Prediction D+1: **Cannot use D+1 realtime price**
2. Realtime prediction: **Cannot use D日 14:00之后实时真实电价**
3. Lift quantiles fitted from **historical data only** (no future lookahead)

## Business Time Mapping

- 00:00 natural → hour_business=24 of previous business day

## Correction Pipeline

```
prediction_pack  ──┐
                   ├─► load_and_merge() ──► ResidualLiftCorrector ──► SpikeGuardrail ──► final_pred
risk_predictions ──┘
```

## Output Contract

Each correction run produces:
- `correction_result.csv` — full corrected DataFrame
- `correction_manifest.json` — profile parameters + metrics
- `metrics_summary.json` — all computed metrics with keys defined above

## Manifest

A `tuning_manifest.json` is placed at `reports/local/p0_exchange/tuning_manifest.json`
after threshold tuning. It records:
- Agent / branch name
- Profiles added
- Files changed
- Commands run
- Remaining risks
