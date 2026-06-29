# P0 Shared Contract — Path Compatibility

## Purpose

Unify CLI interface across all P0 offline-evaluation scripts so the
Runner can execute a full P0 pipeline without editing individual
scripts.

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
2. `{runs_root}/{date}/realtime/model_outputs/{model}/*.csv`
3. `{runs_root}/{date}/realtime/real/all_model_forecasts_long.csv`
4. `{runs_root}/{date}/realtime/fused/fused_predictions.csv`
5. `{runs_root}/{date}/realtime/final/realtime_final_predictions.csv`
6. `{runs_root}/{date}/final/realtime_final_predictions.csv`
7. `{runs_root}/{date}/final/realtime_final_predictions_corrected.csv`
8. `{runs_root}/{date}/compat_fusion/realtime/fused_predictions_corrected.csv`
9. `outputs/{date}/...` (legacy fallback, try same patterns)

## Coverage / Gap Report

Every pack-building run must emit:

| File | Content |
|---|---|
| `coverage_report.csv` | Per-date, per-model: files_found, rows_found |
| `gap_report.csv` | Dates where expected predictions are missing |

## Output Directories

| Directory | Purpose |
|---|---|
| `reports/local/p0_full_run/` | Full evaluation outputs |
| `reports/local/p0_exchange/` | Cross-agent exchange artifacts (manifest, notes) |

## Data Loading Contract

- Prefer `--data-path` when provided
- Fallback: `data/shandong_pmos_hourly.xlsx` then `data/shandong_pmos_hourly.csv`
- Support GBK / UTF-8 / UTF-8-SIG encoding
- Never hardcode local absolute paths
