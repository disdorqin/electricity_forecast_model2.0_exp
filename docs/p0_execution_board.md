# P0 Execution Board

## Status

| Component | Status | Agent | Notes |
|---|---|---|---|
| SA1 — Contract Fix | ✅ Done | SA1 | Field contract unified, business_day/hour_business fix |
| SA2 — Path Compatibility | ✅ Done | SA2 | Unified CLI flags; path resolution from daily_runs/ with outputs/ fallback |
| SA3 — Residual Lift + Guardrail | ✅ Done | SA3 | Core correction pipeline built |
| **SA3 — Threshold Tuning** | **✅ Done** | **SA3** | **Profiles + evaluation scripts complete** |
| P0 Full Run (Offline Eval) | 🟡 Ready for Runner | — | Waiting for prediction pack and Runner launch |
| Spike Correction Pipeline | 🟡 Scripts ready | SA2/SA3 | see scripts/build_realtime_spike_dataset.py, train/predict/evaluate |
| Extreme Diagnostics | ✅ Scripts ready | SA2 | diagnose_extreme_events.py already exists, diagnose_model_regime.py added |

## Target Windows

| Window | start-date | end-date | Priority |
|---|---|---|---|
| Nov–Dec 2025 | `2025-11-01` | `2025-12-31` | P0 (worst perf) |
| Jan–Feb 2026 | `2026-01-01` | `2026-02-28` | P0 |

## Files in scope (allowed to modify)

```
config/p0_spike_correction_profiles.yaml
extreme/realtime_high_spike/residual_lift.py
extreme/realtime_high_spike/guardrail.py
extreme/realtime_high_spike/apply_correction.py
scripts/build_backtest_prediction_pack.py
scripts/diagnose_extreme_events.py
scripts/diagnose_model_regime.py
scripts/build_realtime_spike_dataset.py
scripts/train_realtime_spike_risk.py
scripts/predict_realtime_spike_risk.py
scripts/evaluate_realtime_spike_correction.py
scripts/evaluate_p0_realtime_spike_full.py
tests/test_realtime_spike_guardrail.py
.gitignore
docs/p0_execution_board.md
docs/p0_shared_contract.md
```

## Files NOT to modify

```
production_pipeline.py
validation_tap.py
TimesFM / TimeMixer / RT916 / SGDFNet / LightGBM training entry points
Negative-price module
```

## Script checklist

- [x] `build_backtest_prediction_pack.py` — prediction pack builder
- [x] `diagnose_extreme_events.py` — extreme event diagnosis (existed, updated with --runs-root)
- [x] `diagnose_model_regime.py` — regime detection
- [x] `build_realtime_spike_dataset.py` — spike training dataset builder
- [x] `train_realtime_spike_risk.py` — spike risk model trainer
- [x] `predict_realtime_spike_risk.py` — spike risk predictor
- [x] `evaluate_realtime_spike_correction.py` — correction evaluator (with --profile support)
- [x] `evaluate_p0_realtime_spike_full.py` — P0 full-window evaluator (with --profile support)

## Threshold Tuning

**Branch**: `agent/p0-threshold-tuning`

**Profiles** (config/p0_spike_correction_profiles.yaml):
| Profile | spike_prob_threshold | max_lift_ratio | max_absolute_lift | period_9_16_boost |
|---|---|---|---|---|
| conservative | 0.75 | 0.20 | 200 | 1.0 |
| medium | 0.60 | 0.35 | 350 | 1.15 |
| aggressive | 0.45 | 0.60 | 600 | 1.30 |

**Features**:
- `--profile {conservative,medium,aggressive,all}` CLI
- `--profile-config` for external config
- Explicit overrides (`--spike-prob-threshold`, etc.) take precedence
- Profile metadata in `correction_manifest.json`
- `false_lift_rate` and `normal_hours_degradation` metrics

**Run commands**:
```bash
python scripts/evaluate_realtime_spike_correction.py \
    --prediction-pack <path> --risk-predictions <path> \
    --profile all
```

## Output directories

```
reports/local/p0_full_run/              ← aggregated evaluation outputs
reports/local/p0_exchange/              ← cross-agent exchange artifacts
reports/local/p0_tuning/                ← per-profile tuning outputs
  ├── conservative/correction_manifest.json
  ├── medium/correction_manifest.json
  └── aggressive/correction_manifest.json
```

## CLI contract

All scripts accept at minimum:

```
--data-path, --runs-root, --prediction-pack, --target, --start-date, --end-date, --out-dir
```

Spike correction scripts additionally accept:

```
--profile, --profile-config, --spike-prob-threshold, --max-lift-ratio, --max-absolute-lift, --protect-normal-hours
```
