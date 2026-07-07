# P3 Extreme Price Correction — Shadow Experimental Package

> Pushed from efm3.0 (2026-07-07). Part of the **Electricity Price Forecast 3.0** P3 open exploration:
> *Spike / Residual / Classifier — design & shadow-only validation of an Extreme Price Correction System.*
>
> **Shadow-only. Did NOT modify 2.5 (locked) or 3.0 main chain. Did NOT write `submission_ready.csv`.**

## Verdict (see `exports/.../promotion_decision.json`)
- `recommended_status = CANDIDATE` (single month passed all 16 hard criteria; lacks multi-month stability evidence → not promoted to `shadow`)
- `Final Verdict = PASS`

## Result summary (32 real-time days, 2026-01-25 ~ 02-25, D14 cutoff-safe)
| Subset | sMAPE before | sMAPE after | Δ |
|---|---:|---:|---:|
| Overall | 40.88 | 34.22 | -6.66 |
| Negative (n=213) | 78.14 | 53.75 | -24.39 |
| Spike (n=26) | 39.95 | 36.26 | -3.69 |
| Normal (n=529) | 25.93 | 26.26 | +0.33 (negligible) |

Ablation dropped the generic residual corrector (no benefit, slight normal-hour damage). Negative-price correction is the main contributor; spike correction is safe.

## Layout
```
p3_extreme_price_correction/
├── README.md
├── docs/                                   # 4 markdown reports
│   ├── p3_spike_residual_existing_experience.md
│   ├── p3_extreme_price_correction_design.md
│   ├── p3_technical_live_log.md
│   └── p3_spike_residual_final_report.md
├── experimental/p3_extreme_price_correction/   # 6 modules + pipeline + metrics
│   ├── config.py
│   ├── common_metrics.py
│   ├── build_baseline_features.py
│   ├── negative_price_classifier.py
│   ├── spike_price_classifier.py
│   ├── residual_corrector.py
│   ├── correction_guard.py
│   ├── rollback_guard.py
│   ├── pipeline_shadow.py
│   ├── models.py
│   └── walkforward.py
├── scripts/
│   ├── run_p3_spike_residual_shadow.py
│   ├── evaluate_p3_spike_residual.py
│   └── export_p3_spike_residual_candidate.py
└── exports/efm3_candidates/spike_residual/p3_rt_20260125_20260225_v1_cand/
    ├── spike_residual_predictions.csv
    ├── metrics.json
    ├── before_after_report.md
    ├── ablation_report.md
    ├── design_report.md
    ├── manifest.json
    └── promotion_decision.json
```

## How to run (portability note)
The scripts were validated with `ROOT` hardcoded to the original efm3.0 checkout. To run from
this repo, set `ROOT` in each script (`scripts/run_p3_spike_residual_shadow.py`,
`scripts/evaluate_p3_spike_residual.py`, `scripts/export_p3_spike_residual_candidate.py`) to the
absolute path of **this `p3_extreme_price_correction` directory** (one-line edit per script).
This lets `from experimental.p3_extreme_price_correction import ...` resolve and output land under
`p3_extreme_price_correction/outputs/...` (gitignored).

Required input: `outputs/p3_spike_residual/{run_id}/baseline_features.parquet`, rebuildable via
`experimental/p3_extreme_price_correction/build_baseline_features.py` from a realtime ledger
(`outputs/ledger/realtime/{prediction,actual}/*.parquet`).

## Safety guarantees
- D14 cutoff-safe: no D+1 actual / spike / negative label used as online feature.
- No NaN, 24-hour complete, postflight-safe.
- Correction cap + rollback guard present; every correction carries reason / confidence.

## Next steps (not yet done)
1. Extend to ≥3 months of ledger to confirm stability → re-evaluate for `shadow` promotion.
2. Retrain spike classifier with more spike samples (current P=0.118, weak).
