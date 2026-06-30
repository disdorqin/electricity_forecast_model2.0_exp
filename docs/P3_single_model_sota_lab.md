# P3c — Single-Model SOTA Lab

## Purpose

Identify the 1-2 most promising routes to improve single-model accuracy.
Priority: LightGBM (existing checkpoint + P0 inference pipeline).
Secondary: RT916, SGDFNet, TimeMixer (feasibility diagnosis only — no deep retuning).

## SOTA Lab Directory Structure

Outputs go to `reports/local/p3_sota_lab/` (gitignored):

```
reports/local/p3_sota_lab/
  README.md                     — this file (auto-generated)
  lightgbm/
    experiment_log.md           — tuning experiments run
    hyperparams_tuned.csv       — hyperparameter search results
    feature_importance.md       — feature ablation notes
    cv_results.csv              — cross-validation metrics
  rt916/
    feasibility_checklist.md    — checkpoint / training requirements
  sgdfnet/
    feasibility_checklist.md    — checkpoint / training requirements
  timemixer/
    feasibility_checklist.md    — checkpoint / training requirements
  sota_comparison.csv           — unified single-model ranking table
```

## LightGBM Experiments (Priority)

| Experiment | Status | Notes |
|------------|--------|-------|
| Baseline (ThreeStageLGBM) | Done (Phase 1B) | sMAPE 22.02, severe 80 |
| Hyperparameter tuning | PENDING | Grid search over learning_rate, n_estimators, max_depth |
| Feature ablation | PENDING | Remove low-importance features, test impact |
| Cross-validation | PENDING | Time-series CV to validate stability |
| Quantile calibration | PENDING | Calibrate prediction intervals |

## Deep Model Feasibility (Diagnostic Only)

| Model | Checkpoint | GPU Required | Est. Training | Feasibility |
|-------|-----------|-------------|---------------|-------------|
| RT916 | NOT AVAILABLE | Yes | ~3-4h per model | LOW — no P0 checkpoint |
| SGDFNet | NOT AVAILABLE | Yes | ~15min | MEDIUM — inference script exists |
| TimeMixer | NOT AVAILABLE | Yes | Unknown | LOW — no inference script |
| TimesFM | Pre-trained (legacy TF) | Yes | N/A (zero-shot) | MEDIUM — legacy TF dependency |

## Next Actions

1. Run LightGBM hyperparameter grid search (randomized, ~50 iters)
2. Run LightGBM CV to get stable performance estimate
3. If LightGBM sMAPE < 20.0 with tuning → no deep model needed
4. If LightGBM sMAPE >= 20.0 → evaluate RT916 feasibility
