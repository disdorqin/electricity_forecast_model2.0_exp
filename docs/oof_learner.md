# OOF Learner: ROEL-BGEW

## Rolling-Origin Expert Learner with Backward-Gated Expert Weighting

The OOF learner is an independent fusion system that trains on rolling-origin out-of-fold (OOF) prediction pools to learn optimal model combination strategies for electricity price forecasting.

## Why OOF Learner Instead of Traditional Validation?

Traditional fusion approaches use a fixed 30-day validation window to learn combination weights. This has several limitations:

1. **Limited data**: Only 30 days of validation data may not capture seasonal patterns or regime changes
2. **Overfitting risk**: Optimizing on a small window can lead to overfitting to recent noise
3. **No temporal diversity**: All validation samples come from the same time period

The OOF learner addresses these by using a rolling-origin approach where:
- Multiple monthly folds generate out-of-fold predictions (e.g., train on Jan-Jul, predict Aug; train on Jan-Aug, predict Sep; etc.)
- The learner sees predictions across multiple months, capturing more diverse market conditions
- Each OOF prediction is truly out-of-sample (no data leakage from the prediction period)

## Why Rolling-Origin OOF?

Rolling-origin OOF provides a clean evaluation framework:

1. **No lookahead bias**: Each fold's predictions are generated using only data available at that time
2. **Temporal coverage**: OOF predictions span multiple months, providing diverse training data
3. **Realistic evaluation**: Mimics the actual deployment scenario where models predict future dates

## Why Default to Non-Negative Simplex Weights?

The default fusion strategies (static_convex, bgew) constrain weights to:
- **Non-negative**: w_m >= 0 (no short-selling of models)
- **Sum to 1**: Σw_m = 1 (proper convex combination)

Rationale:
- Negative weights are unstable and can amplify errors
- They often indicate overfitting to noise in the validation period
- Non-negative constraints produce more robust, interpretable combinations
- Signed weights are available as an experimental mode (`static_mild_signed`) for ablation studies

## Why Allow One-Hot (Single Model) Selection?

The learner can select a single model with weight 1.0 and all others at 0.0. This is not a fallback—it's a legitimate learned strategy.

Rationale:
- Sometimes one model consistently outperforms the ensemble in a specific (task, period) segment
- Forcing fusion when one model is clearly superior adds noise
- The candidate selector evaluates both fusion and single-model options, choosing the best
- This produces more interpretable results: "In DA 1_8, TimeMixer is best; in RT 9_16, fuse all models"

## Best-Expert Fallback

The `roel_bgew_fallback` mode (default) implements a comprehensive strategy selection:

1. **Candidate generation**: For each (task, period), generate multiple candidate strategies:
   - Equal weight (baseline)
   - Static convex (optimized non-negative weights)
   - BGEW (time-weighted multiplicative update)
   - Each single model (one-hot)

2. **Meta-validation**: Use last-block holdout to avoid overfitting:
   - Fit candidates on all months except the last
   - Evaluate candidates on the last month
   - Select the best candidate

3. **Final refit**: Refit the selected candidate on the full OOF data for final weights

4. **Routing table**: Output a routing table specifying which strategy to use for each (task, period)

## BGEW: Backward-Gated Expert Weighting

BGEW is a time-aware multiplicative weights update algorithm:

### Algorithm

For each (task, period):
1. Initialize weights: w_m = 1/M for all M models
2. Sort OOF samples by target_day (oldest to newest)
3. For each day t:
   - Compute loss per model: L_m(t)
   - Normalize: L'_m(t) = L_m(t) / median_m(L_m(t))
   - Clip: L'_m(t) = clip(L'_m(t), 0, loss_clip)
   - Compute gate: g(t) = exp(-age(t) / τ), where age = latest_day - t
   - Update: w_m ← w_m × exp(-η × g(t) × L'_m(t))
   - Normalize: w_m ← w_m / Σw_m

### Parameters

- **τ (tau)**: Time constant for gate decay (default 30 days)
  - Larger τ → more uniform weighting across time
  - Smaller τ → stronger emphasis on recent data
- **η (eta)**: Learning rate (default 0.5)
  - Larger η → faster weight adaptation
  - Smaller η → more stable, slower adaptation
- **loss_clip**: Clip normalized loss to avoid outlier explosion (default 5.0)

### Intuition

The gate g(t) gives higher weight to recent samples:
- Recent days (age ≈ 0): g ≈ 1.0 (full weight)
- Old days (age >> τ): g ≈ 0.0 (minimal weight)

This allows the learner to adapt to recent market conditions while still learning from historical patterns.

## Meta-Validation: Avoiding Learner Overfitting

Even though OOF predictions are clean (no data leakage), the learner can still overfit if it:
- Trains on all OOF data
- Selects the best strategy on the same OOF data

**Solution**: Last-block holdout
- Split OOF data into fit months (e.g., Aug-Nov) and eval month (Dec)
- Train candidates on fit months
- Evaluate candidates on eval month
- Select best candidate
- Refit selected candidate on full OOF data (Aug-Dec)

If OOF spans only 1 month, skip holdout and warn that selection is in-sample.

## Usage

### Train the Learner

```bash
python main.py --pipeline oof_learner \
  --oof-path oof_runs/my_pool/oof_long_table.csv \
  --learner-mode roel_bgew_fallback \
  --metric sMAPE_floor50 \
  --tau 30 \
  --eta 0.5 \
  --coverage-threshold 0.95 \
  --output-root learner_runs/my_pool
```

### Apply to Final Forecast

```bash
python main.py --pipeline apply_oof_learner \
  --forecast-path oof_runs/my_pool/escort/escort_2026-06-25_long.csv \
  --learner-artifact learner_runs/my_pool/learner_manifest.json \
  --output-root learner_runs/my_pool/final_2026-06-25
```

### Combined (Train + Apply)

```bash
python main.py --pipeline oof_learner \
  --oof-path oof_runs/my_pool/oof_long_table.csv \
  --forecast-path oof_runs/my_pool/escort/escort_2026-06-25_long.csv \
  --learner-mode roel_bgew_fallback \
  --output-root learner_runs/my_pool
```

## Output Files

After training, the output directory contains:

| File | Description |
|---|---|
| `learner_manifest.json` | Metadata: pool_id, learner_mode, parameters, output paths, warnings |
| `weights.csv` | Final weights per (task, period, model) |
| `routing_table.csv` | Selected strategy per (task, period) |
| `candidate_metrics.csv` | Performance of all candidates per (task, period) |
| `coverage_report.csv` | Model coverage per (task, period) |
| `dynamic_weight_trace.csv` | BGEW weight evolution over time (if BGEW used) |
| `oof_backtest_predictions.csv` | OOF predictions using selected strategies |
| `oof_backtest_metrics.csv` | Metrics on OOF backtest predictions |

If `--forecast-path` is provided, also outputs:

| File | Description |
|---|---|
| `final_fused_predictions.csv` | Final fused predictions for the forecast day |

## Learner Modes

### equal_weight
Simple baseline: all eligible models get equal weight (1/M).

### static_convex
Optimizes non-negative simplex weights (w ≥ 0, Σw = 1) using SLSQP to minimize sMAPE_floor50 or MAE.

### static_mild_signed (experimental)
Allows slight negative weights (-0.1 ≤ w ≤ 1.1) for ablation studies. Not recommended for production.

### bgew
Backward-Gated Expert Weighting: time-aware multiplicative weights update with recency bias.

### roel_bgew_fallback (default, recommended)
Compares all candidates (equal_weight, static_convex, bgew, each single model) and selects the best per (task, period) using meta-validation. Outputs routing table with selected strategy.

## Coverage Handling

The learner checks model coverage per (task, period):
- **Coverage threshold** (default 0.95): Models below this threshold are excluded from learning
- **Coverage report**: Shows expected vs available rows, coverage ratio, and eligibility for each (task, period, model)
- **Missing models at apply time**:
  - If selected_mode is single_model and the model is missing → fallback to equal_weight or static_convex
  - If selected_mode is fusion and some models are missing → renormalize weights on available models
  - If no models are available → error

## Metrics

The learner evaluates candidates using:
- **sMAPE_floor50** (default): Symmetric MAPE with floor-50 clipping (prevents explosion on near-zero prices)
- **MAE**: Mean Absolute Error

Additional metrics reported:
- RMSE
- bias_mean (mean prediction error)
- bias_median
- q90_high_price_MAE (MAE on top 10% price samples)
- q95_high_price_MAE (MAE on top 5% price samples)

## Testing

Run unit tests:

```bash
python -m pytest tests/test_oof_learner.py -v
```

Tests use synthetic data and verify:
1. OOF table normalization
2. Coverage report generation
3. Static convex weight constraints
4. BGEW weight constraints
5. BGEW time gating
6. BGEW loss-based weight updates
7. Candidate selection (one-hot)
8. Candidate selection (fusion)
9. Apply learner with missing model fallback
10. Apply learner output shape (24 rows per target_day)
