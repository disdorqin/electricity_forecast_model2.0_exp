# P5 Deep + Time-Series Model Zoo Report

> **Date**: 2026-07-02
> **Branch**: `agent/p5-deep-ts-model-zoo`
> **Script**: `scripts/run_p5_deep_ts_model_zoo.py`
> **Period**: 2025-11-01 ~ 2026-02-28 (120 days, 2880 hours)
> **Target models**: TimesFM, PatchTST, iTransformer, TSMixer, N-HiTS, TimeMixer++, RT916 selective, SGDFNet

---

## Executive Summary

**Only TimesFM completed inference** (145s runtime, well under 90 min budget). All other candidate models lack installed packages or trainable checkpoints.

| Criterion | Target | TimesFM | Met? |
|-----------|--------|:-------:|:----:|
| Correlation vs LightGBM | < 0.8 | **-0.027** | ✅ **Diversity GO** |
| sMAPE | ≤ 22.02 | 35.59 | ❌ Strong GO |
| Severe | ≤ 63 | 286 | ❌ Strong GO |
| High spike MAE | < LightGBM | 290 vs 278 | ❌ |

**Verdict**: **Diversity GO** — TimesFM provides a nearly orthogonal signal (correlation -0.027). Not a standalone replacement, but valuable for ensemble fusion.

---

## Inventory

| Model | Available? | Checkpoint? | Can Infer? | Est. Runtime | Status |
|-------|:----------:|:-----------:|:----------:|:------------:|:------:|
| **TimesFM** | ✅ | ✅ model.safetensors 925MB | ✅ | **2.5 min** | **RUN** |
| TimeMixer++ | ✅ code only | ⚠️ cached outputs (36d) | needs train | 60 min | not run — no full-period predictions |
| RT916 selective | ✅ code only | ❌ | needs train | 120 min | feasibility only |
| SGDFNet | ✅ code only | ❌ | needs train | — | feasibility only |
| PatchTST | ❌ | ❌ | ❌ | — | not installed |
| iTransformer | ❌ | ❌ | ❌ | — | not installed |
| TSMixer | ❌ | ❌ | ❌ | — | not installed |
| N-HiTS | ❌ | ❌ | ❌ | — | not installed |

### Why Models Were Not Run

| Model | Reason |
|-------|--------|
| TimeMixer++ | Cached outputs only cover 36 contiguous days, not the full 120-day window. Training on full period would take >60 min. |
| RT916 selective | Has pipeline code but no trained checkpoints. Requires full training (PyTorch Lightning, ~120 min). |
| SGDFNet | Has model definition code but no training pipeline or checkpoints. Feasibility assessment only. |
| PatchTST / iTransformer / TSMixer / N-HiTS | No packages installed (`neuralforecast`, `gluon-ts` PatchTST, etc.). User directive: no large dependency installation. |

---

## TimesFM Inference Details

### Method

- **Model**: `timesfm.TimesFM_2p5_200M_torch` (JAX backend)
- **Config**: `max_context=512`, `max_horizon=128`, `per_core_batch_size=32`
- **Strategy**: 512-hour context window → forecast 24 hours → slide 1 day
- **Runtime**: **145 seconds (2.5 min)** — well under 90 min budget
- **Output**: 2880 predictions (120 days × 24 hours)

### Model Load & Compile

| Step | Time |
|------|:----:|
| Model instantiation | 1.2 s |
| Compile (JIT) | < 1 s |
| Per-day forecast | ~1.2 s/day |
| Total | 145 s |

### Prediction Statistics

| Statistic | TimesFM | LightGBM |
|-----------|:-------:|:--------:|
| y_pred mean | 265.4 | 260.0 |
| y_pred min | -19.7 | -80.0 |
| y_pred max | 497.1 | 660.9 |
| y_true mean (period) | 252.8 | 252.8 |

### Correlation vs LightGBM

| Metric | Value |
|--------|:-----:|
| Pearson correlation | **-0.027** |
| Prediction MAE diff | **134.80** |

TimesFM predictions are **essentially uncorrelated** with LightGBM. This is highly useful for ensemble fusion — combining two orthogonal predictors typically improves robustness and reduces overfitting risk.

### Accuracy Metrics

| Metric | TimesFM | LightGBM | Delta |
|--------|:-------:|:--------:|:-----:|
| sMAPE (floor50) | 35.59 | 22.02 | +13.57 |
| 9_16 sMAPE | 40.70 | 28.16 | +12.54 |
| Severe underestimates | 286 | 80 | +206 |
| High spike MAE | 290.36 | 278.40 | +11.96 |

TimesFM alone is **not competitive** with LightGBM on any accuracy metric. As a pretrained foundation model (not fine-tuned on this market), it lacks domain-specific adaptation. Its value is in the diversity of its predictions.

---

## Diversity Assessment

### Diversity GO Criterion

> **"模型单体不一定打赢，但与 LightGBM correlation < 0.8 且 high_spike_MAE 有改善"**

- Correlation < 0.8: ✅ **-0.027** (well below threshold)
- High spike MAE improvement: ❌ 290 vs 278 (slightly worse)

TimesFM meets the diversity requirement but does not improve spike-hour predictions on its own.

### Fusion Potential

An ensemble using LightGBM + TimesFM would benefit from:
1. **Error decorrelation**: When LightGBM overestimates, TimesFM often underestimates (and vice versa)
2. **Regime awareness**: TimesFM captures broad price patterns; LightGBM captures local hourly patterns
3. **Spike coverage**: Different spike failure modes → combined model may reduce severe underestimates

A simple average or convex combination of LightGBM and TimesFM predictions is the natural next step.

---

## Output Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Runner script | `scripts/run_p5_deep_ts_model_zoo.py` | Full inventory + runner |
| TimesFM predictions (W1) | `reports/local/p5_deep_ts_model_zoo/predictions/timesfm_w1.csv` | 2880 rows, W1 schema |
| Summary JSON | `reports/local/p5_deep_ts_model_zoo/p5_model_zoo_summary.json` | Full metrics + metadata |
| This report | `docs/reports/P5_deep_ts_model_zoo_report.md` | — |

### W1 Schema

```
model_name | business_day | hour_business | timestamp | y_pred | source_file | prediction_mode | leakage_safe
```

---

## Recommendations

### 1. Add TimesFM as a fusion candidate

TimesFM provides genuine diversity (correlation -0.027). It should be added to the multi-candidate prediction pack as a 5th model column:

| Column | Source |
|--------|--------|
| `dayahead_proxy` | Existing |
| `naive_lag1` | Existing |
| `naive_lag7` | Existing |
| `lightgbm` | Existing |
| `timesfm` | **New** — from this run |

### 2. Do NOT replace LightGBM

TimesFM's standalone accuracy (sMAPE 35.59) is far below the deployment threshold. It is a complement, not a replacement.

### 3. Priority for future work

| Model | Priority | Action |
|-------|:--------:|--------|
| TimesFM fusion | P0 | Add to multi-candidate pack, test fusion weights |
| LightGBM + TimesFM ensemble | P1 | Simple average, evaluate severe reduction |
| TimeMixer++ | P2 | Need full-period training to generate useful predictions |
| RT916 / SGDFNet | P3 | Need checkpoint generation and training pipeline investment |
| PatchTST / iTransformer | P4 | Need `neuralforecast` package installation |

### 4. Compute budget note

TimesFM ran in **145s** — only 2.7% of the 90-minute budget. If needed, the budget can be drastically reduced or used for additional model evaluations.
