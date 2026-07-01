# P3.4 TimesFM Diversity Smoke Test Report

**Date:** 2026-06-30
**Branch:** agent/p34-timesfm-diversity-smoke
**Status:** COMPLETE

---

## Objective

Test whether TimesFM predictions, added as a small-weight diversity signal to Phase2 anchored predictions, reduce high-spike-hour prediction errors (sMAPE, severe underestimates, high-spike MAE) on top-10 spike days.

## Method

- **Spike days**: Top 10 spike days identified by Phase2 correction analysis (2025-11-01 through 2026-02-24)
- **Base**: Phase2 anchor_90 fused prediction (0.9 LightGBM + 0.033 each of 3 baselines)
- **TimesFM weights tested**: 0.05, 0.10, 0.15 (fused as `(1-w) * anchor + w * timesfm`)
- **Correction**: Phase2 medium profile (p=0.60, lift=0.35/350, period_9_16_boost=1.15)
- **Configs**: lightgbm_only (raw LightGBM), phase2_anchor, anchor+timesfm @ 0.05/0.10/0.15

## Results

| Config | sMAPE | Severe | High-spike MAE | False Lift | Lift Count |
|-------|:-----:|:------:|:--------------:|:----------:|:----------:|
| lightgbm_only | **20.42** | **32** | **341.66** | 0.134 | 32 |
| phase2_anchor | 21.60 | 32 | 359.40 | 0.138 | 33 |
| + timesfm 0.05 | 21.92 | 32 | 361.72 | 0.138 | 33 |
| + timesfm 0.10 | 22.33 | 36 | 364.02 | 0.142 | 34 |
| + timesfm 0.15 | 22.70 | 38 | 366.26 | 0.146 | 35 |

## Analysis

1. **TimesFM at all weights degrades every metric**: Every TimesFM fusion weight (0.05–0.15) produces strictly worse sMAPE, severe, and high-spike MAE vs lightgbm_only and phase2_anchor.
2. **Monotonic degradation with weight**: As TimesFM weight increases from 0.05 → 0.15, all error metrics worsen monotonically. The lowest weight (0.05) still degrades sMAPE from 21.60 → 21.92 and increases high-spike MAE from 359.40 → 361.72.
3. **No diversity benefit visible**: The hypothesis that TimesFM captures different spike patterns than LightGBM is not supported. TimesFM predictions do not add orthogonal information that helps spike-hour correction.
4. **LightGBM-only is best on spike days**: Notably, raw LightGBM (`lightgbm_only`) without any fusion or correction achieves the best spike-day sMAPE (20.42) and high-spike MAE (341.66), suggesting the Phase2 anchor averaging actually dilutes spike-day performance.

## Limitations

- **10 spike days only (n=240 timestamps)**: This is a smoke test. A full 4-month evaluation may show different patterns, but the monotonic degradation at all weights makes improvement unlikely.
- **Single correction profile**: Only medium profile tested. Results may differ with conservative or aggressive profiles.
- **Single TimesFM source**: Using pretrained TimesFM-1 (Google), not fine-tuned on electricity market data.

## Verdict

**NO — TimesFM does not add useful diversity signal.** All TimesFM fusion weights degrade spike-day metrics vs both lightgbm_only and phase2_anchor. Not worth pursuing full 4-month TimesFM run.

## Recommendation

Do not include TimesFM in the Phase3 ensemble. Focus on LightGBM internal improvements (P3.3 weighting) and rolling fusion (P3.1/P3.2) instead.
