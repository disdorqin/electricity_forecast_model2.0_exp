# P4 Final Fusion + Correction Report

**Date:** 2026-07-01
**Branch:** agent/p4-fusion-correction-finalizer
**Status:** PENDING INPUTS

---

## Objective

Combine Window 2 (best LightGBM sample weighting) and Window 3 (best hybrid gate) with Phase2 fusion + correction to achieve DEPLOY GO:

| Metric | Threshold |
|--------|-----------|
| sMAPE | ≤ 20.50 |
| Severe underestimates | ≤ 63 |
| False lift rate | ≤ 10% |
| Normal hours degradation | ≤ 0.50 |

## Combinations

| # | Combo | Description | Status |
|---|-------|-------------|--------|
| 1 | phase2_baseline | Phase2 champion baseline (ref) | ⏳ waiting for inputs |
| 2 | w2_only | W2 best LGBM + Phase2 fusion | ⏳ waiting for W2 |
| 3 | w3_only | Phase2 base + W3 hybrid gate | ⏳ waiting for W3 |
| 4 | w2_plus_w3 | W2 best LGBM + W3 hybrid gate | ⏳ waiting for W2 + W3 |

## Inputs Required

- [ ] Window 1: canonical pack (Phase2 prediction pack)
- [ ] Window 2: best LightGBM candidate CSV
- [ ] Window 3: best hybrid gate CSV
- [ ] Risk predictions (Phase2)

## Results

*Pending — run skeleton with:*

```bash
python scripts/evaluate_p4_final_fusion_correction.py \
    --canonical-pack <phase2_canonical_pack.csv> \
    --window2-csv <w2_best_lightgbm.csv> \
    --window3-csv <w3_best_hybrid_gate.csv> \
    --risk-predictions <risk_predictions.csv> \
    --out-dir reports/local/p4_final_fusion_correction
```

## Verdict

**Pending inputs.**
