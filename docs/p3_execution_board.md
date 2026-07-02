# P3–P5R Execution Board

> **Purpose**: Track all tasks across P3 (complete), P4 (complete), P5 (superseded), and P5R refocused sprint (active).
> **Status**: `[P3a–P5 COMPLETE (all NO-GO) — P5R REFOCUSED SPRINT ACTIVE]`
> **Branch**: `tune-timemixer`
> **Deployment champion**: Phase 2 lightgbm_anchor_90 + medium + normal (sMAPE=20.8675, severe=63)
> **PR #10**: ✅ MERGED — P3 tooling/leakage-fix/rolling framework
> **PR #11**: OPEN — P3.1 severe-aware rolling (NO-GO, low-priority tooling merge)
> **PR #12**: ✅ MERGED — LightGBM internal spike-weighting tooling
> **PR #13**: OPEN — P3.3 spike-gated uplift (research only, NOT deploy candidate)
> **PR #14**: ✅ MERGED — P4 canonical evaluation pack
> **PR #15**: OPEN — P4 LightGBM SOTA tuning (research-only/deferred, NOT deploy candidate)
> **PR #16**: ✅ MERGED — P5 model-zoo unified dataset builder

---

## Strategic Note

**LightGBM and TimesFM are no longer mainline candidates in this repository.**
Per P5R refocus decision, these models will be replaced/upgraded in other repos.
This repo now focuses on:
- **Existing model tuning**: TimeMixer / RT916 / SGDFNet
- **Monitoring module**: prediction health, data drift, coverage, metric ledger, alerts
- **Negative price + residual module**: negative price, low valley, residual correction

---

## Roles

| Window | Role | Scope | Status |
|--------|------|-------|--------|
| **W0** | Runner / 总控 | Board, PR, adjudication, convergence | `ACTIVE` |
| **W1** | Existing Model Tuning | TimeMixer / RT916 / SGDFNet — tune on canonical pack | `ACTIVE` |
| **W2** | Monitoring Module | Coverage, missing models, feature/prediction drift, sMAPE, severe, spike/negative-price miss, alerts | `ACTIVE` |
| **W3** | Negative Price + Residual | Negative price, low valley, residual correction (must not hurt high-price spike module) | `ACTIVE` |
| **W4** | Integration Finalizer | Integrate W1/W2/W3 into unified offline pipeline | `ACTIVE` |

---

## P5R — Refocused Sprint

> **Goal**: Execute refocused model tuning (non-LightGBM, non-TimesFM), build monitoring module, add negative price + residual correction, and integrate into unified offline pipeline.
> **Champion baseline**: Phase 2 lightgbm_anchor_90 + medium + normal — sMAPE=20.8675, severe=63, false_lift≤10%.

### Model Entry Criteria

Any single model must meet **sMAPE ≤ 22.50 OR severe ≤ 63** to enter W4 fusion validation.

### Fusion DEPLOY GO

| Criteria | Threshold |
|----------|-----------|
| sMAPE | ≤ 20.50 |
| severe | ≤ 63 |
| false_lift | ≤ 10% |
| normal_degradation | ≤ 0.5 |

### Monitoring Targets

Must produce daily outputs: coverage, missing models, feature drift, prediction drift, sMAPE (when y_true available), severe_underestimate, high_spike miss, negative_price miss, alerts.

### Negative Price / Residual Targets

- Must not degrade high-price spike module performance.
- Negative/low-price interval MAE or sMAPE must improve.
- Normal hours degradation ≤ 0.5.

### Results Log

| Date | Window | Result | Verdict |
|------|--------|--------|---------|
| 2026-07-02 | **W2** | **Monitoring Module**: `scripts/monitor_prediction_health.py` — 12 checks (coverage, missing models, duplicates, feature missing, prediction drift, feature drift, sMAPE/MAE, severe, high-spike, negative-price, abnormal lift, alert level). 21/21 tests pass. Outputs: `daily_health.json`, `daily_health.md`, `alerts.json`. | ✅ COMPLETE — 21 tests pass |
| 2026-07-02 | **W3** | **Negative Price + Residual Module**: `extreme/negative_price/` (8 files) — schema, labels, features, risk_model, residual_correction, guardrail, apply_negative_correction. `scripts/evaluate_p5_negative_residual_module.py`. 26/26 tests pass. Mutual exclusion with high_spike guaranteed. | ✅ COMPLETE — 26 tests pass |

---

## Blockers

| ID | Blocker | Status |
|----|---------|--------|
| B20 | Rolling fusion severe exceedance | INACTIVE |
| B21 | P3 NO-GO — no rolling approach beats Phase2 | CLOSED |
| B22 | P4 NO-GO — W4 baseline mismatch invalidated results | CLOSED |
| B23 | LightGBM / TimesFM moved out of main repo scope | RESOLVED — P5R strategic shift |
| B24 | **No approach beats Phase 2 on sMAPE + severe simultaneously** | **OPEN — P5R objective** |

## Next Actions

1. ✅ PR #10 merged (P3 rolling framework)
2. ✅ PR #12 merged (LightGBM weighting tooling)
3. ✅ PR #14 merged (P4 canonical eval pack)
4. ✅ PR #16 merged (P5 model-zoo dataset)
5. ⏳ PR #15: research-only/deferred — NOT deploy candidate
6. ⏳ PR #13: spike-gated uplift — deferred, research only
7. ⏳ PR #11: P3.1 severe-aware rolling — low-priority tooling
8. 🏃 **P5R W1**: Tune TimeMixer / RT916 / SGDFNet on canonical pack
9. 🏃 **P5R W2**: Build monitoring module
10. ✅ **P5R W3**: Build negative price + residual correction — COMPLETE
11. 🏃 **P5R W4**: Integrate once W1/W2/W3 are ready
12. 🎯 **First P5R fusion candidate meeting DEPLOY GO** → new champion
13. 🎯 **Fallback**: Deploy Phase 2 as production
