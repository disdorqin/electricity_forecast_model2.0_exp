# Phase 12: End-to-End Intraday Shadow Replay + Propagation Fix

**Date:** 2026-07-03
**Branch:** tune-timemixer
**Status:** Complete

## 1. Phase 11 状态回顾

Phase 11 完成了 Intraday Tracker 在主链路中的集成，包括：

- 新增 `corrections/intraday_tracker/` 模块（schema、adapter、policy、apply、manifest）
- 在 production pipeline 的 Step 4 fusion 后、Step 5 classifier 前插入 Step 4b
- 新增 5 个 CLI 参数（`--intraday-pack`、`--intraday-mode`、`--intraday-config`、`--cutoff-hour`、`--prediction-mode`）
- 41 个测试全部通过，commit `5d7d5f6` 已 push 到 tune-timemixer

Phase 11 遗留的关键问题：adapter 的 `normalize_intraday_pack` 未透传 `n_observed`、`residual_std_today` 等字段，导致 policy gating 将这些字段默认为 0，从而把所有行都判定为 DISABLED。

## 2. Propagation 修复

### P0: Prediction Column Propagation

`apply.py` 在 Phase 11 中只更新 `rt_pred` 列，但 Step 5 classifier 读取的是 `y_fused`。这导致 correction 虽然显示 applied，但实际不会传递到 classifier 和最终输出。

**修复方案：**
- 引入 `PREDICTION_COLUMNS = ["rt_pred", "y_fused", "y_pred"]` 常量
- 在 correction applied 时，同步更新所有存在的预测列
- 新增 `intraday_prediction_column_updated`（bool）和 `intraday_updated_columns`（逗号分隔字符串）字段
- shadow mode 下不更新任何预测列

### P1: Adapter 字段透传

`adapter.py` 的 `normalize_intraday_pack` 只映射了 14 个固定列，丢弃了 `n_observed`、`observed_hours`、`residual_std_today`、`bias_direction` 等 policy gating 所需的关键字段。

**修复方案：**
- 在直接映射列表中增加 7 个 pass-through 字段
- 在 `_default_for` 中为这些字段添加安全默认值
- 在 `policy.py` 中增加 `observed_hours` 作为 `n_observed` 的 fallback

**修复后效果：** policy 正确地将 112 行中的 108 行判定为 LOW_WEIGHT，4 行判定为 SHADOW_ONLY（修复前全部 112 行被错误判定为 DISABLED）。

## 3. 新增测试结果

### test_intraday_mainline_propagation.py（11 个测试）

覆盖以下场景：

1. 输入只有 `y_fused` 时，low_weight 更新 `y_fused`
2. `y_fused` 等于混合计算值
3. 输入只有 `y_pred` 时，low_weight 更新 `y_pred`
4. 输入有 `rt_pred` + `y_fused` 时，low_weight 同步更新两列
5. shadow mode 不更新 `y_fused`
6. shadow mode 不更新 `rt_pred`
7. classifier 读取的 `y_fused` 是修正后的值
8. `rt_pred_before_intraday` 保留原始值
9. `rt_pred_after_intraday` 等于修正后的最终值
10. FULL_DAY 模式下不更新任何列
11. missing pack 时不更新任何列

**全部 52 个测试通过**（原有 41 + 新增 11）。

## 4. y_fused / y_pred / rt_pred 同步更新验证

Smoke test 验证了以下场景：

- **shadow mode:** `y_fused` 不变，shadow_rows > 0，applied_rows == 0
- **low_weight mode:** `y_fused` 改变，4 行 applied，fusion 公式 `(1-w)*base + w*corrected` 验证通过
- **FULL_DAY mode:** `y_fused` 不变，强制禁用
- **missing pack:** `y_fused` 不变，safe fallback

## 5. Smoke Pipeline 结果

```
[PASS] shadow mode   — y_fused unchanged, shadow_rows > 0, applied_rows == 0
[PASS] low_weight    — y_fused changed, 4 rows applied, fusion formula verified
[PASS] FULL_DAY      — FULL_DAY correctly disabled, y_fused unchanged, applied_rows == 0
[PASS] missing pack  — Safe fallback on empty pack, y_fused unchanged

ALL 4 SCENARIOS PASSED
```

## 6. 真实 Phase 10 Pack

使用了深度支线仓库 `disdorqin/electricity_forecast_deep_sgdf_delta` 的 Phase 10 输出：

- **文件：** `reports/local/phase10/intraday_tracker_stability/correction_pack_phase10.csv`
- **规模：** 112 行，覆盖 28 个交易日，cutoff_hour=12，target_hour 覆盖 13-16
- **无重复：** 每个 (business_day, target_hour) 对唯一
- **Policy 分布：** 108 行 LOW_WEIGHT，4 行 SHADOW_ONLY
- **平均 confidence：** 0.675，平均 fusion_weight：** 0.093

## 7. 对齐 Replay

使用 2026 年 2 月数据完成了端到端 shadow replay：

- **Fused predictions：** `synthetic_fused_2026_02.csv`（168 行，28 天 x 6 小时）
- **Ground truth：** `ground_truth_2026_02.csv`（224 行）
- **Intraday pack：** Phase 10 correction_pack_phase10.csv（112 行）

Shadow replay 脚本新增了自动去重功能：当 pack 包含多个 cutoff_hour 的预测时，默认保留最高 cutoff_hour 的行（即观测数据最多的版本）。

## 8. Replay 结果

### Overall sMAPE (floor=50)

| Mode | sMAPE | Samples | Gain vs Baseline (pp) |
|------|-------|---------|-----------------------|
| baseline (no correction) | 0.7061 | 168 | — |
| shadow | 0.7061 | 168 | 0.0000 |
| low_weight | 0.7048 | 168 | +0.1263 |
| high_weight | 0.7048 | 168 | +0.1263 |

### Per-Bucket sMAPE

| Mode | Negative (n=115) | Spike (n=9) | Normal (n=44) |
|------|-------------------|-------------|---------------|
| baseline | 0.6111 | 1.0214 | 0.8897 |
| low_weight | 0.6105 | 1.0191 | 0.8870 |

### Policy Distribution

- LOW_WEIGHT: 108 行（96.4%）
- SHADOW_ONLY: 4 行（3.6%）
- Guardrail: 66 行标记为 negative_price_risk

### 分析

low_weight 和 high_weight 模式产生相同结果，因为 Phase 10 pack 中没有 HIGH_WEIGHT 策略的行（全部为 LOW_WEIGHT 或 SHADOW_ONLY）。correction 在所有 bucket 中均产生了轻微改善，其中 normal bucket 改善最明显（-0.0028）。

注意：本次 replay 使用的 fused predictions 为合成数据（基于 Phase 10 的 base_pred 生成），非真实主链路 fusion 输出。正式指标需在真实主链路运行后确认。

## 9. 是否建议进入真实 Shadow 试运行

**建议：可以进入真实 shadow 试运行。**

理由：

1. Propagation bug 已修复并通过 52 个测试验证
2. Smoke test 4/4 通过，FULL_DAY 禁用和 safe fallback 工作正常
3. Shadow replay 显示 correction 方向正确，所有 bucket 均有轻微改善
4. Policy gating 工作正常，96.4% 的行被允许 low_weight 融合
5. 代码层面的去重和字段透传问题已解决

建议试运行 1-2 周后，根据真实 shadow 数据评估是否切换到 low_weight 模式。

## 10. 指标真实性声明

本报告中的所有指标均通过实际运行脚本产出，未伪造任何数据。需要注意的是，fused predictions 为合成数据而非真实主链路输出，因此 sMAPE 数值的绝对值仅供参考，相对变化趋势（correction 方向）更有参考价值。

---

### 修改文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `corrections/intraday_tracker/adapter.py` | 修改 | 新增 7 个 pass-through 字段 + 默认值 |
| `corrections/intraday_tracker/policy.py` | 修改 | observed_hours 作为 n_observed 的 fallback |
| `scripts/run_intraday_shadow_replay.py` | 修改 | 新增去重逻辑、baseline 计算、gain 指标 |
| `scripts/evaluate_intraday_mainline_integration.py` | 修改 | 新增 --simulate-modes、--no-ground-truth 支持 |
| `tests/test_intraday_mainline_propagation.py` | 新增 | 11 个 propagation 测试 |
| `docs/PHASE12_INTRADAY_SHADOW_REPLAY.md` | 新增 | 本文档 |
