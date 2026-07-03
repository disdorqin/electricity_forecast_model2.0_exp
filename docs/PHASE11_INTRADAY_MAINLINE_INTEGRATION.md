# Phase 11: Intraday Tracker Main Pipeline Integration

**Date:** 2026-07-03
**Phase:** 11
**Repo:** disdorqin/electricity_forecast_model2.0_exp (主仓库)

## 1. 修改文件列表

### 新增文件

| 文件 | 说明 |
|------|------|
| `corrections/__init__.py` | corrections 包初始化 |
| `corrections/intraday_tracker/__init__.py` | 模块初始化 |
| `corrections/intraday_tracker/schema.py` | 输入输出字段定义 + ValidationResult |
| `corrections/intraday_tracker/adapter.py` | load/normalize/validate intraday pack |
| `corrections/intraday_tracker/policy.py` | 主仓库第二层 policy gating |
| `corrections/intraday_tracker/apply.py` | 融合逻辑 (shadow/low_weight/high_weight) |
| `corrections/intraday_tracker/manifest.py` | manifest JSON + markdown report 生成 |
| `config/intraday_tracker.yaml` | 策略配置文件 |
| `scripts/evaluate_intraday_mainline_integration.py` | 主链路集成评估脚本 |
| `tests/test_intraday_mainline_adapter.py` | adapter 测试 (12 tests) |
| `tests/test_intraday_mainline_policy.py` | policy 测试 (12 tests) |
| `tests/test_intraday_mainline_apply.py` | apply 测试 (9 tests) |
| `tests/test_intraday_mainline_manifest.py` | manifest 测试 (8 tests) |
| `docs/PHASE11_INTRADAY_MAINLINE_INTEGRATION.md` | 本文档 |

### 修改文件

| 文件 | 说明 |
|------|------|
| `cli/parser.py` | 新增 --intraday-pack, --intraday-mode, --intraday-config, --cutoff-hour, --prediction-mode |
| `pipelines/production_pipeline.py` | 新增 _step4b_intraday_correction，在 Step 4 (fusion) 和 Step 5 (classifier) 之间接入 |

## 2. 测试结果

```
41 passed in 1.17s
```

- test_intraday_mainline_adapter: 12 passed
- test_intraday_mainline_policy: 12 passed
- test_intraday_mainline_apply: 9 passed
- test_intraday_mainline_manifest: 8 passed

## 3. 接入主 pipeline 的位置

在 `_step4_fusion` (Step 4) 之后、`_step5_classifier` (Step 5) 之前接入。

实际调用顺序：
```
Step 4: Fusion → fused_predictions.csv (y_fused)
Step 4b: IntradayTracker correction (NEW)
Step 5: Classifier (ExtremPriceClf)
Step 6: Final Outputs
```

Step 4b 只在 `target == "realtime"` 时执行。

## 4. CLI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--intraday-pack PATH` | None | 深度支线导出的 intraday correction pack CSV 路径 |
| `--intraday-mode MODE` | shadow | shadow / low_weight / high_weight / off |
| `--intraday-config PATH` | config/intraday_tracker.yaml | 策略配置文件路径 |
| `--cutoff-hour INT` | None | 覆盖 cutoff hour |
| `--prediction-mode MODE` | FULL_DAY | FULL_DAY / INTRADAY |

## 5. shadow / low_weight / high_weight 行为

### shadow mode
- 不改变 `rt_pred` / `y_fused`
- 记录 `intraday_shadow_pred` 和 `intraday_shadow_delta`
- `intraday_applied = False`
- 用于离线评估和试运行

### low_weight mode
- 对 policy 允许的行：`rt_pred_final = (1 - 0.12) * rt_pred + 0.12 * intraday_corrected_pred`
- policy=DISABLED/SHADOW_ONLY 的行不改变
- `intraday_applied = True` (仅对实际融合的行)

### high_weight mode
- 对 policy 允许的行：`rt_pred_final = (1 - 0.22) * rt_pred + 0.22 * intraday_corrected_pred`
- 要求 cutoff >= 14 且 confidence >= 0.55
- 不满足条件的行降级为 LOW_WEIGHT

## 6. FULL_DAY 禁用验证

**已验证。**

测试 `test_full_day_forces_disabled` 和 `test_full_day_disables_correction` 确认：
- `prediction_mode=FULL_DAY` 时所有行 policy=DISABLED, fusion_weight=0
- 即使传入有效 pack，也不应用任何修正
- manifest 记录 `fallback_reason = "prediction_mode=FULL_DAY"`

## 7. safe fallback 验证

**已验证。**

测试 `test_missing_pack_safe_fallback` 确认：
- 空 pack 时所有字段安全填充
- `rt_pred` 不变
- `stats["safe_fallback"] = True`
- `stats["fallback_reason"] = "empty_pack"`

其他 fallback 场景：
- 无 base prediction column → fallback
- pack validation 失败 → fallback + manifest 记录原因
- import error → fallback + warning

## 8. manifest/report 输出样例

输出目录：`outputs/{date}/reports/local/phase11/intraday_mainline/`

| 文件 | 说明 |
|------|------|
| `intraday_mainline_manifest.json` | 完整运行 manifest |
| `intraday_application_report.md` | 人类可读报告 |
| `intraday_rows.csv` | 匹配到的 intraday 行 |
| `final_with_intraday_shadow.csv` | 完整预测表（含 shadow 列） |

manifest 示例：
```json
{
  "intraday_enabled": true,
  "intraday_mode": "shadow",
  "prediction_mode": "INTRADAY",
  "pack_path": "...",
  "pack_rows": 8,
  "matched_rows": 8,
  "applied_rows": 0,
  "shadow_rows": 8,
  "disabled_rows": 0,
  "avg_fusion_weight": 0.12,
  "avg_confidence": 0.6,
  "policy_counts": {"LOW_WEIGHT": 8},
  "guardrail_counts": {},
  "fallback_reason": null,
  "safe_fallback": true
}
```

## 9. 主链路评估结果

评估脚本 `scripts/evaluate_intraday_mainline_integration.py` 已创建。

由于当前没有实际线上 pack 数据（深度支线的 pack 需要由 SGDFNet 实时推理生成），本次未实际跑出集成评估指标。

评估脚本支持：
- 从 base forecast + intraday pack + ground truth 计算指标
- shadow / low_weight / high_weight 三种模式对比
- 按 bucket (normal/spike/negative) 分析
- 按 policy decision 分析
- 自动裁决 (GO / SHADOW_ONLY / NO-GO)

## 10. 是否建议进入 shadow 试运行

**建议进入 shadow 试运行。**

理由：
1. 所有 41 个测试通过，覆盖 FULL_DAY 禁用、safe fallback、shadow/low_weight/high_weight 行为
2. 接入方式最小侵入，不改变现有 pipeline 结构
3. 默认 shadow mode 不影响现有预测
4. 第二层 policy defense 确保不盲信外部 pack
5. 完整的 manifest/report 输出支持审计

建议试运行步骤：
1. 使用 `--intraday-mode shadow --prediction-mode INTRADAY` 运行 1-2 周
2. 收集 shadow 数据，评估 would-have 改善
3. 确认 policy 决策合理后再切换到 low_weight

## 11. 指标真实性声明

所有测试均通过实际 pytest 运行验证。未伪造任何指标。

评估脚本已创建但未实际跑出集成指标（缺少线上 pack 数据），这是预期行为——Phase 11 的目标是集成基础设施，不是跑评估。
