# Production Pipeline

端到端电价预测主流程编排。

## 命令用法

### 单日运行

```bash
python main.py 2026-06-25
```

等价于 `python main.py --pipeline full --date 2026-06-25`。

### 时间段运行

```bash
python main.py --start 2026-06-01 --end 2026-06-07
```

从 2026-06-01 到 2026-06-07 每一天都完整执行主流程。

### 强制重跑

```bash
python main.py 2026-06-25 --force
```

忽略缓存，重新执行所有步骤。

## 执行流程

每个日期按以下 5 步顺序执行：

1. **OOF 池生成** — 调用 `rolling_oof.RollingOriginOrchestrator` 生成 rolling-origin out-of-fold 预测池
2. **模型预测** — 对每个 formal 模型调用 `predict_range` 生成当天预测
3. **学习器训练** — 调用 `run_roel_bgew_fallback` 训练 ROEL-BGEW 学习器
4. **融合** — 调用 `apply_learner_to_forecast` 使用学习器权重融合各模型预测
5. **负电价分类器** — 仅 realtime，调用 `classifier_bridge` 修正极端价格

## 输出目录结构

```
outputs/
  2026-06-25/
    run_manifest.json
    logs/pipeline.log
    dayahead/
      01_model_oof/oof_long_table.csv
      02_model_forecasts/{model}/forecast_predictions.csv
      02_model_forecasts/all_model_forecasts_long.csv
      03_learner/{weights.csv, routing_table.csv, ...}
      04_fusion/fused_predictions.csv
    realtime/
      01_model_oof/oof_long_table.csv
      02_model_forecasts/{model}/forecast_predictions.csv
      02_model_forecasts/all_model_forecasts_long.csv
      03_learner/{weights.csv, routing_table.csv, ...}
      04_fusion/fused_predictions.csv
      05_classifier/fused_predictions_corrected.csv
    final/
      dayahead_final_predictions.csv
      realtime_final_predictions.csv
      realtime_final_predictions_corrected.csv
      submission_ready.csv
```

## 缓存与 Resume 机制

每个步骤执行前会检查对应输出文件是否存在且非空：

- 如果 `run_manifest.json` 的 status 为 `complete` 且 final 文件都存在，整个日期直接 SKIP
- 如果某个中间步骤的输出已存在，该步骤 SKIP，从缺失步骤继续（resume）
- `--force` 强制忽略所有缓存

## OOF 学习器接入

学习器使用 ROEL-BGEW (Rolling-Origin Expert Learner with Backward-Gated Expert Weighting)：

- 输入：OOF 长期表（rolling-origin 预测池）
- 候选策略：equal_weight, static_convex, bgew, single_model
- 选择机制：last_block_holdout meta-validation
- 输出：routing_table (每个 task+period 的最优策略) + weights

## 负电价分类器接入

仅对 realtime 融合结果执行。当分类器检测到极端价格事件且融合预测 <=100 时，修正为 -80.0。分类器失败不阻断主流程，manifest 中标记 `"realtime_classifier": "failed"`。

## Formal 模型列表

```python
FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["sgdfnet", "timemixer", "rt916", "timesfm"]
```

## 常见失败与排查

| 现象 | 原因 | 解决 |
|------|------|------|
| OOF pool generation failed | 数据路径错误或模型训练失败 | 检查 `--data-path`，查看 `logs/pipeline.log` |
| No model predictions | 模型权重文件缺失 | 先运行 `python main.py --pipeline train` |
| Learner training failed | OOF 表为空或格式错误 | 检查 `01_model_oof/oof_long_table.csv` |
| Validation errors | 输出行数不对或字段缺失 | 查看 manifest 中的 `validation_errors` |
| Classifier failed | 分类器数据覆盖范围不足 | 检查 `--clf-data` 路径 |

## 旧命令兼容

以下旧命令仍可使用：

```bash
python main.py --pipeline model_stage --date 2026-06-25
python main.py --pipeline learner_stage --date 2026-06-25
python main.py --pipeline fuse_stage --date 2026-06-25
python main.py --pipeline classifier_stage --date 2026-06-25
python main.py --pipeline rolling_oof --oof-start-month 2026-01 --oof-end-month 2026-05
python main.py --pipeline oof_learner --oof-path oof_runs/.../oof_long_table.csv
python main.py --pipeline apply_oof_learner --forecast-path ... --learner-artifact ...
```
