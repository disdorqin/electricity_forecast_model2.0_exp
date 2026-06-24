# 窗口3提示词 — TimeMixer调优

你是窗口3，负责调优TimeMixer模型。请先阅读以下文档：
- `docs/coordination_plan.md` — 完整协调方案
- `docs/metrics_calculation.md` — SMAPE计算口径
- `docs/项目执行逻辑与陪跑步骤对齐.md` — 预测训练范式
- `docs/STATUS.md` — 其他窗口状态

## 你的核心任务

调优TimeMixer，使其在实时电价预测上稳定达到 SMAPE ≤ 15%（准确率 ≥ 85%）。

### TimeMixer架构概览

- **位置**: `TimeMixer/` 目录
- **设备**: GPU (RTX 4060)
- **核心**: Multi-scale decomposable mixing with past/future dual-input
- **两个backbone可选**: TimeMixerBackbone（默认）, TimesNetBackbone

### 可以改的内容（不只是调参！）

1. **模块代码**: 可以修改 `TimeMixer/backbones.py`, `TimeMixer/repro_pipeline.py` 等
2. **网络结构**: 可以调整层数、隐藏维度、注意力机制等
3. **损失函数**: 可以尝试 risk_hour_weighted, risk_peak_weighted 等
4. **校准策略**: 可以调整 segment bias calibration, regime-aware affine 等
5. **超参数**: hidden_dim, blocks, scales, dropout, lr, epochs 等

### 调参重点

- `seq_len`: 默认168（1周），可以尝试更长或更短
- `hidden_dim`: 默认64，可以尝试更大
- `blocks`: 默认2，可以尝试更多层
- `scales`: 默认3，多尺度分解的层数
- `dropout`: 默认0.1
- `lr`: 默认1e-3
- `epochs`: 默认30，patience=15

### 当前问题

TimeMixer在某些日期不产出forecast_predictions.csv，需要检查为什么。

## 沟通协议（必须严格遵守）

### 沟通规则

1. **每次开始新任务前**: 读 `docs/STATUS.md` 和 `docs/dialogues/` 下的最新文件
2. **每次修改文件前**: 读 `docs/STATUS.md` 确认文件没被锁定
3. **开始任务时**: 更新 STATUS.md，状态改为 🟡进行中
4. **完成任务时**: 更新 STATUS.md，状态改为 ✅完成，写明测试结果
5. **有重要发现时**: 在 `docs/dialogues/` 中写对话文件通知其他窗口
6. **需要协助时**: 在 STATUS.md 底部"问题/求助区"留言

### 对话文件格式

```
docs/dialogues/window3_to_all.md
```

内容示例：
```markdown
# 窗口3 → 全体

## 发现

TimeMixer在某些日期forecast阶段失败，原因是xxx。

## 打算

下一步修改repro_pipeline.py中的xxx逻辑。

## 需要协助

窗口2：融合学习器能否处理部分模型缺失的情况？
```

### Git分支

```bash
git checkout -b tune-timemixer
```

## 测试命令

### 多日测试（必须！）

```bash
# 测试多个日期
python main.py 2026-06-17 --stage-models timemixer --target realtime --validation-days 3
python main.py 2026-06-18 --stage-models timemixer --target realtime --validation-days 3
python main.py 2026-06-19 --stage-models timemixer --target realtime --validation-days 3

# 日前测试
python main.py 2026-06-17 --stage-models timemixer --target dayahead --validation-days 3
python main.py 2026-06-18 --stage-models timemixer --target dayahead --validation-days 3
python main.py 2026-06-19 --stage-models timemixer --target dayahead --validation-days 3
```

**通过标准**:
1. TimeMixer单模型 SMAPE ≤ 15%
2. 多日测试稳定
3. 所有日期都能产出预测结果（无缺失）

## 不能修改的文件

```
lightgbm/        ← 不改
TimesFM/         ← 不改
SGDFNet/         ← 不改
RT916_SpikeFusionNet/ ← 窗口4负责
fusion/          ← 窗口1和窗口2负责
```
