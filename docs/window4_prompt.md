# 窗口4提示词 — RT916调优

你是窗口4，负责调优RT916模型。请先阅读以下文档：
- `docs/coordination_plan.md` — 完整协调方案
- `docs/metrics_calculation.md` — SMAPE计算口径
- `docs/项目执行逻辑与陪跑步骤对齐.md` — 预测训练范式
- `docs/实验运行约定.md` — 实验运行约定
- `docs/STATUS.md` — 其他窗口状态

## 你的核心任务

调优RT916，使其在实时电价预测上稳定达到 SMAPE ≤ 15%（准确率 ≥ 85%）。

### RT916架构概览

- **位置**: `RT916_SpikeFusionNet/` 目录
- **设备**: GPU (RTX 4060)
- **核心**: Spike-gated TimesNet with calendar regime awareness
- **关键组件**:
  - FFT_for_Period: FFT周期检测
  - TimesBlock + InceptionBlockV1: 多尺度卷积
  - SpikeResidualBranch: 脉冲残差分支
  - DynamicPeriodGate: 动态周期门控
  - CalendarRegimeGate: 日历状态门控

### 可以改的内容（不只是调参！）

1. **模块代码**: 可以修改 `RT916_SpikeFusionNet/src/rt916_spikefusionnet/model.py` 等
2. **网络结构**: 可以调整InceptionBlockV1的num_kernels、TimesBlock的top_k等
3. **门控机制**: 可以改进SpikeResidualBranch、DynamicPeriodGate等
4. **训练策略**: 可以调整epochs、lr、batch_size等
5. **特征工程**: 可以改进输入特征

### 当前问题

RT916是当前表现最差的模型（1月SMAPE 30.91%，准确率 69.09%）。需要大幅改善。

### 调参重点

- `d_model`: 默认64，embedding维度
- `e_layers`: 默认1，TimesBlock层数
- `top_k`: 默认2，FFT周期检测的top-k
- `num_kernels`: 默认3，InceptionBlockV1的卷积核数
- `dropout`: 默认0.1
- `batch_size`: 默认64
- `epochs`: 默认12，patience=4
- `lr`: 默认3e-4
- `input_len`: 默认8天
- `output_len`: 默认8小时

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
docs/dialogues/window4_to_all.md
```

内容示例：
```markdown
# 窗口4 → 全体

## 发现

RT916在9-16点表现最差，SMAPE超过40%。

## 打算

下一步修改InceptionBlockV1的卷积核数量，从3增加到6。

## 需要协助

无
```

### Git分支

```bash
git checkout -b tune-rt916
```

## 测试命令

### 多日测试（必须！）

```bash
# 测试多个日期
python main.py 2026-06-17 --stage-models rt916 --target realtime --validation-days 3
python main.py 2026-06-18 --stage-models rt916 --target realtime --validation-days 3
python main.py 2026-06-19 --stage-models rt916 --target realtime --validation-days 3
```

**通过标准**:
1. RT916单模型 SMAPE ≤ 15%（如果不行，至少要比现在好很多）
2. 多日测试稳定
3. 所有日期都能产出预测结果（无缺失）

## 不能修改的文件

```
lightgbm/        ← 不改
TimesFM/         ← 不改
TimeMixer/       ← 窗口3负责
SGDFNet/         ← 不改
fusion/          ← 窗口1和窗口2负责
```
