# R3D-Tap-GEF: Rolling 3-Day Validation Tap with Gated Expert Fusion

## 概述

R3D-Tap-GEF 是电力现货价格预测系统的 production 融合方案，取代旧的 rolling-origin OOF 月度池方案。核心思路：用最近 30 天的 10 个三日验证窗口（validation tap）评估模型表现，通过门控专家融合（Gated Expert Fusion）动态学习各模型在不同时段的最优权重，再经约束凸优化（Weighted Convex Refit）微调，最终融合预测日各模型输出。

每日运行命令：

```bash
python main.py 2026-02-01
python main.py 2026-02-01 --force   # 强制重跑
```

## 1. 为什么不默认 Rolling-Origin OOF

旧方案（rolling-origin OOF 月度池）的瓶颈：

- 生成 OOF 池极慢：5 个月度 fold × 2 个 target × 多个模型 ≈ 14+ 小时（TimeMixer 单 fold 80 分钟）。
- 月度粒度粗：一个月内模型表现可能变化很大，月度平均权重无法捕捉近期趋势。
- 复用粒度不灵活：OOF 池按月共享，月中新增数据无法即时反映。
- 与 production 解耦：OOF 池是独立预计算步骤，容易与实际预测日脱节。

R3D-Tap-GEF 的优势：

- 验证窗口紧贴预测日（D-30 ~ D-1），权重反映最近 30 天模型表现。
- 三日粒度比月度细 10 倍，能更灵敏地捕捉模型退化或数据漂移。
- 每次运行独立计算验证 tap，不依赖预生成的 OOF 池。
- 总计算量：10 个三日 fold vs 5 个月度 fold，TimeMixer 从 ~80min/fold 降到 ~15min/fold。

## 2. 为什么选择 6 个月训练窗口

6 个月是山东省电力现货市场的经验平衡点：

- 太短（<3 个月）：模型无法学到完整的季节性模式（如光伏出力曲线、供暖负荷）。
- 太长（>12 个月）：山东电力市场结构变化快（新能源装机增长、政策调整），过远数据引入噪声。
- 6 个月覆盖至少一个完整的季节过渡（如夏→冬或冬→夏），对负荷和新能源出力的主要周期模式足够。

每个 fold 的训练窗口独立：`train_start = train_end - 6 months`，严格 chronological，不做随机切分。

## 3. 为什么用 10 个三日 Tap Fold

设计参数：`tap_folds = 10`，`tap_block_days = 3`，`validation_days = 30`。

10 × 3 = 30 天，恰好覆盖一个月的验证窗口。选择三日而非单日的原因：

- 单日验证方差太大，某天的极端天气或数据异常会严重扭曲权重。
- 三日块（block）能平滑日间波动，同时保持足够的时间分辨率。
- 三日块与电力市场的"工作日-周末"周期部分对齐（虽然不完美，但比单日好）。

Fold 生成规则（预测日 D）：

```
Fold 0: train to D-31 → predict D-30 ~ D-28  (age_block=9, 最远)
Fold 1: train to D-28 → predict D-27 ~ D-25  (age_block=8)
...
Fold 9: train to D-4  → predict D-3  ~ D-1   (age_block=0, 最近)
```

每条预测标注 `horizon_day ∈ {1, 2, 3}`（块内第几天），用于 horizon gate。

## 4. TimesFM 的特殊处理

TimesFM（Google TimesFM-2.5-200m）是预训练基础模型，不针对特定市场微调。

在 validation tap 中：

- 不训练。使用 `train_end` 之前的历史上下文做 cutoff-safe 推理，预测 `test_start ~ test_end`。
- 保证数据隔离：TimesFM 看不到 `train_end` 之后的任何信息。

在 real forecast 中：

- 使用 `D-1` 之前的历史上下文，直接预测 D 日 24 小时。

虽然 TimesFM 不训练，但其预测结果必须进入 validation tap，学习器根据 TimesFM 在验证集上的表现更新其融合权重。这确保 TimesFM 在不同时段的贡献被动态评估，而非固定为等权。

## 5. Gate 和 Weighted Convex Refit 公式

### 5.1 Gate 设计

每条样本的 sample gate 决定其在权重更新中的影响力：

```
sample_gate = recency_gate × horizon_gate
```

其中：

```
recency_gate = exp(-age_block / tau_block)      # tau_block = 3.0
horizon_gate = exp(-(horizon_day - 1) / tau_horizon)  # tau_horizon = 2.0
```

- `recency_gate`：age_block=0（最近 fold 9）时 gate=1，age_block=9（最远 fold 0）时 gate ≈ 0.05。
- `horizon_gate`：horizon_day=1（块内第一天）时 gate=1，horizon_day=3 时 gate ≈ 0.37。

### 5.2 BGEW 更新

从最近 fold 到最远 fold 依次更新（Fold 9 → Fold 0）：

1. 计算每个模型在 `task + period + tap_fold_id` 上的加权 sMAPE floor50 loss：
   ```
   loss_m = weighted_sMAPE_floor50(y_true, y_pred_m, sample_gate)
   ```

2. 归一化（除以同组模型的 median loss），clip 到 [0.25, 4.0]：
   ```
   normalized_loss_m = clip(loss_m / median_loss, 0.25, 4.0)
   ```

3. 指数衰减更新：
   ```
   w_m = w_m * exp(-eta * recency_gate * normalized_loss_m)
   w_m = max(w_m, weight_floor)
   w_m = w_m / sum(w_m)
   ```

默认参数：`eta = 0.8`，`weight_floor = 0.03`。

### 5.3 Weighted Convex Refit

BGEW 给出初始权重 `w_bgew` 后，做一次约束凸优化微调：

```
min_w  Σ_i sample_gate_i * Loss(y_i, Σ_m w_m * pred_{m,i})
       + λ * ||w - w_bgew||²

约束:
  w_m >= weight_floor  (∀m)
  Σ w_m = 1
```

其中 `Loss = 0.7 * sMAPE_floor50 + 0.3 * normalized_MAE`，`λ = 0.05`。

优化器：scipy SLSQP。如果优化成功用 refit 权重（`weight_source = convex_refit`），失败则 fallback 到 BGEW 权重（`weight_source = bgew_fallback`）。

正则项 `λ * ||w - w_bgew||²` 防止 refit 权重偏离 BGEW 太远，在验证集较小时提供稳定性。

## 6. 输出目录结构

```
outputs/{date}/
  run_manifest.json          # 运行清单，记录每步状态
  logs/
    pipeline.log             # 完整日志

  dayahead/
    validation/              # 10 fold 验证结果
      validation_tap_long_table.csv
      tap_manifest.json
      folds/
        fold_00/
          lightgbm_predictions.csv
          timesfm_predictions.csv
          timemixer_predictions.csv
        ...
    real/                    # 预测日各模型输出
      lightgbm/
        forecast_predictions.csv
      timesfm/
        forecast_predictions.csv
      timemixer/
        forecast_predictions.csv
      all_model_forecasts_long.csv
    fused/                   # 融合结果
      weights.csv
      routing_table.csv
      dynamic_weight_trace.csv
      candidate_metrics.csv
      coverage_report.csv
      fused_predictions.csv
      fused_debug.csv
    final/
      dayahead_final_predictions.csv

  realtime/
    validation/              # 同 dayahead/validation 结构
    real/                    # 同 dayahead/real 结构（多 sgdfnet, rt916）
    fused/                   # 同 dayahead/fused 结构
    final/
      realtime_final_predictions.csv
      realtime_final_predictions_corrected.csv   # 分类器修正后
      classifier_report.json

  final/                     # 顶层汇总
    dayahead_final_predictions.csv
    realtime_final_predictions.csv
    realtime_final_predictions_corrected.csv
    submission_ready.csv
```

### 缓存行为

- `outputs/{date}` 已存在且无 `--force`：打印 "该日期已经预测过。如需重跑，请使用 --force。" 并退出。
- 有 `--force`：删除整个 `outputs/{date}` 目录并重新执行。
- 单步级缓存：如果某步的中间文件已存在且非空，跳过该步（除非 `--force`）。

## 7. 甲方每日运行

### 日常运行

```bash
# 预测明天（推荐 D-1 15:00 后运行）
python main.py 2026-06-26

# 检查输出
cat outputs/2026-06-26/run_manifest.json
```

### 批量回测

```bash
# 预测整个二月
python main.py 2026.2.1-2026.2.28

# 强制重跑某天
python main.py 2026-02-15 --force
```

### 调参（高级）

```bash
python main.py 2026-02-01 \
  --tau-block 2.0 \
  --tau-horizon 1.5 \
  --eta 1.0 \
  --weight-floor 0.05 \
  --lambda-refit 0.1 \
  --force
```

### 输出解读

- `weights.csv`：每个 task+period 组合的模型权重。`weight_source` 列标注是 `convex_refit`（优化成功）还是 `bgew_fallback`。
- `dynamic_weight_trace.csv`：每个 fold 更新后各模型权重的变化过程，用于诊断。
- `fused_predictions.csv`：融合后的逐小时预测，含 `available_models` 和 `weight_summary` 列。
- `submission_ready.csv`：DA+RT 合并的提交格式。

### 验收检查

运行后可通过以下标准快速验收：

1. `validation_tap_long_table.csv` 覆盖 D-30 ~ D-1
2. `tap_fold_id` 共 10 个值（0~9）
3. 每个 fold 覆盖 3 天
4. `horizon_day` 仅为 1, 2, 3
5. `age_block` 为 0~9
6. final 文件 24 行（24 个时段）
7. weights 每个 task+period 组合 sum ≈ 1
