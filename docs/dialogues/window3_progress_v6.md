# 窗口3 → 全体

## 时间: 2026-06-23 22:30

## 实验结果汇总

### 6轮实验对比

| 配置 | DA overall | RT overall | RT 1_8 | RT 9_16 | RT 17_24 |
|------|-----------|-----------|--------|---------|----------|
| 基线 (default) | 26.14% | 42.82% | 40.37% | 63.13% | 24.30% |
| v2: +attention | 25.85% | 40.89% | 33.51% | 61.43% | 27.32% |
| v3: residual_blend | 25.14% | 38.76% | 33.04% | 57.92% | 24.90% |
| v4: +risk_peak_weighted | 25.14% | 39.60% | 34.16% | 58.61% | 25.58% |
| v5: TimesNet+seq336 | 24.55% | 37.73% | 31.18% | 57.78% | 23.81% |
| **v6: no-seg+auto-calib** | **23.35%** | **35.61%** | **25.29%** | **54.98%** | 26.36% |

### 最佳配置 (v6)
- Backbone: TimesNet
- seq_len: 336 (2周)
- hidden_dim: 128, blocks: 4, dropout: 0.15
- target_mode: residual_blend
- rt_calibration_mode: rt_916_auto
- segment_training: disabled (单模型预测24h)
- lr: 3e-4, epochs: 40

### 关键发现

1. **residual_blend**模式比direct模式显著提升了RT预测质量（-4% overall）
2. **TimesNet** backbone比TimeMixer backbone略好（-1% overall）
3. **禁用segment training**反而提升了效果（-2% overall），说明单模型学到了更好的泛化表示
4. **auto calibration**自动选择了spike_day_affine，优于固定校准策略
5. **RT 9_16 (solar period)**是最大瓶颈（54.98%），即使最佳校准也无法降至15%以下

### 仍存在的问题

1. RT 9_16 (solar hours) SMAPE仍高达54.98%，远高于15%目标
2. 所有配置的R2在RT 9_16都是负数，说明模型甚至不如均值预测
3. DA 9_16也较差（35.40%），但DA 17_24已达标（14.24%）

### 可能的下一步

1. 深入分析RT 9_16为什么如此难预测（可能是数据本身的问题）
2. 尝试用外部数据（如天气预报的精度提升）
3. 尝试ensemble方法结合多个模型
4. 与窗口1（尖峰识别）和窗口2（融合学习器）协作

## 需要协助

1. 窗口1：尖峰识别模块能否识别solar period的特殊模式？
2. 窗口2：融合学习器能否利用多个模型的互补性来降低RT 9_16的误差？
3. 窗口4：RT916在RT 9_16上表现如何？
