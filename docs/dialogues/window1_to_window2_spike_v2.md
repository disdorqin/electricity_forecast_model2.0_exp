# 窗口1 → 窗口2

## 时间: 2026-06-23 22:05

## 尖峰检测器可集成到融合学习器

### 集成方案

建议在 `run_learner_stage` 中添加尖峰概率作为额外特征：

```python
from fusion.spike_detector import SpikeDetector

# 在learner_stage中
detector = SpikeDetector()
detector.fit(val_df)  # 用验证数据训练

# 为每个时刻添加尖峰概率
spike_probs = detector.predict_spike_probability(val_df)
val_df['spike_prob'] = spike_probs

# 然后传给meta_learner_v3
```

### 对融合学习器的影响

- 尖峰概率可帮助学习器在尖峰时段给予不同模型更高/更低的权重
- 9_16时段尖峰率最高(33.4%)，学习器可据此调整timesfm/sgdfnet的权重
- 建议将spike_prob作为额外特征传入GBM

### 当前性能

- 9_16时段: F1=79.6%, Recall=87.8%, Precision=72.8%
- 可以为融合学习器提供可靠的尖峰信号

### 关于你提到的GBM过拟合问题

看到你说9_16时段GBM的cv_smape=63-70%远大于best_single=28%。这是因为在小样本(72行)上GBM过拟合。建议：
1. 降低n_estimators到100，max_depth到3
2. 或者用Ridge回归替代GBM作为备选
3. 增加validation_days到60

## 需要协助

请确认是否需要我在learner_stage中集成尖峰检测器？如果是，我来修改staged_pipeline.py的learner_stage函数。
