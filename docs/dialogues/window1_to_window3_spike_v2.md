# 窗口1 → 窗口3

## 时间: 2026-06-23 22:00

## 尖峰检测器改进完成

### 接口格式

```python
from fusion.spike_detector import SpikeDetector, SpikeDetectorConfig

# 初始化
config = SpikeDetectorConfig()
detector = SpikeDetector(config)

# 训练（用历史数据）
detector.fit(history_df)

# 预测尖峰概率
probs = detector.predict_spike_probability(df)  # -> pd.Series, 每个时刻的概率

# 预测尖峰标签（带momentum）
labels = detector.predict(df, threshold=0.35, use_momentum=True)  # -> pd.Series, 0/1

# 单独使用momentum
probs_mom = detector.apply_momentum(probs)
```

### 输入格式

`df` 必须包含以下列：
- `时刻`: 时间戳
- `日前电价`: DA价格
- `实时电价`: RT价格（可选，有则更好）
- `直调负荷预测值`, `新能源总加预测值`, `竞价空间预测值`（可选，有则更好）

### 输出格式

- `predict_spike_probability(df)`: 返回 `pd.Series`，值域[0,1]，每个时刻的尖峰概率
- `predict(df)`: 返回 `pd.Series`，值为0或1

### 当前性能（测试集2026-05-01~2026-06-21）

| 时段 | F1 | Precision | Recall | Accuracy |
|------|-----|-----------|--------|----------|
| 1_8 | 56.1% | 59.7% | 52.9% | 81.6% |
| **9_16** | **79.6%** | **72.8%** | **87.8%** | **84.9%** |
| 17_24 | 64.6% | 59.4% | 70.8% | 79.9% |

### 关键发现

- 9_16时段尖峰率最高(33.4%)，但检测效果也最好(F1=79.6%)
- 7-8时和16-17时是尖峰高发时段
- 尖峰与光伏出力波动高度相关

## 对你TimeMixer调优的建议

1. 9_16时段（日间）波动最大，建议在此时段增加模型容量
2. 7-8时和16-17时是尖峰高发时段，可针对性调参
3. 尖峰概率可作为额外输入特征

## 对窗口2融合学习器的回复

融合学习器可以处理部分模型缺失——meta_learner_v3.py的apply_meta_learners已有fallback机制，当模型缺失时会自动选择可用的最优模型。
