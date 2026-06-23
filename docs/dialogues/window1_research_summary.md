# 窗口1 → 全体

## 时间: 2026-06-23 22:15

## 研究总结：尖峰检测技术与经验

### 一、从现有负电价分类器学到的经验

学习了 `ExtremPriceClf` 的两阶段级联架构，关键经验：

1. **两阶段级联**：Stage1轻量级 + Stage2灰度区精细化
2. **动态阈值**：基于P-R曲线在precision约束下最大化recall
3. **特征工程**：
   - 时序特征：lag(24h, 48h, 7天), ramp(差分)
   - 日历特征：节气、节假日、星期
   - 专家特征：竞价空间为负、中午高空间
   - 极值水位：当前值/历史均值的比率
4. **Momentum规则**：如果前一时刻是尖峰，降低当前时刻的判断门槛
5. **类别权重**：scale_pos_weight处理样本不平衡

### 二、论文调研

1. **arXiv:2602.05430 (AAAI'26)** - "Day-Ahead EPF with Spike Regularization"
   - 提出spike regularization策略
   - 评估了TTM, MOIRAI, MOMENT, TimesFM等基础模型
   - TSFMs比传统方法提升37.4% MAPE
   - 关键：使用exogenous factors(天气、日历)

2. **arXiv:1711.11053 (NeurIPS'17)** - "Multi-Horizon Quantile Recurrent Forecaster"
   - 分位数回归用于概率预测
   - 处理"event spikes"和cold starts

### 三、尖峰检测器v2改进

基于以上经验，v2版本改进：

| 改进项 | v1 | v2 |
|--------|-----|-----|
| 模型 | 单GBM | 两阶段级联(GBM+灰度区) |
| 特征 | 价格+时间 | 价格+负荷+新能源+竞价空间+日历+节气 |
| Momentum | 无 | 有(0.85) |
| 9_16 F1 | ~50% | **79.6%** |
| Recall | ~34% | **87.8%** |

### 四、集成到融合学习器的方案

```python
# 在staged_pipeline.py的run_learner_stage中
from fusion.spike_detector import SpikeDetector

# 训练尖峰检测器
detector = SpikeDetector()
detector.fit(val_df)

# 添加尖峰概率作为特征
val_df['spike_prob'] = detector.predict_spike_probability(val_df)

# 传给meta_learner_v3
```

### 五、对各窗口的建议

**窗口2（融合学习器）：**
- GBM过拟合问题：降低n_estimators到100，max_depth到3
- 或增加validation_days到60
- 可以将spike_prob作为额外特征

**窗口3（TimeMixer）：**
- 9_16时段尖峰率最高(33.4%)，建议在此时段增加模型容量
- 7-8时和16-17时是尖峰高发时段

**窗口4（RT916）：**
- RT模型在7-10时早高峰表现最差
- 建议针对早高峰时段调参

## 需要协助

1. 窗口2：是否需要我在learner_stage中集成尖峰检测器？
2. 窗口3/4：基线测试结果如何？需要尖峰时段的详细数据吗？
