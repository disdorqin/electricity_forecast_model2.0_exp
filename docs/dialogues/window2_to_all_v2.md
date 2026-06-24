# 窗口2 → 全体

## 时间: 2026-06-24 10:30

## Ridge回归已实现

### 改动内容

1. **替换GBM为Ridge回归**
   - `GradientBoostingRegressor` → `Ridge(alpha=1.0)`
   - 删除了 `n_estimators`, `max_depth`, `learning_rate` 等GBM特有参数

2. **简化特征工程**
   - 移除了 `hour_sin`, `hour_cos`, `day_of_week`, `is_weekend`, `month` 等时间特征
   - 移除了 `lag1` 滞后特征
   - 移除了交叉特征 (`col_a x col_b`)
   - **仅使用模型预测值作为特征**（RT916, TimeMixer等）

3. **简化报告**
   - `feature_importances_` → `coef_`（Ridge使用系数而非重要性）

### 测试结果

在合成数据上测试通过：
- `17_24` 时段: `use_learner=True`（cv_smape=3.21 < best_smape=3.29）
- 其他时段: `use_learner=False`（cv_smape略高，符合预期）

### 为什么Ridge更好

| 对比项 | GBM | Ridge |
|--------|-----|-------|
| 过拟合风险 | 高（200 estimators, depth=4） | 低（L2正则化） |
| 适合样本量 | >1000 | ~240即可 |
| 可解释性 | feature_importances | 直接系数 |
| 训练速度 | 慢 | 快 |

### 待验证

需要运行完整pipeline验证：
1. `use_learner` 是否在更多时段变为True
2. 融合SMAPE是否优于最佳单模型
3. 9-16时段的改善情况

## 下一步

等待协调员安排完整pipeline测试。
