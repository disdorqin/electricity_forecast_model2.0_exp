# 窗口2 → 全体

## 时间: 2026-06-23 21:15

## 状态

meta_learner_v3.py 3个bug已修复并提交(improve-learner分支, aa6f67f)。

## 修复内容

1. 移除y_true数据泄漏
2. 修复50/50平均bug（直接用GBM）
3. 增强特征工程（添加星期/月份/滞后特征）
4. 添加use_learner标志

## 测试结果

多日测试DA融合(2026-06-17/18):
- learner判定use_learner=False（所有时段）
- 回退到最优单模型
- 2026-06-17: 融合SMAPE=28.77%（vs最优单模型lightgbm 7.35%）
- 2026-06-18: 融合SMAPE=14.57%（vs最优单模型timesfm 10.55%）

## 问题分析

use_learner=False的原因：cv_smape > best_single_smape
- 9_16时段: timesfm的cv_smape=63-70%远大于best_single=28%
- 这说明GBM在小样本(72行)上过拟合

## 下一步

1. 降低GBM复杂度（n_estimators=100, max_depth=3）
2. 或者增加训练样本（扩大validation_days）
3. 跑实时(RT)测试

## 需要协助

无
