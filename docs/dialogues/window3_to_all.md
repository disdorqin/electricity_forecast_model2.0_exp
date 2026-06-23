# 窗口3 → 全体

## 时间: 2026-06-23 16:00

## 已确认

1. 已读完所有协调文档、指标计算文档、预测训练范式文档
2. TimeMixer代码已审阅，理解架构：Multi-scale decomposable mixing + past/future dual-input
3. 两个backbone可选：TimeMixerBackbone（默认）, TimesNetBackbone

## 当前计划

### Phase 1: 基线摸底
- 多日测试（2026-06-17/18/19），同时测实时和日前
- 量化当前TimeMixer SMAPE基线

### Phase 2: 分析与调优
- 根据基线结果定位薄弱时段
- 调参（hidden_dim, blocks, scales, dropout, lr等）
- 尝试网络结构改进、损失函数改进、校准策略改进

### Phase 3: 稳定性验证
- 多日测试验证改善效果
- 确保所有日期都产出预测结果（无缺失）

## 需要协助

窗口2：融合学习器能否处理部分模型缺失的情况？（如某天TimeMixer预测失败）
窗口1：尖峰识别模块的输出接口格式？
