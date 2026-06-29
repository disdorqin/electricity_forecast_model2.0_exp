# Prediction Ledger Bootstrap — 每日预测账本冷启动

## 目标

为后续 Regime-Ledger-GEF（RLG）权重学习系统提供每日预测账本骨架。
当前**不做**复杂权重学习，也不替换 R3D-Tap-GEF 主线，只实现从今天开始每日累积
的 prediction ledger 结构和质量检查。

## 背景决策

- **不以历史 backfill 作为第一主线**，而是从真实每日预测开始累积
- 满 30 天后启用正式 ledger 学习（Regime-Ledger-GEF）
- P0 阶段仍以 realtime extreme high spike 诊断为主
- ledger 本身不参与当前 P0 融合决策，仅做数据积累

## 账本 Schema

每行对应一个模型对一个业务日某小时的预测。完整的列定义见 [](../ledger/schema.py)。

| 列名 | 类型 | 说明 |
|---|---|---|
| run_date | str | pipeline 运行日期 |
| forecast_date | str | 业务日 (business_day) |
| hour_business | Int64 | 业务小时 1~24 |
| timestamp | str | 自然时间戳 |
| target | str | dayahead 或 realtime |
| model_name | str | 模型名 |
| y_pred | float64 | 模型原始预测 |
| base_fused_pred | float64 | 融合预测（不含 spike 修正） |
| spike_corrected_pred | float64 | spike 修正后预测（可空） |
| final_pred | float64 | 最终输出预测 |
| y_true | float64 | 真实值（后续回填） |
| period | str | 1_8 / 9_16 / 17_24 |
| available_data_cutoff | str | 数据截止时间描述 |
| pipeline_version | str | pipeline 版本标识 |
| source_file | str | 来源文件路径 |
| created_at | str | 账本行创建时间 |

## 核心模块

### 

- 列名常量 + dtype 映射
- : 校验 DataFrame 是否符合 schema
- : 业务小时 → 自然时间戳
- : 自然时间戳 → 业务小时

### 

- : 主入口，扫描 pipeline 产出并追加
- : 扫描 outputs/{date}/ 下的标准文件
- 自动去重：以 (run_date, forecast_date, hour_business, target, model_name) 为键
- 按 target 分文件写入：ledger_dayahead.csv / ledger_realtime.csv + 合并 ledger.csv

### 

- : 全面质量检查
- 检查项：
  1. 每个 target/business_day 是否 24 行（×模型数）
  2. hour_business 是否 1~24
  3. timestamp 00:00 是否正确映射到上一业务日 hour 24
  4. 是否重复写入
  5. 是否缺模型
  6. 是否缺 final_pred

## 使用方式



## 输出路径约定

| 文件 | 路径 |
|---|---|
| 合并账本 | data/local_ledger/ledger.csv |
| 日前账本 | data/local_ledger/ledger_dayahead.csv |
| 实时账本 | data/local_ledger/ledger_realtime.csv |

以上路径均已加入 .gitignore。

## 30 天后启用权重学习

累积满 30 天后，可执行以下步骤启用 Regime-Ledger-GEF 权重学习：

1. **回填 y_true**: 从原始数据按 (forecast_date, hour_business) 匹配真实电价
   
2. **启用权重学习**: 开发 Regime-Ledger-GEF 模块，以 ledger 为训练数据
   - 按 regime（时间段）分组学习模型权重
   - 支持在线更新（每日追加后增量学习）
   - 融合权重输出作为 R3D-Tap-GEF 的可选输入
3. **切换决策**: 比较 RLG vs R3D-Tap-GEF 在验证期的 sMAPE_floor50
   - 如果 RLG 显著更优，替换 Step 3 的 learner
   - 否则维持当前 R3D-Tap-GEF 主线

## 注意事项

- 不要修改 production_pipeline.py 主流程
- 不要在分支上提交实际账本文件（已在 .gitignore 中忽略）
- ledger 设计仅为数据积累，当前不参与融合决策
