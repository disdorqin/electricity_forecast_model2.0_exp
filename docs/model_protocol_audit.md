# 基础模型协议审计报告

> 审计日期: 2026-06-24  
> 审计范围: `electricity_forecast_model2.0_exp` 与 `electricity_forecast_model2.0`  
> 审计目的: 为 rolling-origin OOF 池统一协议做好准备

---

## 审计维度

对每个模型从以下10个维度进行审计：

| # | 维度 | 说明 |
|---|------|------|
| 1 | `trains_actually` | 模型是否真实训练（非占位符） |
| 2 | `current_train_mode` | 训练模式：daily_walk_forward / window_once / pretrained |
| 3 | `current_predict_mode` | 预测模式：daily / batch / pretrained_inference |
| 4 | `is_daily_walk_forward` | 是否逐日重新训练 |
| 5 | `is_window_once` | 是否训练一次预测整个区间 |
| 6 | `uses_cache` | 是否使用缓存文件 |
| 7 | `uses_pretrained` | 是否使用预训练权重 |
| 8 | `supports_explicit_train_window` | 是否支持显式 train_start/train_end |
| 9 | `supports_explicit_test_window` | 是否支持显式 test_start/test_end |
| 10 | `leakage_risk` | 数据泄露风险等级 |

---

## 审计结果

### LightGBM（日前 + 实时）

| 审计项 | 结果 | 详细说明 |
|--------|------|----------|
| trains_actually | **yes** | 训练4个LGBM子模型（valley/solar/solar_clf/peak），真实训练 |
| current_train_mode | 实时=**daily_walk_forward**, 日前=**window_once** | 实时：while循环逐日训练。日前：只训练一次然后循环预测 |
| is_daily_walk_forward | 实时=**yes**, 日前=**no** | 实时模式合格，日前模式需改造 |
| is_window_once | 实时=**no**, 日前=**yes** | 日前模式需改为 daily_walk_forward |
| uses_cache | **no** | 训练时保存 jblib 文件但主流程不用缓存 |
| uses_pretrained | **no** | 不使用外部预训练权重 |
| leakage_risk | **medium** | 实时 info_cutoff_dt 不一致；日前使用 prev_day 特征合理 |
| timestamp_alignment_risk | **medium** | 1秒偏移法（总体一致），但实时非预测温度路径有偷看风险 |
| required_changes | 日前模式需改为 daily_walk_forward；info_cutoff_dt 统一化 |

---

### TimeMixer

| 审计项 | 结果 | 详细说明 |
|--------|------|----------|
| trains_actually | **yes** | 完整训练循环（前向/反向/AdamW/CosineAnnealing/EarlyStop） |
| current_train_mode | **window_once** | 训练一次，批量预测整个测试月 |
| is_daily_walk_forward | **no** | 不在测试月上逐日滚动重训练 |
| is_window_once | **yes** | 训练窗口固定 |
| uses_cache | **no** | 不读取任何 .pt/.pth 文件 |
| uses_pretrained | **no** | 每次从零训练 |
| leakage_risk | **low** | 训练窗口明确排除测试月；cutoff=D-1 15:00；pred_da_map 使用预测值 |
| timestamp_alignment_risk | **low** | business_hour 映射正确（00:00→24） |
| required_changes | 新增 block 和 daily walk-forward 模式 |

---

### TimesFM

| 审计项 | 结果 | 详细说明 |
|--------|------|----------|
| trains_actually | **no** | 纯预训练模型，无训练/finetune |
| current_train_mode | **pretrained** | 加载 google/timesfm-2.5-200m-pytorch |
| is_daily_walk_forward | **n/a** | 无训练环节 |
| is_window_once | **n/a** | 无训练环节 |
| uses_cache | **yes** | 检查 backtest CSV 缓存文件 |
| uses_pretrained | **yes** | Google TimesFM 预训练权重 |
| leakage_risk | **low** | 上下文截止于预测日前；目标列排除于外生变量 |
| timestamp_alignment_risk | **low** | trading_day 计算一致（00:00→前一天） |
| required_changes | cutoff-safe 缓存策略；fold 专用预测函数 |

---

### SGDFNet

| 审计项 | 结果 | 详细说明 |
|--------|------|----------|
| trains_actually | **yes** | HistGradientBoostingRegressor，逐日训练 |
| current_train_mode | **daily_walk_forward** | 严格 cutoff protocol（decision_hour=15） |
| is_daily_walk_forward | **yes** | 逐日训练+预测 |
| is_window_once | **no** | 不使用窗口一次性训练 |
| uses_cache | **no** | 每个 decision_day 从头训练 |
| uses_pretrained | **no** | 不使用预训练权重 |
| leakage_risk | **low** | 严格 cutoff 协议；可见数据 frame 正确构造 |
| timestamp_alignment_risk | **low** | 00:00→前一天 business_hour=24 |
| required_changes | 基本合格，只需统一接口封装 |

---

### RT916/SpikeFusionNet

| 审计项 | 结果 | 详细说明 |
|--------|------|----------|
| trains_actually | **yes** | 每个时段训练神经网络（12 epoch + AdamW + 早停） |
| current_train_mode | **window_once** | 默认 retrain_daily=False，可选 True |
| is_daily_walk_forward | **no**（默认） | 训练层面不是，仅推理逐日；可选 retrain_daily=True |
| is_window_once | **yes**（默认） | 训练只做一次 |
| uses_cache | **yes**（推理时） | 加载 MinMaxScaler .pkl + 模型 .pth |
| uses_pretrained | **no** | 不加载外部预训练权重 |
| leakage_risk | **HIGH** | **训练时未调用 apply_asof_cutoff_for_inference**，lag 特征可能看到未来数据 |
| timestamp_alignment_risk | **HIGH** | adjust_date_for_0am 仅改日期不改小时，与其他模型不一致 |
| required_changes | **高优先级**：修复 asof_cutoff + 时间对齐 + 强制 retrain_daily |

---

## 总结

| 模型 | 协议兼容性 | 主要差距 |
|------|-----------|----------|
| LightGBM(实时) | ✅ 合格 | - |
| LightGBM(日前) | ❌ 需改 | window_once → daily_walk_forward |
| TimeMixer | ❌ 需改 | window_once → block/daily |
| TimesFM | ⚠️ 需增强 | 缓存需 cutoff-safe 验证 |
| SGDFNet | ✅ 合格 | 只需统一接口 |
| RT916 | ❌ 严重 | asof_cutoff 缺失 + 时间不对齐 |

### 改进优先级

1. **P0**: RT916 修复 asof_cutoff + 时间对齐
2. **P0**: TimeMixer 新增 daily/block walk-forward
3. **P1**: LightGBM 日前改为 daily_walk_forward
4. **P1**: TimesFM cutoff-safe 缓存
5. **P2**: SGDFNet 统一接口

---

## 统一协议要求（所有模型必须满足）

1. `train_end < test_start`（训练截止于预测开始前）
2. 训练集不包含 test 期真实标签
3. test 期只用于预测和评估，不参与调参
4. 内部 early stopping/validation 只在训练窗口内做时间顺序切分
5. 禁止随机 8:2 切分时间序列
6. 每个 business_day 输出 24 行（01:00-24:00）
7. 00:00 统一归属前一天（business_hour=24）
