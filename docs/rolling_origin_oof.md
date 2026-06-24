# Rolling-Origin OOF 预测池

> 统一基础模型 rolling-origin out-of-fold 预测池生成流程

---

## 1. 为什么升级到 rolling-origin OOF 池

### 当前问题

原有 pipeline 使用单一验证窗口（如30天静态窗口）：

```
训练 2024-01~2025-12 → 验证 2026-01-01~2026-01-30
```

**存在的问题：**
- 30天验证集太短，且只有一段
- 不同模型训练/预测协议不一致（有的 window_once，有的 daily_walk_forward）
- 学习器拿到的 OOF 数据口径不一致，导致融合效果差
- 单段验证集的权重泛化能力有限

### Rolling-Origin 的优势

```
Fold 0: 训练 2023-01~2026-07 → 预测 2026-08
Fold 1: 训练 2023-01~2026-08 → 预测 2026-09
Fold 2: 训练 2023-01~2026-09 → 预测 2026-10
Fold 3: 训练 2023-01~2026-10 → 预测 2026-11
Fold 4: 训练 2023-01~2026-11 → 预测 2026-12
```

- 5段 OOF 预测，覆盖不同时间分布
- 每段严格 cutoff-safe（训练截止于预测开始前）
- 学习器用多段数据训练，权重更稳健

---

## 2. Rolling-Origin 与普通 8:2 的区别

| 维度 | 普通 8:2 | Rolling-Origin |
|------|----------|----------------|
| 切分方式 | 一次随机/time-based 切分 | 多段 expanding 窗口 |
| 验证段数 | 1段 | N段（N=目标月数） |
| 数据量 | 20%数据用于验证 | 每段一个完整月的 OOF |
| 时间覆盖 | 单一段时间段 | 覆盖多个目标月 |
| 权重稳定性 | 依赖单段 | 多段交叉验证 |

---

## 3. 基础模型内部 validation 与学习器训练数据的区别

| 层次 | 数据 | 用途 |
|------|------|------|
| 基础模型内部 validation | 训练窗口的后20%（时间顺序） | early stopping / 超参选择 |
| OOF 预测 | 训练窗口截止后的完整目标月 | 作为学习器的训练数据 |
| 学习器训练 | 所有 fold 的 OOF 预测合并 | 学习最优融合权重 |

**关键原则：** 学习器训练数据必须是真正 out-of-fold 的 —— 基础模型从未见过目标月的真实值。

---

## 4. OOF 数据如何避免泄露

### Expanding Window 协议

```
Fold 0: train_end = 2026-07-31, test = 2026-08 (无重叠)
Fold 1: train_end = 2026-08-31, test = 2026-09 (无重叠)
...
```

**保证措施：**
1. `train_end < test_start`（训练截止于预测开始前）
2. 每个 fold 训练时：所有 lag/rolling 特征截止于 train_end
3. 不同 fold 严格顺序执行（fold 1 在 fold 0 之后）
4. 缓存按 train_end 隔离（避免不同 fold 的缓存污染）

---

## 5. 最终陪跑为什么仍需要用最新可用数据训练

- Rolling-origin OOF 池用于**学习融合权重**
- 预测明天时，用**截止到昨天的最新数据**重新训练所有模型
- 这样基础模型利用了最新的市场信息
- 学习器用历史 OOF 池学到的权重来融合最新预测

---

## 6. TimeMixer Rolling-Mode 三种模式

| 模式 | 训练次数（30天） | 严格性 | 计算成本 | 推荐场景 |
|------|-----------------|--------|----------|----------|
| `window_once` | 1次 | 低 | 最低 | 基准对比 |
| `block` (默认7天) | ~5次 | 中 | 中等 | 日常实验 |
| `daily` | 30次 | 高 | 高 | 正式 OOF 池 |

**推荐 OOF 池使用 `daily` 模式**，与 LightGBM 实时和 SGDFNet 保持一致的 daily_walk_forward 协议。

---

## 7. 输出文件结构

```
oof_runs/
  oof_2026-08_to_2026-12_expanding/
    manifest.json                     # OofPoolManifest
    protocol_audit.json               # 全局审计报告
    oof_long_table.csv                # 统一 long-table（学习器唯一数据源）
    
    folds/
      fold_0/
        fold_spec.json
        audits/
          lightgbm__dayahead__audit.json
          ...
        lightgbm/dayahead/
          fold_0_dayahead_raw.csv
          fold_0_dayahead_long.csv
        ...
    
    escort/
      escort_2026-01-01_long.csv
```

### Long-Table 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| task | str | "dayahead" / "realtime" |
| model_name | str | 模型名称 |
| fold_id | int | fold 编号 |
| target_day | str | YYYY-MM-DD |
| ds | datetime | 时间戳 |
| period | str | "1_8" / "9_16" / "17_24" |
| hour_business | int | 1-24 |
| y_true | float | 真实值 |
| y_pred | float | 预测值 |

---

## 8. 后续学习器如何使用 oof_long_table.csv

```python
# 加载 OOF 数据
import pandas as pd
long_table = pd.read_csv("oof_runs/oof_2026-08_to_2026-12_expanding/oof_long_table.csv")

# 按 task + period 分组学习权重
for (task, period), group in long_table.groupby(["task", "period"]):
    X = group.pivot(index=["target_day", "ds"], columns="model_name", values="y_pred")
    y = group["y_true"].drop_duplicates()
    # 学习融合权重...
```

---

## 9. CLI 使用

```bash
# 完整 OOF 池生成
python main.py --pipeline rolling_oof \
  --date 2026-12-31 \
  --oof-start-month 2026-08 --oof-end-month 2026-12 \
  --data-path data/shandong_pmos_hourly.xlsx

# 陪跑预测
python main.py --pipeline rolling_oof \
  --date 2026-06-25 \
  --escort-date 2026-06-25

# 自定义模型和模式
python main.py --pipeline rolling_oof \
  --date 2026-12-31 \
  --oof-start-month 2026-08 --oof-end-month 2026-12 \
  --oof-models lightgbm,timemixer \
  --timemixer-rolling-mode block --timemixer-block-days 7
```
