# 窗口2提示词 — 学习器 + 动态路由

你是窗口2，负责修复和升级融合学习器，并实现动态路由。请先阅读以下文档：
- `docs/coordination_plan.md` — 完整协调方案
- `docs/metrics_calculation.md` — SMAPE计算口径
- `docs/STATUS.md` — 其他窗口状态

## 你的核心任务

### 任务1：修复融合学习器bug

`fusion/meta_learner_v3.py` 有3个严重bug：

**Bug 1**: 真实预测时50/50平均（line 269）
```python
# 当前代码（错误）
y_pred = (gbm_preds + best_preds) / 2.0

# 应改为
y_pred = gbm_preds  # 直接用GBM
```

**Bug 2**: 数据泄漏（lines 259-265）
```python
# 当前代码（错误）- 用y_true决定选哪个模型
if has_y_true:
    gbm_smape = smape_floor50(group_y[valid], gbm_preds[valid])
    best_smape = smape_floor50(group_y[valid], best_preds[valid])
    y_pred = best_preds if best_smape < gbm_smape else gbm_preds

# 应改为
y_pred = gbm_preds  # 不再比较，直接用GBM
```

**Bug 3**: 特征工程不足
- 缺少星期特征（工作日/周末差异巨大）
- 缺少节假日特征
- 缺少月/季节特征
- 缺少滞后特征（前1小时价格）

### 任务2：实现动态路由

实现 `fusion/dynamic_router.py`，根据市场状态动态调整融合权重。

### 接口设计

```python
class DynamicRouter:
    def __init__(self, history_df: pd.DataFrame):
        """用历史数据训练路由策略"""
        pass

    def route(self, df: pd.DataFrame, model_predictions: dict) -> pd.Series:
        """根据当前状态选择模型或调整权重"""
        pass
```

## 沟通协议（必须严格遵守）

### 沟通规则

1. **每次开始新任务前**: 读 `docs/STATUS.md` 和 `docs/dialogues/` 下的最新文件
2. **每次修改文件前**: 读 `docs/STATUS.md` 确认文件没被锁定
3. **开始任务时**: 更新 STATUS.md，状态改为 🟡进行中
4. **完成任务时**: 更新 STATUS.md，状态改为 ✅完成，写明测试结果
5. **有重要发现时**: 在 `docs/dialogues/` 中写对话文件通知其他窗口
6. **需要协助时**: 在 STATUS.md 底部"问题/求助区"留言

### 对话文件格式

```
docs/dialogues/window2_to_all.md
```

### Git分支

```bash
git checkout -b improve-learner
```

## 测试命令

### 多日测试（必须！）

```bash
# 测试多个日期
python main.py 2026-06-17 --stage-models lightgbm,timesfm --target dayahead --validation-days 3
python main.py 2026-06-18 --stage-models lightgbm,timesfm --target dayahead --validation-days 3
python main.py 2026-06-19 --stage-models lightgbm,timesfm --target dayahead --validation-days 3

# 实时测试
python main.py 2026-06-17 --stage-models sgdfnet,timemixer --target realtime --validation-days 3
python main.py 2026-06-18 --stage-models sgdfnet,timemixer --target realtime --validation-days 3
python main.py 2026-06-19 --stage-models sgdfnet,timemixer --target realtime --validation-days 3
```

**通过标准**:
1. 融合 SMAPE 不劣于最优单模型
2. 多日测试稳定
3. 链路完整跑通

## 不能修改的文件

```
lightgbm/        ← 不改
TimesFM/         ← 不改
TimeMixer/       ← 窗口3负责
RT916_SpikeFusionNet/ ← 窗口4负责
fusion/spike_detector.py ← 窗口1负责
```
