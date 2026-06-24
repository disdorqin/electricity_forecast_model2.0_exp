# 窗口1提示词 — 尖峰识别模块

你是窗口1，负责实现尖峰识别模块。请先阅读以下文档：
- `docs/coordination_plan.md` — 完整协调方案
- `docs/metrics_calculation.md` — SMAPE计算口径
- `docs/STATUS.md` — 其他窗口状态

## 你的核心任务

实现一个尖峰识别模块，用于识别极端电价时段。

### 设计思路

不再机械复现模型，而是结合现有框架：
- 基于历史数据统计特征（均值、方差、分位数）
- 基于时间特征（时段、星期、节假日）
- 输出：每个时刻的尖峰概率

### 实现位置

`fusion/spike_detector.py`（新增文件）

### 接口设计

```python
class SpikeDetector:
    def __init__(self, history_df: pd.DataFrame):
        """用历史数据训练尖峰检测器"""
        pass

    def predict_spike_probability(self, df: pd.DataFrame) -> pd.Series:
        """预测每个时刻的尖峰概率"""
        pass

    def is_spike(self, prob: float, threshold: float = 0.7) -> bool:
        """判断是否为尖峰"""
        pass
```

### 尖峰定义

根据 `docs/metrics_calculation.md`：
- 实时价格远高于日前价格的时段
- 或者价格异常高/低的时段
- 具体阈值需要根据历史数据确定

### 测试方法

```python
# 在Python中测试
from fusion.spike_detector import SpikeDetector
import pandas as pd

df = pd.read_excel("data/shandong_pmos_hourly.xlsx")
detector = SpikeDetector(df)
test_df = df.tail(48)  # 最后48小时
probs = detector.predict_spike_probability(test_df)
print(probs)
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
docs/dialogues/window1_to_all.md
```

内容示例：
```markdown
# 窗口1 → 全体

## 发现

尖峰时段主要集中在9-16点（光伏出力高峰），占比约15%。

## 打算

下一步计划将尖峰概率作为特征输入融合学习器。

## 需要协助

窗口2：融合学习器能否接受尖峰概率作为额外特征？
```

### Git分支

```bash
git checkout -b spike-detector
```

## 测试命令

```bash
# 测试链路是否跑通
python main.py 2026-06-19 --stage-models lightgbm,timesfm --target dayahead --validation-days 3
```

## 不能修改的文件

```
lightgbm/        ← 不改
TimesFM/         ← 不改
TimeMixer/       ← 窗口3负责
SGDFNet/         ← 窗口2负责（融合部分）
RT916_SpikeFusionNet/ ← 窗口4负责
fusion/meta_learner_v*.py ← 窗口2负责
```
