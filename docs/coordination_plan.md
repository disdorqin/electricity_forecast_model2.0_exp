# 四窗口并行协调方案

## 项目基本信息

### 硬件环境

- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU
- **Conda环境**: `epf-2`，路径 `D:\computer_download\environment\conda\epf-2\python.exe`
- **PyTorch**: 2.5.1+cu121，CUDA可用

### 模型清单

| 模型 | 任务类型 | 设备 | 代码位置 | 可改？ |
|------|---------|------|---------|--------|
| lightgbm | DA | CPU | `lightGBM/` | ❌ 不改 |
| timesfm | DA+RT | GPU | `TimesFM/` | ❌ 不改 |
| timemixer | DA+RT | GPU | `TimeMixer/` | ✅ 可改模块 |
| sgdfnet | RT | CPU | `SGDFNet/` | ✅ 可改模块 |
| rt916 | RT | GPU | `RT916_SpikeFusionNet/` | ✅ 可改模块 |

**日前(DA)模型**: lightgbm, timesfm, timemixer（3个）
**实时(RT)模型**: timesfm, timemixer, sgdfnet, rt916（4个）

### DA→RT联动

- DA预测的D+1（24点）会作为RT预测的输入特征
- RT不是独立的，它显式依赖DA的预测结果

### 分时段

- 1-8点（低谷/valley）
- 9-16点（日间/光伏/solar）
- 17-24点（晚高峰/peak）

## 目标

| 指标 | 目标 | 稳定性要求 |
|------|------|-----------|
| 日前准确率 | ≥ 90%（SMAPE ≤ 10%） | 连续一个月达标 + 跨月稳定 |
| 实时准确率 | ≥ 85%（SMAPE ≤ 15%） | 连续一个月达标 + 跨月稳定 |

**当前状态**: DA已基本达标，RT差距较大，需要重点攻关。

## SMAPE计算公式（必须准确！）

```python
def smape_floor50(y_true, y_pred, eps=1e-6):
    """
    SMAPE-floor50: 先裁剪再计算
    - y_true 和 y_pred 中低于50的值都被裁剪到50
    - 分母也做了eps保护防止除零
    - 返回值是百分比数值（如12.5表示12.5%）
    """
    true_clip = np.where(y_true < 50.0, 50.0, y_true)
    pred_clip = np.where(y_pred < 50.0, 50.0, pred_clip)
    denom = (np.abs(true_clip) + np.abs(pred_clip)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(pred_clip - true_clip) / denom) * 100.0)

# Accuracy = 100% - SMAPE%
```

**关键细节**:
- 裁剪到50是为了避免低价区间对误差的放大效应
- 负电价和接近零的价格都会被裁剪到50
- 返回值是百分比数值，如 `9.81` 表示 9.81%
- Accuracy = 100 - SMAPE，如 SMAPE=9.81% → Accuracy=90.19%

## 硬性约束

1. **不要修改 lightgbm/ 和 TimesFM/**
2. **不要数据泄露**：训练集和验证集严格按时间划分
3. **所有改动必须保证链路跑通**
4. **持续执行直到整月稳定达标**

## 四窗口分工

### 窗口1 — 尖峰识别模块

**任务**: 实现尖峰识别模块，识别极端电价时段
**可以改**: `fusion/spike_detector.py`（新增）
**不能改**: 其他窗口负责的文件

### 窗口2 — 学习器 + 动态路由

**任务**: 修复和升级融合学习器，实现动态路由
**可以改**: `fusion/meta_learner_v*.py`, `fusion/dynamic_router.py`（新增）
**不能改**: 其他窗口负责的文件

### 窗口3 — TimeMixer调优

**任务**: 调优TimeMixer，可以改模块代码（不只是调参）
**可以改**: `TimeMixer/` 目录下的文件
**不能改**: 其他窗口负责的文件

### 窗口4 — RT916调优

**任务**: 调优RT916，可以改模块代码（不只是调参）
**可以改**: `RT916_SpikeFusionNet/` 目录下的文件
**不能改**: 其他窗口负责的文件

## 沟通机制

### 1. 对话文件夹

创建 `docs/dialogues/` 文件夹，各窗口在此放置对话文件：

```
docs/dialogues/
├── window1_to_window2.md  ← 窗口1给窗口2的消息
├── window2_to_window1.md  ← 窗口2给窗口1的回复
├── window3_to_all.md      ← 窗口3给全体的消息
└── ...
```

**规则**:
- 每次有重要决策或发现时，写一个对话文件
- 文件名格式: `window{N}_to_window{M}.md` 或 `window{N}_to_all.md`
- 开始工作前先读 `docs/dialogues/` 下的最新文件
- 对话内容包括：发现了什么、打算做什么、需要什么协助

### 2. STATUS.md — 状态同步

每个窗口在开始/完成/遇到问题时更新 `docs/STATUS.md`。

### 3. Git提交

每次有意义的改动都 commit，commit message 说明改了什么。

## 测试规范

### 多日测试（必须！）

不要只测一天，至少测3天对比：

```bash
# 测试多个日期
python main.py 2026-06-17 --stage-models timemixer --target realtime --validation-days 3
python main.py 2026-06-18 --stage-models timemixer --target realtime --validation-days 3
python main.py 2026-06-19 --stage-models timemixer --target realtime --validation-days 3
```

### 对比基线

每次测试后，与之前的基线对比：
- 之前的SMAPE是多少？
- 改善了多少？
- 是否稳定？

### 通过标准

1. 融合 SMAPE 不劣于最优单模型
2. 所有模型输出非空（无 NaN/掉队）
3. 链路完整跑通
4. 多日测试稳定

## 执行顺序

### Phase 1：摸底（各窗口并行）

各窗口同时对自己负责的模块进行多日测试，量化当前基线。

### Phase 2：各自修改（并行）

各窗口在自己的分支上修改代码，每完成一个阶段性改动就：
1. 跑多日测试
2. 更新 STATUS.md
3. 在 dialogues/ 中写对话文件通知其他窗口

### Phase 3：集成

1. 读取所有窗口的 STATUS.md 和 dialogues/
2. 合并各分支
3. 跑全量测试
4. 解决冲突

### Phase 4：稳定验证

1. 连续运行整月数据
2. 跨月验证
3. 确认达标后交付
