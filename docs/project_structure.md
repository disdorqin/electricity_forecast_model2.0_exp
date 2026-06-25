# Project Structure Guide

## 目录结构

```
electricity_forecast_model2.0_exp/
│
├── main.py                    # 统一入口
├── cli/
│   └── parser.py              # CLI 参数解析
│
├── pipelines/                 # 生产 pipeline
│   ├── production_pipeline.py # R3D-Tap-GEF 6步主流程
│   ├── validation_tap.py      # 10-fold 三日验证 tap
│   └── r3d_output_validator.py# 输出校验
│
├── fusion/
│   ├── learners/
│   │   ├── r3d_tap_gef.py    # Gated Expert Fusion 学习器
│   │   ├── metrics.py         # 指标函数 (sMAPE, MAE, RMSE)
│   │   └── apply_learner.py   # 学习器应用工具
│   └── classifier_bridge.py   # 负电价分类器桥接
│
├── rolling_oof/               # 模型适配器框架
│   ├── adapters/              # 各模型 adapter 实现
│   │   ├── base.py            # BaseRollingAdapter 抽象基类
│   │   ├── lightgbm.py
│   │   ├── timemixer.py
│   │   ├── timesfm.py
│   │   ├── sgdfnet.py
│   │   └── rt916.py
│   ├── contracts.py           # FoldSpec, FoldResult 数据类
│   └── scheduler.py           # ADAPTER_REGISTRY, 调度器
│
├── data/                      # 原始数据
│   └── shandong_pmos_hourly.csv  # GBK 编码
│
├── outputs/                   # 生产输出 (按日期)
│   └── {date}/
│       ├── run_manifest.json
│       ├── dayahead/
│       │   ├── validation/
│       │   ├── real/
│       │   ├── fused/
│       │   └── final/
│       ├── realtime/
│       │   ├── validation/
│       │   ├── real/
│       │   ├── fused/
│       │   └── final/
│       └── final/
│
├── models/                    # 模型权重 (git-ignored)
│   └── timesFM/
│
├── docs/                      # 文档
│   ├── r3d_tap_gef.md        # 架构文档
│   ├── runtime_optimization.md# 加速指南
│   └── project_structure.md  # 本文件
│
├── scripts/                   # 工具脚本
│   ├── run_smoke_test.py
│   ├── inspect_outputs.py
│   └── benchmark_runtime.py
│
├── archive/                   # 归档 (旧脚本，不删除)
│   └── old_scripts/
│
├── TimeMixer/                 # TimeMixer 模型代码 (不移动)
├── TimesFM/                   # TimesFM 模型代码 (不移动)
├── SGDFNet/                   # SGDFNet 模型代码 (不移动)
├── RT916/                     # RT916 模型代码 (不移动)
└── LightGBM/                  # LightGBM 模型代码 (不移动)
```

## 生产入口

```bash
# 单日预测
python main.py 2026-02-01

# 强制重跑
python main.py 2026-02-01 --force

# 快速开发测试
python main.py 2026-02-01 --force --fast-dev-run

# 日期范围
python main.py 2026.2.1-2026.2.28
```

## 目录规则

1. **生产主流程**只放在 `pipelines/production_pipeline.py`
2. **R3D 学习器**只放在 `fusion/learners/r3d_tap_gef.py`
3. **临时测试脚本**放在 `scripts/`
4. **旧实验脚本**移动到 `archive/old_scripts/`，不删除
5. **模型目录**（TimeMixer、TimesFM、SGDFNet、RT916、LightGBM）不移动，避免 import 断掉
6. 移动后跑 `python main.py --help` 确保不报错

## 输出结构

每个日期目录包含：
- `run_manifest.json`：运行清单，记录每步状态和警告
- `{target}/validation/`：10-fold 验证结果
- `{target}/real/`：预测日各模型输出
- `{target}/fused/`：权重、融合结果、debug 信息
- `{target}/final/`：最终预测（含分类器修正）
- `final/`：顶层汇总（DA+RT+submission_ready）

## 缓存行为

- `outputs/{date}` 已存在且无 `--force`：提示已预测过，退出
- 有 `--force`：删除整个目录重新执行
- 单步级缓存：中间文件已存在且非空则跳过
