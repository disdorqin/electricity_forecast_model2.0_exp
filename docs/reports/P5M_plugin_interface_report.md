# P5M Plugin Interface Report

## 1. 为什么需要接口模块

现有预测流水线（`pipelines/production_pipeline.py`）内部深度耦合多个模型
（LightGBM、TimeMixer、TimesFM、SGDFNet、RT916）的适配逻辑。当一个外部
模型需要接入融合、修正、监控链路时，必须：

- 了解内部 adapter 架构
- 修改 production_pipeline.py 本身
- 绕过多层抽象才能完成简单的 CSV 接入

**接口模块（`pipeline_ext/`）** 的解耦目标：

| 问题 | 接口模块方案 |
|------|-------------|
| 模型适配逻辑散落在多个 adapter 中 | 统一 `PredictionProvider` 抽象 |
| 修正逻辑无统一基类 | `CorrectionModule` 定义 `apply(df) → df` |
| 监控指标散写在脚本中 | `MonitorModule` 定义 `run(df) → dict` |
| 模型 CSV 无统一校验 | `io.load_prediction_csv` 执行 schema + 业务规则检查 |
| 生产管线需经过完整部署才能测试 | `DryRunPipeline` 支持本地快速 smoke test |

## 2. 外部模型如何接入

### 2.1 准备 CSV

任何外部模型的预测结果只需满足 `pipeline_ext.schema.REQUIRED_FIELDS`：

```text
model_name, business_day, hour_business, timestamp, y_pred,
source_file, prediction_mode, leakage_safe
```

### 2.2 使用 load_prediction_csv 校验

```python
from pipeline_ext.io import load_prediction_csv

df = load_prediction_csv("external_model_outputs/predictions.csv")
```

校验链：
1. 必需字段存在 → 2. leakage_safe === true → 3. y_pred 无缺失 → 4. (business_day, hour_business) 唯一 → 5. (timestamp, model_name) 唯一（可选关闭）

### 2.3 可选：实现 PredictionProvider

```python
from pipeline_ext.modules import PredictionProvider
from pipeline_ext.registry import register_prediction_provider

class MyModelProvider(PredictionProvider):
    def load_predictions(self, path: str) -> pd.DataFrame:
        # custom loading logic
        return df

register_prediction_provider("my_model", MyModelProvider())
```

## 3. Correction 模块如何接入

### 3.1 实现 CorrectionModule

```python
from pipeline_ext.modules import CorrectionModule

class MyCorrection(CorrectionModule):
    name = "my_correction"

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["y_pred"] = out["y_pred"].clip(lower=0)
        return out
```

### 3.2 注册

```python
from pipeline_ext.registry import register_correction_module
register_correction_module("my_correction", MyCorrection())
```

### 3.3 执行顺序

Correction 模块按注册顺序执行。无依赖时按名称排序。

## 4. Monitor 模块如何接入

### 4.1 实现 MonitorModule

```python
from pipeline_ext.modules import MonitorModule

class MyMonitor(MonitorModule):
    name = "my_monitor"

    def run(self, df: pd.DataFrame) -> dict:
        return {
            "n_rows": len(df),
            "mean_pred": float(df["y_pred"].mean()),
            "neg_count": int((df["y_pred"] < 0).sum()),
        }
```

### 4.2 注册

```python
register_monitor_module("my_monitor", MyMonitor())
```

### 4.3 结果收集

所有 Monitor 返回 JSON-serialisable dict，汇聚后写入 `monitor_report.json`。

## 5. 当前不接 production_pipeline

`pipeline_ext/` 当前设计为**独立的 smoke-test 层**：

- 不导入 `pipelines.production_pipeline`
- 不依赖内部 adapter 结构
- 不修改任何现有生产代码

`DryRunPipeline` 只做 dry-run：

```bash
python scripts/run_p5m_plugin_pipeline_smoke.py \
    --prediction-pack outputs/my_model/predictions.csv \
    --correction-modules identity,clamp_low \
    --out-dir outputs/plugin_smoke
```

## 6. 后续 Production Integration 入口

当外部模型经过 smoke test 验证后，接入 production 的推荐路径：

1. 在 `pipeline_ext/` 基础上扩展 `ProductionAdapter`，包装已有 `DryRunPipeline`
2. 在 `pipelines/production_pipeline.py` 的 Step 2/3 之间插入：

```python
from pipeline_ext.pipeline import DryRunPipeline
from pipeline_ext.registry import get_module

# 预检查：dry-run 验证
dry = DryRunPipeline()
dry.run_from_path("external_preds.csv")

# production 集成：直接在 main pipeline 中调用
correction = get_module("my_correction")
df = correction.apply(df)
```

3. 可选择将 `pipeline_ext.pipeline` 提升为正式的 `PipelineStage`，加入
   `staged_pipeline.py` 的生产流程

## 文件清单

```text
pipeline_ext/
  __init__.py    — Package init, exports
  schema.py      — Unified prediction schema + validation helpers
  registry.py    — Module registry (provider / correction / monitor)
  io.py          — CSV load + full validation chain
  modules.py     — ABCs: PredictionProvider, CorrectionModule, MonitorModule
  pipeline.py    — DryRunPipeline orchestration

scripts/
  run_p5m_plugin_pipeline_smoke.py — Smoke test CLI

tests/
  test_p5m_plugin_interface.py     — Coverage: schema, registry, correction order,
                                     leakage-safe, edge cases
```
