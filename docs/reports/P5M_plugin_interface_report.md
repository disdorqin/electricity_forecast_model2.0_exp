# P5M Plugin Interface Report

## 1. 为什么需要接口模块

现有预测流水线内部深度耦合多个模型的适配逻辑。当一个外部
模型需要接入融合、修正、监控链路时，必须了解内部 adapter 架构并修改
生产管线代码。

**Plugin 接口模块（`plugin/`）** 的解耦目标：

| 问题 | 接口模块方案 |
|------|-------------|
| 模型适配逻辑散落 | 统一 `PipelineAdapter` facade |
| 修正逻辑无统一基类 | `CorrectionModule` ABC + `CorrectionRegistry` |
| 监控指标散写在脚本中 | `MonitorModule` ABC + `MonitorRegistry` |
| 模型 CSV 无统一校验 | `PredictionTableSpec` + `standardize_predictions` |
| 生产管线需完整部署才能测试 | `PipelineAdapter.run()` 支持显式 df |

## 2. P5 External Model 数据契约

### 2.1 必需字段（8 列）

```text
model_name       — 模型名称（任意字符串）
business_day     — 交易日日期 YYYY-MM-DD
hour_business    — 交易小时 1-24（hour 24 = 次日 00:00，归属 business_day D）
timestamp        — 完整时间戳（可由 business_day + hour_business 自动推断）
y_pred           — 预测价格
source_file      — 来源 CSV 文件名
prediction_mode  — "dayahead" | "realtime" | "external"
leakage_safe     — 必须为小写字符串 "true"
```

### 2.2 可选字段（9 列）

```text
y_true           — 实际价格（评估用，外部 CSV 可缺失）
base_fused_pred  — 融合后修正前预测
final_pred       — 最终预测
high_spike_prob  — 高尖峰概率
negative_prob    — 负价格概率
low_valley_prob  — 低谷概率
module_name      — 修正/监控模块标识
task             — "dayahead" | "realtime"（兼容旧格式）
period           — "1_8" | "9_16" | "17_24"（可自动推断）
```

### 2.3 核心规则

- `leakage_safe` 必须是字符串 `"true"`（严格校验，大写 `True`、`"false"` 均拒绝）
- Hour 24 约定：00:00 时间戳 → `business_day - 1`, `hour_business = 24`
- (model_name, business_day, hour_business) 构成主键，不允许重复
- 支持 `allow_long_format=True` 以允许 ensemble 场景下的多行/小时
- 兼容旧别名：`target_day` → `business_day`，`ds` → `timestamp`

### 2.4 使用示例

```python
from plugin.schema import PredictionTableSpec, standardize_predictions

spec = PredictionTableSpec()
df = standardize_predictions(raw_df, spec=spec)
```

## 3. 外部 CSV 加载

### 3.1 ExternalPredictionSource 配置

```python
from plugin.external_loader import ExternalPredictionSource, load_external_predictions

source = ExternalPredictionSource(
    path="external_model_outputs/predictions.csv",
    column_mapping={"date": "business_day", "hour": "hour_business", "pred": "y_pred"},
    model_name_override="my_model",
    source_file_tag="my_model_v1",
    prediction_mode_override="dayahead",
)

df = load_external_predictions(source)
```

### 3.2 加载流程

1. 读取 CSV
2. 按 column_mapping 重命名列（identity 映射也可）
3. 应用 overrides（model_name、source_file、prediction_mode）
4. 调用 `standardize_predictions()` 执行完整规范化

## 4. Correction 模块

### 4.1 定义 CorrectionModule

```python
from plugin.correction_base import CorrectionModule

class MyCorrection(CorrectionModule):
    @property
    def name(self) -> str:
        return "my_correction"

    def correct(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        out = df.copy()
        out["y_pred"] = out["y_pred"].clip(lower=0)
        return out

    def validate(self, df: pd.DataFrame) -> bool:
        return "y_pred" in df.columns
```

### 4.2 注册与执行

```python
from plugin.correction_registry import CorrectionRegistry

registry = CorrectionRegistry()
registry.register(MyCorrection())
result = registry.run_all(df)
```

Correction 按注册顺序依次执行。

## 5. Monitor 模块

### 5.1 定义 MonitorModule

```python
from plugin.monitor_base import MonitorModule

class MyMonitor(MonitorModule):
    @property
    def name(self) -> str:
        return "my_monitor"

    def monitor(self, df: pd.DataFrame, **kwargs) -> dict:
        return {
            "n_rows": len(df),
            "mean_pred": float(df["y_pred"].mean()),
            "neg_count": int((df["y_pred"] < 0).sum()),
        }
```

### 5.2 注册与执行

```python
from plugin.monitor_registry import MonitorRegistry

registry = MonitorRegistry()
registry.register(MyMonitor())
report = registry.run_all(df)
```

所有 Monitor 返回 JSON-serialisable dict，汇聚为 `{name.key: value}` 格式。

## 6. PipelineAdapter 编排

```python
from plugin.pipeline_adapter import PipelineAdapter

adapter = PipelineAdapter()
adapter.register_external_source(source)
adapter.register_correction(MyCorrection())
adapter.register_monitor(MyMonitor())

# 自动加载外部 CSV → 修正 → 监控
result_df, metrics_report = adapter.run()

# 或传入已有 DataFrame
result_df, metrics_report = adapter.run(df=existing_df)
```

## 7. 文件清单

```text
plugin/
  __init__.py             — Package init
  schema.py               — PredictionTableSpec + standardize_predictions
  external_loader.py      — ExternalPredictionSource + CSV 加载
  correction_base.py      — CorrectionModule ABC
  correction_registry.py  — CorrectionRegistry + run_corrections
  monitor_base.py         — MonitorModule ABC
  monitor_registry.py     — MonitorRegistry + run_monitors
  pipeline_adapter.py     — PipelineAdapter facade

tests/
  test_p5m_plugin_interface.py         — 70 tests: schema, registry, pipeline
  test_p5m_negative_residual_module.py — 24 tests: negative correction module

scripts/
  evaluate_p5m_negative_residual_module.py — Negative correction evaluation
```

## 8. 测试覆盖（70 tests）

| 测试类 | 说明 |
|--------|------|
| TestColumnAliases | target_day/ds 别名映射 |
| TestTimestampMapping | timestamp ↔ (business_day, hour_business) 转换，hour 24 映射 |
| TestPredictionTableSpec | schema 校验 8 必需 + 9 可选 + 主键 + long-format |
| TestStandardizePredictions | standardize_predictions 完整规范化 |
| TestExternalLoader | CSV 加载、映射、override |
| TestCorrectionModule | ABC 基类 + 具体实现 |
| TestCorrectionRegistry | 注册/去重/执行/校验 |
| TestMonitorModule | ABC 基类 + 具体实现 |
| TestMonitorRegistry | 注册/去重/执行 |
| TestStandaloneFunctions | run_corrections / run_monitors 独立函数 |
| TestPipelineAdapter | PipelineAdapter 端到端编排 |
| TestP5Contract | y_true 非必需、leakage_safe 严格校验、hour 24 |
