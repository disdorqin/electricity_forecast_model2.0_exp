# electricity_forecast_model2.0_exp 验证与修复提示词

> **角色**：你是验证工程师 + bug 修复者。先验证、定位问题、修复、再验证。
> **分支**：`tune-timemixer`
> **立即可做**：`cd` 到项目目录后直接跑命令。只改代码中定位到的 bug，不做结构大改。

---

## 0. 环境

```
项目根目录：D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp
Python: D:/computer_download/environment/conda/epf-2/python.exe
Conda env: epf-2 (torch 2.5.1+cu121)
GPU: NVIDIA GeForce RTX 4060 Laptop (若存在)
```

每个命令前先 `cd` 到项目根目录。

---

## 一、先跑快速验证（≤ 2 分钟）

### Step A: CLI 和 CPU 模型

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp"

# A1: CLI 无报错
"D:/computer_download/environment/conda/epf-2/python.exe" main.py --help

# A2: LightGBM CPU 全链路（约 6 秒）
"D:/computer_download/environment/conda/epf-2/python.exe" main.py 2026-02-01 --force --target dayahead --models lightgbm --no-amp
```

**A2 通过标准**：
- stdout 末尾包含 `Validation tap: 720 rows`
- stdout 包含 `tap_fold_id has 10 values (got 10)`
- stdout 包含 `date coverage D-30~D-1 ... got 2026-01-02~2026-01-31`
- stdout 包含 `final CSV dayahead_final_predictions.csv has 24 rows (got 24)`
- stdout 包含 3 条 `weights sum ≈ 1`（17_24、1_8、9_16 均 PASS）
- exit code = 0

---

## 二、验证已修复的 bug：load_segment_checkpoint pred_len 推断

如果 Step A2 通过，此 bug 已在上一个 commit（`2626239`）修复。

你需要手动验证修复是否生效。最简单的方法：

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp"

# 跑 TimeMixer online，只需跑完 base train + block 0 predict，看是否报错
"D:/computer_download/environment/conda/epf-2/python.exe" main.py 2026-02-01 --force --target dayahead --models timemixer --timemixer-online-epochs 1 --no-amp
```

**通过标准**：
- 不出现 `size mismatch for head.7.weight`
- 不出现 `loading state_dict for TimeMixerBackbone`
- 日志出现 `[buffered_online/dayahead] Base train complete` 或类似
- exit code = 0 (或 `complete_with_warnings`)

如果仍然报 `size mismatch`：
1. 读 `TimeMixer/repro_pipeline.py` 第 2766-2775 行，确认 `head_keys[-1]` 被使用（不是 `head_key[0]`）
2. 若代码已是最新但仍有 bug，打印 state_dict 中所有 head.*.weight 的 shape，找出哪个才是 pred_len：
```python
for k in sorted(state_dict.keys()):
    if k.startswith("head.") and "weight" in k:
        print(f"  {k}: {state_dict[k].shape}")
```
3. 根据打印结果调整 pred_len 推断逻辑

---

## 三、完整 Day-ahead 验证（30-60 分钟，GPU 必须）

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp"

"D:/computer_download/environment/conda/epf-2/python.exe" main.py 2026-02-01 --force --target dayahead --models lightgbm,timesfm,timemixer --timemixer-online-epochs 1 --timesfm-inference-mode block --no-amp
```

**通过标准**：
- 无 segfault、无 CUDA OOM、无 exception
- `Validation tap: ____ rows` — 3 模型应接近 3×30×24 = **2160 rows**
- `tap_fold_id has 10 values`
- `final CSV dayahead_final_predictions.csv has 24 rows`
- `weights sum ≈ 1` 对 3 个 period 都 PASS
- runtime_report.csv 存在：`outputs/2026-02-01/dayahead/validation/runtime_report.csv`

**如果失败，收集信息**：
```bash
# 查看最后 50 行日志
tail -50 outputs/2026-02-01/logs/pipeline.log

# 查看 runtime report
cat outputs/2026-02-01/dayahead/validation/runtime_report.csv

# 查看 validation tap 行数
"D:/computer_download/environment/conda/epf-2/python.exe" -c "import pandas as pd; df=pd.read_csv('outputs/2026-02-01/dayahead/validation/validation_tap_long_table.csv'); print(f'rows={len(df)}, models={df.model_name.unique().tolist()}, folds={sorted(df.tap_fold_id.unique())}')"
```

---

## 四、完整 Realtime 验证（30-60 分钟，GPU 必须）

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp"

"D:/computer_download/environment/conda/epf-2/python.exe" main.py 2026-02-01 --force --target realtime --models sgdfnet,timemixer,rt916,timesfm --timemixer-online-epochs 1 --rt916-online-epochs 1 --sgdfnet-fold-strategy 3x10 --timesfm-inference-mode block --no-amp
```

**通过标准**：
- 无 segfault、无 CUDA OOM
- `Validation tap: ____ rows` — 4 模型应接近 4×30×24 = **2880 rows**
- `tap_fold_id has 10 values`
- `final CSV realtime_final_predictions.csv has 24 rows`
- `weights sum ≈ 1` 对 3 个 period 都 PASS

**RT916 如果 buffered online 失败**：
- 应自动 fallback 到 `single_train_range`（source_confidence=0.70）
- stdout 中出现 `single_train_range` 字样
- 不应回退到 `_run_online_per_fold`（那会导致 10 次 base train）

---

## 五、完整双 Target 验证（60-120 分钟）

当 Step 三 + 四各自通过后：

```bash
cd "D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp"

"D:/computer_download/environment/conda/epf-2/python.exe" main.py 2026-02-01 --force --timemixer-online-epochs 1 --rt916-online-epochs 1 --sgdfnet-fold-strategy 3x10 --timesfm-inference-mode block --no-amp
```

**通过标准**：
- 两个 target 均完成，exit code 0
- 输出目录包含所有要求文件（见下方检查）
- daily 模式可选验证（切回 daily 后重跑）

**输出目录完整性检查**：
```bash
echo "=== Dayahead ===" && ls outputs/2026-02-01/dayahead/validation/ && echo "---" && ls outputs/2026-02-01/dayahead/fused/ && echo "---" && ls outputs/2026-02-01/dayahead/final/
echo "=== Realtime ===" && ls outputs/2026-02-01/realtime/validation/ && echo "---" && ls outputs/2026-02-01/realtime/fused/ && echo "---" && ls outputs/2026-02-01/realtime/final/
echo "=== Final ===" && ls outputs/2026-02-01/final/
```

---

## 六、常见问题排查

### 问题 1：TimesFM segfault
```
症状：Segmentation fault (core dumped)，发生在 import transformers 或加载 TimesFM 模型时
排查：
  1. nvidia-smi 确认驱动正常
  2. python -c "import torch; print(torch.cuda.is_available())" 返回 True
  3. 尝试单独 import TimesFM: python -c "from TimesFM.infer import predict_price_for_range"
  4. 如果单独 import 也 segfault → 环境问题，不是代码 bug
  5. 用 --timesfm-inference-mode block --models (去掉 timesfm) 先跳过
```

### 问题 2：TimeMixer CUDA OOM
```
症状：RuntimeError: CUDA out of memory
解决：
  1. 关闭其他 GPU 进程
  2. 不需要改代码。如果硬件限制无法解决，标记为环境限制
```

### 问题 3：fill_y_true 警告
```
症状：fill_y_true: cannot read data/shandong_pmos_hourly.csv
原因：data_path 是相对路径或文件不存在
检查：ls data/shandong_pmos_hourly.csv
如果存在但路径不对，用 --data-path 指向绝对路径重跑
如果不影响学习器（y_true 在 LightGBM 输出中已有），这是良性警告
```

### 问题 4：learner trace 顺序验证
```bash
"D:/computer_download/environment/conda/epf-2/python.exe" -c "
import pandas as pd
df = pd.read_csv('outputs/2026-02-01/dayahead/fused/dynamic_weight_trace.csv')
if not df.empty:
    for (t,p), g in df.groupby(['task','period']):
        order = g.drop_duplicates('tap_fold_id').sort_values('age_block')['tap_fold_id'].tolist()
        expected = list(range(9,-1,-1))
        status = 'PASS' if order == expected else 'FAIL'
        print(f'{t}/{p}: trace order={order} [{status}]')
else:
    print('Trace is empty (expected with 1 model)')
"
```

---

## 七、链路通过后：项目结构清理

**仅在所有验证命令跑通后再做**。

### 7.1 创建新目录结构

```
runners/          ← 模型 adapter / registry（从 rolling_oof/adapters 迁移）
  registry.py
  adapters/
    lightgbm.py
    timesfm.py
    timemixer.py
    sgdfnet.py
    rt916.py

runtime/          ← 新目录，放加速/调度/profile
  optim.py
  resource_scheduler.py
  profiles.py

archive/          ← 旧文件归档
  old_fusion_scripts/
  old_pipelines/
  old_timesfm_tf/
```

### 7.2 TF/ 合并到 TimesFM/

```
TimesFM/
  infer.py
  pipeline.py
  src/
  _legacy/                    ← 从 TF/ 移入
    price_forecast_copy_分时段预测.py
```

### 7.3 清理规则

- **不要删除文件**，用 `git mv` 移动
- 移动后全局搜索 import 路径，全部修成新的 import path
- 确认 `python -c "from runners.registry import ADAPTER_REGISTRY"` 能 import
- 废弃文件移到 `archive/`，不被 import 的可以不移

---

## 八、README 更新

全部验证通过后，更新 `README.md` 包含：

1. 项目目标：山东电力现货市场日前/实时电价预测
2. 四阶段 pipeline 流程图（文字版即可）
3. 5 个模型的 validation 策略表
4. TimeMixer/RT916 buffered online 机制（base train + 3×10day + seasonal replay）
5. 学习器 R3D-Tap-GEF 公式（四因子 gate + BGEW + convex refit）
6. 输出目录结构
7. 快速验证命令：
```bash
# Smoke test (CPU only, ~6s)
python main.py 2026-02-01 --force --target dayahead --models lightgbm --no-amp

# GPU 最低参数 day-ahead
python main.py 2026-02-01 --force --target dayahead --timemixer-online-epochs 1 --timesfm-inference-mode block --no-amp

# 完整单日
python main.py 2026-02-01 --force --no-amp
```
8. 常见错误排查（对应上面第六节）

---

## 九、最终汇报

全部完成后按这个格式汇报：

```
=== 验证结果 ===
1. Step A2 (lightgbm-cpu): [PASS/FAIL] — 720 rows / 24 final / 3 weights sum≈1
2. Step 二 (timemixer online fix): [PASS/FAIL] — 无 head mismatch / 或具体报错
3. Step 三 (dayahead GPU): [PASS/FAIL] — 验证行数 / final 行数 / weights
4. Step 四 (realtime GPU): [PASS/FAIL] — 验证行数 / final 行数 / weights
5. Step 五 (full double): [PASS/FAIL] — 目录完整性
6. 项目结构清理: [DONE/SKIP]
7. README 更新: [DONE/SKIP]

=== 关键数据 ===
- dayahead validation tap 行数: ____ (目标 ≈2160)
- realtime validation tap 行数: ____ (目标 ≈2880)
- dayahead real forecast 行数: ____ (目标 72)
- realtime real forecast 行数: ____ (目标 96)
- 单日总 runtime: ____ 秒 / ____ 分钟
- 最慢模型: ____ (耗时 ____ 秒)
- learner trace 顺序: 9→0 [PASS/FAIL]
- TimesFM cutoff-safe: [VERIFIED/UNVERIFIED]
- classifier corrected 文件: [EXISTS/NOT EXISTS/SKIPPED]
- RT916 fallback 使用情况: [none / single_train_range / per-fold (BUG)]
- 输出目录: [COMPLETE/INCOMPLETE]
```
