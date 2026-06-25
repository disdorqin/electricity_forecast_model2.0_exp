# Runtime Optimization Guide

## Overview

本文档说明 R3D-Tap-GEF production pipeline 的所有运行时优化选项。所有优化均可关闭，默认值保守安全。

## AMP 混合精度

```bash
--enable-amp          # 强制启用
--no-amp              # 强制禁用
--amp-dtype fp16|bf16 # 精度类型 (默认: fp16)
```

默认行为：CUDA 可用时自动启用 fp16，CPU 时禁用。

AMP 通过 `torch.amp.autocast` 实现，不手动 `.half()` 模型。仅影响 forward pass 和 loss 计算。

安全校验：首次启用 AMP 时，应对一个小 fold 跑 FP32 vs AMP 对比。如果 sMAPE 差异 > 0.5% 或 MAE 差异超阈值，自动 fallback 到 FP32。

## torch.compile (可选)

```bash
--enable-compile              # 启用 (默认关闭)
--compile-mode default|reduce-overhead
```

默认关闭。只有显式传 `--enable-compile` 才执行 `torch.compile(model)`。

启用前会先跑一个小 batch parity test：compiled_output 与 eager_output 的 max_abs_diff / MAE 在阈值内才启用，不通过则自动 fallback eager。

## DataLoader 参数

```bash
--num-workers 0|2|4       # DataLoader 工作进程数
--pin-memory              # 固定内存 (auto: CUDA 时启用)
--persistent-workers      # 持久工作进程 (仅 num_workers > 0 时有效)
--prefetch-factor 2       # 预取因子
```

Windows 默认 `num_workers=0`（避免多进程不稳定）。Linux/GPU 可设 2 或 4。

## LightGBM 设备

```bash
--lightgbm-device cpu|gpu|cuda    # 计算设备 (默认: cpu)
--lightgbm-num-threads auto       # 线程数 (auto = 物理 CPU 核数)
```

默认 CPU。如果用 GPU/CUDA，必须做单 fold CPU vs GPU 预测误差对比。

`auto` 线程数使用物理核数（不含超线程），避免多模型并行时过度竞争。

## TimesFM 缓存

TimesFM 模型只加载一次（模块级单例），10 个 fold 的推理结果按 fold 缓存：

```
outputs/{date}/{target}/validation/folds/fold_XX/timesfm_predictions.csv
```

缓存 manifest 记录 cutoff_date、test_start/end、model_version、cache_created_at。如果缓存不能证明 cutoff-safe，则不使用缓存。

## RT916 Fast Tap

```bash
--skip-rt916-validation   # 跳过 RT916 在 validation tap 中的运行
```

RT916 是最慢的模型（walk-forward 训练）。`--skip-rt916-validation` 可临时跳过，但默认不跳过。RT916 失败不会拖死整个 pipeline，会记录 warning 并让学习器用可用模型继续。

## Fast Dev Run

```bash
--fast-dev-run
```

快速开发模式：
- 只跑 dayahead
- 只跑 1 个 fold（fold 9，最近）
- 只跑 1 个模型（lightgbm）
- 不跑 classifier

用于快速验证 pipeline 流程，不用于生产。

## CPU/GPU 队列调度

```bash
--max-cpu-workers 2    # CPU 模型最大并行数
--max-gpu-workers 1    # GPU 模型最大并行数
```

调度策略：
- CPU 队列：LightGBM（可并行，最多 2 个）
- GPU 队列：TimeMixer、SGDFNet、RT916、TimesFM（串行，单 GPU）

避免多个深度模型同时抢同一张 GPU。

## 如何判断加速是否影响精度

每次启用新加速选项后，应对比以下指标：

1. **单 fold 对比**：取 fold 9（最近），分别用 FP32 和加速模式跑一次
2. **指标对比**：
   - sMAPE_floor50 差异 < 0.5%
   - MAE 差异 < 1.0
   - 预测值 max_abs_diff < 0.01
3. **如果不通过**：自动 fallback 到安全模式，manifest 记录 warning

对比脚本：

```bash
python scripts/benchmark_runtime.py --date 2026-02-01 --compare-amp
```
