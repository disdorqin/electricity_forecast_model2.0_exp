# P3 Spike Residual / Classifier — 实时技术日志（WorkBuddy 工作笔记本）

> 本文件是本次长任务的"实时更新窗口"。每完成一个阶段/关键决策即追加更新，作为跨会话记忆与进度追踪。
> 任务性质：shadow-only 研究，不修改 2.5 / 3.0 正式链路，不写 submission_ready.csv。
> 最后更新：2026-07-06

---

## 0. 任务边界（硬性，来自用户指令 + 工作记忆锁定）

- ⛔ 禁止写入/改动 `electricity_forecast_model2.5`（只读锁定）。
- ⛔ 禁止修改 `electricity_price_forecast3.0`（efm3.0）正式链路文件：`main.py`/`pipelines/`/`fusion/`/`runners/`/`ExtremPriceClf/`/各模型目录。
- ⛔ 禁止覆盖 `final_outputs`、禁止写入 `submission_ready.csv`。
- ✅ 允许：在 efm3.0 下新增 `experimental/`、`docs/p3_*`、`scripts/run_p3_*`、`outputs/p3_spike_residual/`、`exports/efm3_candidates/` 等独立实验子目录。
- ✅ 所有修正必须 shadow-only：`applied=false` 默认；即便 `applied=true` 也只进实验输出，不进正式结果。
- ✅ 严格 D14 cutoff-safe：只用 D-1 14:00 前可见信息；禁止 D 日 14:00 后 / D+1 actual 作在线特征。

## 1. 环境侦察（2026-07-06）

- 四个研究仓本地齐备（同父目录 `其他资料/`）：
  - `efm3.0` = 3.0 总仓（remote→price_forecast3.0，仅只读使用，不 push）
  - `electricity_forecast_model2.0_exp` = 经验源（git@github ...2.0_exp.git）
  - `deep_sgdf_delta_repo` = DeepSGDFDelta 实验（仅 models/scripts/tests）
  - `electricity_forecast_model2.5` = 只读锁定源
- 数据分析环境：managed python 3.13.12，已带 pandas 3.0.3 / pyarrow 24.0.0（无需建 venv）。
- **评估数据金矿**：efm3.0 `outputs/ledger/` 存 consolidated ledger parquet。
  - RT 预测 3072 行 = 4 模型(rt916/sgdfnet/timemixer/timesfm) × 32 天 × 24h
  - RT 真实 768 行 = 32 天满 24h（2026-01-25 ~ 2026-02-25），data_cutoff 显示 D-1 14:00（cutoff-safe ✓）
  - 负价 213h（167h=-80 地板），尖峰>500 共 26h（最高 1291），覆盖两大目标现象。

## 2. 阶段 A 经验提炼（完成，见 docs/p3_spike_residual_existing_experience.md）

- 2.5 现状：ExtremPriceClf 两阶段级联负价分类器（雷达+灰度区GBM，标签≤-50，动态灰度recall≥0.95），**但 2.5 后期改为"最终输出纯融合电价，仅单独输出-80概率文件"** → 即分类器已不再改价。P3 正是验证"是否值得改价"。
- 2.0_exp 最宝贵资产：`extreme/negative_price`(NegativeResidualCorrector)、`extreme/realtime_high_spike`(ResidualLiftCorrector)、`residual_stack`(上行只正/下行只负/互斥/护栏/全链路reason)、`PHASE12_INTRADAY_SHADOW_REPLAY.md`(SHADOW_ONLY门控范式)。
- SGDFNet：`delta=RT-DA` + 分段偏差校准（最像残差校正器，可复用思路）。
- RT916："Spike"为分段+SMAPE-floor50损失（基础预测器非校正器）；注意其曾有的目标泄漏已修复（推理时目标派生特征需 cutoff 后重算）。
- TimeMixer：spike 处理以配置候选形式存在，无失败案例文档；仅作基础模型输入。

## 3. 基线验证（build_baseline_features.py，run_id=p3_rt_20260125_20260225_v1）

- 融合基线 = inverse-MAE expanding-window 集成（leakage-free）。
- overall: MAE=92.9, RMSE=136.4, sMAPE_floor50=40.9
- negative(n=213): MAE=117.9, sMAPE=78.1（最差）；actual 中位 -80.0；fused 中位 -4.3（47%预测为正）→ **负价漏判严重，修正价值最高**
- spike(n=26): MAE=249.5, sMAPE=39.9；fused 中位 401.7，88%低估>100 → 上行修正有空间
- normal(n=529): MAE=75.1, sMAPE=25.9（最好）
- period: 9_16 最差(sMAPE 54.8)，17_24 最好(24.0)
- 已落盘 `outputs/p3_spike_residual/p3_rt_20260125_20260225_v1/baseline_features.parquet`（768 行，含特征+original_pred+actual）

## 4. 设计要点（阶段 B，进行中）

- 三校正器 + 双护栏 + 报告，全部可关闭/可回滚/可 shadow。
- negative corrector：保守 —— 仅当 negative_prob 高 且 original_pred ≤ 阈值(暂定100) 时推向 -80 地板；命中率/精度由护栏保障。
- spike corrector：有界上行 lift（max_lift_ratio 暂定 0.35，max_absolute_lift 暂定 350，period_9_16_boost）。
- residual corrector：SGDFNet 式 delta=RT-DA 分段偏差校准（expanding 历史残差）。
- guard：correction cap / 价格范围[-100,1500] / 正常时段损伤护栏 / NaN / 24h 完整。
- rollback：NaN/缺小时/超cap/正常时段恶化/低置信 → 回退 original_pred。

## 5. 阶段进度（全部完成 ✅）

| 阶段 | 状态 | 产物 |
|------|------|------|
| A 经验梳理 | ✅ | docs/p3_spike_residual_existing_experience.md |
| B 系统设计 | ✅ | docs/p3_extreme_price_correction_design.md |
| C shadow 原型 | ✅ | experimental/p3_extreme_price_correction/*.py, scripts/run_p3_*, outputs/.../spike_residual_predictions.csv |
| D before/after | ✅ | reports/spike_residual_before_after_report.md, metrics.json |
| E ablation | ✅ | reports/spike_residual_ablation_report.md |
| F candidate 包 | ✅ | exports/efm3_candidates/spike_residual/{run_id}/* |

## 6. 最终结论（2026-07-06）

- 候选配置（optimized_config）：负价分类器 + 尖峰分类器（残差校正 DROP）。
- 单月 32 天 shadow 评估通过全部 16 条标准：
  - overall sMAPE 40.88 → 34.22（-6.66）
  - 负价 78.14 → 53.75（-24.39），分类 P=0.894/R=0.636/F1=0.743
  - 尖峰 39.95 → 36.26（-3.69），分类器弱(P=0.118)但修正安全
  - 正常时段 +0.33（不明显），period 17_24 +0.14（不恶化）
  - 无 NaN / 24h 完整 / D14 cutoff 安全 / rollback 可用 / shadow-only
- recommended_status = **candidate**（单月达标，但缺多月份稳定性证据，不升 shadow）。
- Final Verdict = **PASS**。
- 关键教训：负价修正是绝对主力且保守护栏（cap≤50）可压住假阳性；残差校正经 ablation 证实无收益已丢弃；尖峰分类器需更多样本重训。
- 路径坑：Write 曾把最终报告建到全角"其他資料"目录，已移回并清理。

## 6. 关键决策记录

- run_id = `p3_rt_20260125_20260225_v1`（数据范围+版本，稳定可复现）。
- original_pred 采用 inverse-MAE rolling 集成（非生产BGEW，但透明且 leakage-free），将在报告中明确标注。
- 评估集仅 32 天（≈1 个月），不足以声称"多月份稳定"；最高建议状态为 `candidate`（设计完成+单月验证）或视稳定性 `shadow`。严禁 `champion`。
