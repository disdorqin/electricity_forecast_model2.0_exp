# 实时状态同步

> **规则**: 每个窗口在开始任务前读取此文件，在开始/完成/遇到问题时更新自己的状态段。
> **时间戳**: 每次更新必须带时间戳，格式 `[MM-DD HH:MM]`。
> **状态**: ⬜待开始 | 🟡进行中 | ✅完成 | 🔴阻塞 | ⏸️等待

---

## 窗口1 - 尖峰识别模块

**状态:** ✅ 完成（v2改进版 + 管道集成）
**当前任务:** fusion/spike_detector.py v2已完成，已集成到staged_pipeline.py
**改动文件:** fusion/spike_detector.py (v2), pipelines/staged_pipeline.py (集成spike detector), test_spike_detector.py, analyze_spikes.py
**分支:** spike-detector
**时间戳:** [06-23 22:00]
**备注:** 
- v2改进: 两阶段级联(GBM+灰度区)、momentum规则、丰富特征(负荷/新能源/竞价空间/日历)
- SpikeDetector训练准确率89.88%，9_16时段F1=79.6%，Recall=87.8%
- 已集成到staged_pipeline.py的learner_stage（自动训练、预测、保存spike_predictions.csv）
- 全链路测试通过：DA SMAPE=10.37%, Accuracy=89.63%
- 已回复窗口2和窗口3的对话

---

## 窗口2 - 学习器+动态路由

**状态:** 🟡 进行中
**当前任务:** GBM参数已降低，但use_learner仍为False，需要进一步分析
**改动文件:** fusion/meta_learner_v3.py [🔒 锁定], pipelines/staged_pipeline.py [🔒 锁定]
**分支:** improve-learner
**时间戳:** [06-23 21:45]
**备注:** 已提交两次(aa6f67f, 122a91b)。降低GBM参数后2026-06-19 DA融合SMAPE=15.56%，但use_learner仍False。CV SMAPE仍大于best_single_smape。需要考虑：(1)进一步简化GBM (2)增加训练样本 (3)换用更简单的学习器如Ridge。

---

## 窗口3 - TimeMixer调优

**状态:** 🟡 进行中
**当前任务:** Phase 2 — 调优进行中，6轮实验完成，RT overall从42.82%降至35.61%
**改动文件:** TimeMixer/backbones.py, TimeMixer/repro_pipeline.py
**分支:** tune-timemixer
**时间戳:** [06-23 22:30]
**备注:** 最佳配置v6: TimesNet backbone, seq_len=336, hidden_dim=128, blocks=4, residual_blend, auto calibration, no segment training. RT 9_16仍54.98%是主要瓶颈。

---

## 窗口4 - RT916调优

**状态:** 🟡进行中
**当前任务:** Phase 2 - 第一轮改进（模型容量+训练策略+损失函数）
**改动文件:** core.py, annual_model.py, annual_loss.py
**分支:** tune-rt916
**时间戳:** [06-23 21:45]
**备注:** 
- 基线: val SMAPE=34.45%, forecast SMAPE=58.27%
- 改进1: d_model 64→128, e_layers 1→2, top_k 2→3, num_kernels 3→6, dropout 0.1→0.2
- 改进2: epochs 12→30, patience 4→8, lr 3e-4→5e-4, batch_size 64→32
- 改进3: loss加SMAPE项, protected_weight 1.5→2.0, tail_alpha 1.0→1.5
- 改进4: 模型输出clamp范围 -0.35~1.80 → -0.5~2.0
- 改进5: early stopping用SMAPE替代MAE

---

## 问题/求助区

（无）
