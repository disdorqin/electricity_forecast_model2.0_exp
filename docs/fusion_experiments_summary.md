# Fusion Runs 实验总结文档

**生成时间**: 2026-06-24 18:56:56
**实验总数**: 59

## 目录

1. [实验总览](#实验总览)
2. [实验详细记录](#实验详细记录)

---

## 实验总览

| 序号 | 实验名称 | 实验时间 | 实验类型 | 日前sMAPE | 实时sMAPE | 日前套利 | 实时套利 |
|------|----------|----------|----------|-----------|-----------|----------|----------|
| 1 | baseline_2026JanMay | 2026-06-23 17:26 | baseline | N/A | N/A | | |
| 2 | dayahead | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 3 | dayahead_202603_v1 | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 4 | dayahead_202603_v2_12m | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 5 | dayahead_202604_v1 | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 6 | dayahead_202604_v2_12m | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 7 | dayahead_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 8 | feb_single_model_audit | 2026-06-23 17:26 | audit | N/A | N/A | | |
| 9 | final_202602_full_pipeline | 2026-06-23 17:26 | final | N/A | N/A | | |
| 10 | final_fixed_202605 | 2026-06-23 17:26 | final | N/A | N/A | | |
| 11 | full_suite_20250201_20250228 | 2026-06-23 17:26 | suite | N/A | N/A | | |
| 12 | full_suite_20260201_20260228 | 2026-06-23 17:26 | suite | N/A | N/A | | |
| 13 | full_suite_20260201_20260228_v2 | 2026-06-23 17:26 | suite | N/A | N/A | | |
| 14 | full_suite_20260201_20260228_v3 | 2026-06-23 17:26 | suite | N/A | N/A | | |
| 15 | full_suite_20260201_20260228_v4 | 2026-06-23 17:26 | suite | N/A | N/A | | |
| 16 | fusion_v1_formal | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 17 | fusion_v1_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 18 | fusion_v2_enhanced_tm | 2026-06-23 17:26 | unknown | 25.03 | 16.29 | | |
| 19 | historical_monthly_benchmarks | 2026-06-23 17:26 | benchmark | N/A | N/A | | |
| 20 | lightgbm_batch_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 21 | lightgbm_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 22 | monthly_repro | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 23 | monthly_repro_apr_probe | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 24 | monthly_repro_apr_probe_fixedrunner | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 25 | monthly_repro_feb_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 26 | monthly_repro_mar_probe | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 27 | monthly_repro_mar_probe_fixedrunner | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 28 | quick_repro_202602 | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 29 | realtime | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 30 | realtime_20260201_20260228 | 2026-06-23 17:26 | unknown | N/A | N/A | | |
| 31 | realtime_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 32 | repro_training_length_probe | 2026-06-23 17:26 | reproduction | N/A | N/A | | |
| 33 | rolling_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 34 | smoke_dayahead_full_entry | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 35 | smoke_dayahead_full_entry_v2 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 36 | smoke_dayahead_meta | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 37 | smoke_dayahead_no_lgbm | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 38 | smoke_dayahead_protocol_v3 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 39 | smoke_dayahead_with_lgbm_final | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 40 | smoke_final_20260201_20260202 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 41 | smoke_fixed_20260501_20260507 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 42 | smoke_full_suite_delivery_check | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 43 | smoke_full_suite_delivery_check2 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 44 | smoke_full_suite_final | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 45 | smoke_joint_report_check | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 46 | smoke_protocol_v2 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 47 | smoke_realtime_full_entry | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 48 | smoke_realtime_protocol_v2 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 49 | smoke_realtime_protocol_v3 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 50 | timemixer_candidate_probe | 2026-06-23 17:26 | probe | N/A | N/A | | |
| 51 | timemixer_default_probe | 2026-06-23 17:26 | probe | N/A | N/A | | |
| 52 | timemixer_single_task_smoke | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 53 | timesfm_data_ablation | 2026-06-23 17:26 | ablation | N/A | N/A | | |
| 54 | timesfm_rt_skipstyle_probe | 2026-06-23 17:26 | probe | N/A | N/A | | |
| 55 | unified_entry | 2026-06-23 17:26 | unified_entry | N/A | N/A | | |
| 56 | unified_entry_may1 | 2026-06-23 17:26 | unified_entry | N/A | N/A | | |
| 57 | unified_entry_may1_v2 | 2026-06-23 17:26 | unified_entry | 35.71 | 17.49 | | |
| 58 | unified_entry_smoke_da | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |
| 59 | unified_entry_smoke_da2 | 2026-06-23 17:26 | smoke_test | N/A | N/A | | |

---

## 实验详细记录

### 1. baseline_2026JanMay

- **实验时间**: 2026-06-23 17:26
- **实验类型**: baseline
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\baseline_2026JanMay`
- **日前运行**: 否
- **实时运行**: 否

---

### 2. dayahead

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead`
- **日前运行**: 否
- **实时运行**: 否

---

### 3. dayahead_202603_v1

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead_202603_v1`
- **日前运行**: 否
- **实时运行**: 否

---

### 4. dayahead_202603_v2_12m

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead_202603_v2_12m`
- **日前运行**: 否
- **实时运行**: 否

---

### 5. dayahead_202604_v1

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead_202604_v1`
- **日前运行**: 否
- **实时运行**: 否

---

### 6. dayahead_202604_v2_12m

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead_202604_v2_12m`
- **日前运行**: 否
- **实时运行**: 否

---

### 7. dayahead_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\dayahead_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 8. feb_single_model_audit

- **实验时间**: 2026-06-23 17:26
- **实验类型**: audit
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\feb_single_model_audit`
- **日前运行**: 否
- **实时运行**: 否

---

### 9. final_202602_full_pipeline

- **实验时间**: 2026-06-23 17:26
- **实验类型**: final
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\final_202602_full_pipeline`
- **日前运行**: 否
- **实时运行**: 否

---

### 10. final_fixed_202605

- **实验时间**: 2026-06-23 17:26
- **实验类型**: final
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\final_fixed_202605`
- **日前运行**: 否
- **实时运行**: 否

---

### 11. full_suite_20250201_20250228

- **实验时间**: 2026-06-23 17:26
- **实验类型**: suite
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\full_suite_20250201_20250228`
- **日前运行**: 否
- **实时运行**: 否

---

### 12. full_suite_20260201_20260228

- **实验时间**: 2026-06-23 17:26
- **实验类型**: suite
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\full_suite_20260201_20260228`
- **日前运行**: 否
- **实时运行**: 否

---

### 13. full_suite_20260201_20260228_v2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: suite
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\full_suite_20260201_20260228_v2`
- **日前运行**: 否
- **实时运行**: 否

---

### 14. full_suite_20260201_20260228_v3

- **实验时间**: 2026-06-23 17:26
- **实验类型**: suite
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\full_suite_20260201_20260228_v3`
- **日前运行**: 否
- **实时运行**: 否

---

### 15. full_suite_20260201_20260228_v4

- **实验时间**: 2026-06-23 17:26
- **实验类型**: suite
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\full_suite_20260201_20260228_v4`
- **日前运行**: 否
- **实时运行**: 否

---

### 16. fusion_v1_formal

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\fusion_v1_formal`
- **日前运行**: 否
- **实时运行**: 否

---

### 17. fusion_v1_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\fusion_v1_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 18. fusion_v2_enhanced_tm

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\fusion_v2_enhanced_tm`
- **日前运行**: 是
- **实时运行**: 是

**关键指标:**

| 任务 | 时段 | sMAPE |
|------|------|-------|
| dayahead | overall | 25.03 |
| dayahead | 1_8 | 44.75 |
| dayahead | 9_16 | 10.42 |
| dayahead | 17_24 | 19.94 |
| realtime | overall | 16.29 |
| realtime | 1_8 | 20.91 |
| realtime | 9_16 | 7.17 |
| realtime | 17_24 | 20.80 |

---

### 19. historical_monthly_benchmarks

- **实验时间**: 2026-06-23 17:26
- **实验类型**: benchmark
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\historical_monthly_benchmarks`
- **日前运行**: 否
- **实时运行**: 否

---

### 20. lightgbm_batch_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\lightgbm_batch_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 21. lightgbm_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\lightgbm_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 22. monthly_repro

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro`
- **日前运行**: 否
- **实时运行**: 否

---

### 23. monthly_repro_apr_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro_apr_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 24. monthly_repro_apr_probe_fixedrunner

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro_apr_probe_fixedrunner`
- **日前运行**: 否
- **实时运行**: 否

---

### 25. monthly_repro_feb_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro_feb_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 26. monthly_repro_mar_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro_mar_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 27. monthly_repro_mar_probe_fixedrunner

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\monthly_repro_mar_probe_fixedrunner`
- **日前运行**: 否
- **实时运行**: 否

---

### 28. quick_repro_202602

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\quick_repro_202602`
- **日前运行**: 否
- **实时运行**: 否

---

### 29. realtime

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\realtime`
- **日前运行**: 否
- **实时运行**: 否

---

### 30. realtime_20260201_20260228

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unknown
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\realtime_20260201_20260228`
- **日前运行**: 否
- **实时运行**: 否

---

### 31. realtime_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\realtime_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 32. repro_training_length_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: reproduction
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\repro_training_length_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 33. rolling_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\rolling_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 34. smoke_dayahead_full_entry

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_full_entry`
- **日前运行**: 否
- **实时运行**: 否

---

### 35. smoke_dayahead_full_entry_v2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_full_entry_v2`
- **日前运行**: 否
- **实时运行**: 否

---

### 36. smoke_dayahead_meta

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_meta`
- **日前运行**: 否
- **实时运行**: 否

---

### 37. smoke_dayahead_no_lgbm

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_no_lgbm`
- **日前运行**: 否
- **实时运行**: 否

---

### 38. smoke_dayahead_protocol_v3

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_protocol_v3`
- **日前运行**: 否
- **实时运行**: 否

---

### 39. smoke_dayahead_with_lgbm_final

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_dayahead_with_lgbm_final`
- **日前运行**: 否
- **实时运行**: 否

---

### 40. smoke_final_20260201_20260202

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_final_20260201_20260202`
- **日前运行**: 否
- **实时运行**: 否

---

### 41. smoke_fixed_20260501_20260507

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_fixed_20260501_20260507`
- **日前运行**: 否
- **实时运行**: 否

---

### 42. smoke_full_suite_delivery_check

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_full_suite_delivery_check`
- **日前运行**: 否
- **实时运行**: 否

---

### 43. smoke_full_suite_delivery_check2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_full_suite_delivery_check2`
- **日前运行**: 否
- **实时运行**: 否

---

### 44. smoke_full_suite_final

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_full_suite_final`
- **日前运行**: 否
- **实时运行**: 否

---

### 45. smoke_joint_report_check

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_joint_report_check`
- **日前运行**: 否
- **实时运行**: 否

---

### 46. smoke_protocol_v2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_protocol_v2`
- **日前运行**: 否
- **实时运行**: 否

---

### 47. smoke_realtime_full_entry

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_realtime_full_entry`
- **日前运行**: 否
- **实时运行**: 否

---

### 48. smoke_realtime_protocol_v2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_realtime_protocol_v2`
- **日前运行**: 否
- **实时运行**: 否

---

### 49. smoke_realtime_protocol_v3

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\smoke_realtime_protocol_v3`
- **日前运行**: 否
- **实时运行**: 否

---

### 50. timemixer_candidate_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: probe
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\timemixer_candidate_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 51. timemixer_default_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: probe
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\timemixer_default_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 52. timemixer_single_task_smoke

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\timemixer_single_task_smoke`
- **日前运行**: 否
- **实时运行**: 否

---

### 53. timesfm_data_ablation

- **实验时间**: 2026-06-23 17:26
- **实验类型**: ablation
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\timesfm_data_ablation`
- **日前运行**: 否
- **实时运行**: 否

---

### 54. timesfm_rt_skipstyle_probe

- **实验时间**: 2026-06-23 17:26
- **实验类型**: probe
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\timesfm_rt_skipstyle_probe`
- **日前运行**: 否
- **实时运行**: 否

---

### 55. unified_entry

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unified_entry
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\unified_entry`
- **日前运行**: 是
- **实时运行**: 否

---

### 56. unified_entry_may1

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unified_entry
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\unified_entry_may1`
- **日前运行**: 是
- **实时运行**: 否

---

### 57. unified_entry_may1_v2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: unified_entry
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\unified_entry_may1_v2`
- **日前运行**: 是
- **实时运行**: 是

**关键指标:**

| 任务 | 时段 | sMAPE |
|------|------|-------|
| dayahead | overall | 35.71 |
| dayahead | 1_8 | 39.27 |
| dayahead | 9_16 | 14.14 |
| dayahead | 17_24 | 53.71 |
| realtime | overall | 17.49 |
| realtime | 1_8 | 31.82 |
| realtime | 9_16 | 13.45 |
| realtime | 17_24 | 7.19 |

---

### 58. unified_entry_smoke_da

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\unified_entry_smoke_da`
- **日前运行**: 否
- **实时运行**: 否

---

### 59. unified_entry_smoke_da2

- **实验时间**: 2026-06-23 17:26
- **实验类型**: smoke_test
- **路径**: `D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\electricity_forecast_model2.0_exp\fusion_runs\unified_entry_smoke_da2`
- **日前运行**: 否
- **实时运行**: 否

---
