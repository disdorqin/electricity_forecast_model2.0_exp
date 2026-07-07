"""
run_p3_spike_residual_shadow.py — 执行 P3 shadow-only 修正并写实验输出。

严格 shadow-only：只读 ledger 构建的 baseline_features.parquet，只写
outputs/p3_spike_residual/{run_id}/spike_residual_predictions.csv，绝不触碰
final_outputs / submission_ready.csv / 任何正式链路文件。

用法：
  python scripts/run_p3_spike_residual_shadow.py
  python scripts/run_p3_spike_residual_shadow.py --no-spike --no-residual
  python scripts/run_p3_spike_residual_shadow.py --neg-thresh 0.6 --run-id p3_ablation_negonly
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = "D:/作业/大创_挑战杯_互联网/大学生创新创业计划\大创实现\其他资料\efm3.0"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd
from experimental.p3_extreme_price_correction import config as cfg_mod
from experimental.p3_extreme_price_correction.pipeline_shadow import run_correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--baseline", default=None, help="baseline_features.parquet 路径")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-negative", action="store_true")
    ap.add_argument("--no-spike", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--optimized", action="store_true",
                    help="使用阶段 D/E 调优后的候选配置（optimized_config）")
    ap.add_argument("--neg-thresh", type=float, default=None)
    ap.add_argument("--spk-thresh", type=float, default=None)
    ap.add_argument("--neg-act-pred-cap", type=float, default=None)
    ap.add_argument("--spk-lift-ratio", type=float, default=None)
    ap.add_argument("--cap-abs", type=float, default=None)
    args = ap.parse_args()

    cfg = cfg_mod.optimized_config() if args.optimized else cfg_mod.default_config()
    if args.run_id:
        cfg.RUN_ID = args.run_id
    if args.no_negative:
        cfg.negative_classifier_enabled = False
    if args.no_spike:
        cfg.spike_classifier_enabled = False
    if args.no_residual:
        cfg.residual_corrector_enabled = False
    if args.neg_thresh is not None:
        cfg.NEG_THRESH = args.neg_thresh
    if args.spk_thresh is not None:
        cfg.SPK_THRESH = args.spk_thresh
    if args.neg_act_pred_cap is not None:
        cfg.NEG_ACT_PRED_CAP = args.neg_act_pred_cap
    if args.spk_lift_ratio is not None:
        cfg.SPK_LIFT_RATIO = args.spk_lift_ratio
    if args.cap_abs is not None:
        cfg.CAP_ABS = args.cap_abs

    baseline = args.baseline or f"{ROOT}/outputs/p3_spike_residual/{cfg.RUN_ID}/baseline_features.parquet"
    df = pd.read_parquet(baseline)
    # 保证按时间排序（walk-forward 一致性）
    df = df.sort_values("target_day").reset_index(drop=True)

    out, summary = run_correction(df, cfg)

    out_dir = args.out_dir or f"{ROOT}/outputs/p3_spike_residual/{cfg.RUN_ID}"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/spike_residual_predictions.csv"
    out.to_csv(out_path, index=False)

    cfg.to_json(f"{out_dir}/_config_used.json")
    import json
    with open(f"{out_dir}/_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("wrote:", out_path)
    print("summary:", json.dumps(summary, ensure_ascii=False))
    print("shadow_only = True for all rows; NOT written to submission_ready.csv")


if __name__ == "__main__":
    main()
