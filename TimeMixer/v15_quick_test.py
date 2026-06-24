"""v15 quick test on single day for TimeMixer with 9-16 loss weighting."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction, smape  # noqa: E402

DATA_PATH = r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\epf\data\shandong_pmos_hourly.csv"


def make_v15_cfg(test_date: str, output_dir: str, *, epochs: int = 40) -> RunConfig:
    """Build a v15 config: same as v14 baseline plus 9-16 loss weighting."""
    month = pd.Timestamp(test_date).strftime("%Y-%m")
    return RunConfig(
        data_path=DATA_PATH,
        output_dir=output_dir,
        month=month,
        test_start=test_date,
        test_end_exclusive=(pd.Timestamp(test_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        append_leaderboard=False,
        # 架构与 v14 一致
        backbone="timesnet",
        rt_916_backbone=None,
        seq_len=168,
        epochs=epochs,
        batch_size=32,
        hidden_dim=128,
        blocks=6,
        scales=3,
        dropout=0.15,
        lr=2e-4,
        weight_decay=1e-4,
        patience=15,
        seed=42,
        device="auto",
        cutoff_hour_da=15,
        cutoff_hour_rt=15,
        segment_training=False,
        # 目标模式与残差 baseline
        target_mode="residual_blend",
        da_target_mode="residual_blend",
        rt_target_mode="residual_blend",
        da_calibration_mode="none",
        da_loss_mode="l1",
        da_under_weight_multiplier=1.25,
        rt_calibration_mode="rt_916_auto",
        rt_loss_mode="l1",
        # === v15 核心改进：9-16 时段感知 loss ===
        rt_916_loss_weight=1.5,
        rt_916_spike_penalty=0.3,
        rt_916_spike_threshold=350.0,
        da_916_loss_weight=1.25,
    )


def smape_report(csv_path: str, task: str) -> dict[str, dict[str, float]]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    sub = df[df["task"] == task].dropna(subset=["y_true", "y_pred"])
    out: dict[str, dict[str, float]] = {}
    for period in ["overall", "1_8", "9_16", "17_24"]:
        if period == "overall":
            sp, st = sub["y_pred"].to_numpy(), sub["y_true"].to_numpy()
        else:
            mask = sub["period"] == period
            sp = sub.loc[mask, "y_pred"].to_numpy()
            st = sub.loc[mask, "y_true"].to_numpy()
        if len(sp) == 0:
            out[period] = {"n": 0, "smape": float("nan"), "mae": float("nan")}
            continue
        out[period] = {
            "n": int(len(sp)),
            "smape": smape(sp, st),
            "mae": float(np.mean(np.abs(sp - st))),
        }
    return out


def run_one(test_date: str, output_dir: str, *, epochs: int = 40) -> dict[str, object]:
    out = Path(output_dir)
    if (out / "predictions_raw.csv").exists() and (out / "metrics_by_period.csv").exists():
        print(f"[skip] {output_dir} already has results")
    else:
        cfg = make_v15_cfg(test_date, output_dir, epochs=epochs)
        t0 = time.time()
        run_monthly_reproduction(cfg)
        elapsed = time.time() - t0
        print(f"[time] {test_date} v15 run took {elapsed:.1f}s")
    return {
        "date": test_date,
        "da": smape_report(str(out / "predictions_raw.csv"), "da"),
        "rt": smape_report(str(out / "predictions_raw.csv"), "rt"),
    }


def compare(v15_result: dict[str, object], v14_csv: str) -> None:
    v14 = pd.read_csv(v14_csv, encoding="utf-8-sig")
    print(f"\n=== Compare v14 (CPU baseline) vs v15 (9-16 weighted) for {v15_result['date']} ===")
    for task, label in [("da", "DA"), ("rt", "RT")]:
        print(f"\n{label}:")
        print(f"  {'period':8s} {'v14_sMAPE':>10s} {'v15_sMAPE':>10s} {'v14_MAE':>10s} {'v15_MAE':>10s} {'v14_n':>5s} {'v15_n':>5s}")
        for period in ["overall", "1_8", "9_16", "17_24"]:
            v14_row = v14[(v14["task"] == task) & (v14["period"] == period)]
            if v14_row.empty:
                continue
            v14_s = float(v14_row["sMAPE"].iloc[0])
            v14_m = float(v14_row["MAE"].iloc[0])
            v14_n = int(v14_row["n"].iloc[0])
            v15 = v15_result[task][period]
            print(
                f"  {period:8s} {v14_s:>10.2f} {v15['smape']:>10.2f} "
                f"{v14_m:>10.2f} {v15['mae']:>10.2f} {v14_n:>5d} {v15['n']:>5d}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", default=["2026-06-15", "2026-06-17", "2026-06-18"])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--out-root", default="TimeMixer/outputs_v15_916w")
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []
    for d in args.dates:
        sub = out_root / f"single_{d}"
        sub.mkdir(parents=True, exist_ok=True)
        result = run_one(d, str(sub), epochs=args.epochs)
        compare(result, "TimeMixer/outputs_v14_168h_b6/metrics_by_period.csv")
        summary.append(result)
    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote summary to {out_root/'summary.json'}")


if __name__ == "__main__":
    main()
