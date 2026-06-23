"""Quick baseline test for TimeMixer on specific dates."""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from TimeMixer.repro_pipeline import RunConfig, run_monthly_reproduction, smape

def test_date(date_str, target="realtime"):
    month = pd.Timestamp(date_str).strftime("%Y-%m")
    cfg = RunConfig(
        data_path=r"D:\作业\大创_挑战杯_互联网\大学生创新创业计划\大创实现\其他资料\epf\data\shandong_pmos_hourly.csv",
        output_dir=f"TimeMixer/outputs_quick_{date_str}",
        month=month,
        test_start=date_str,
        test_end_exclusive=(pd.Timestamp(date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        append_leaderboard=False,
    )
    result = run_monthly_reproduction(cfg)
    
    raw = pd.read_csv(f"TimeMixer/outputs_quick_{date_str}/predictions_raw.csv", encoding="utf-8-sig")
    
    task_filter = "da" if target == "dayahead" else "rt"
    filtered = raw[raw["task"] == task_filter].dropna(subset=["y_true", "y_pred"])
    
    if filtered.empty:
        print(f"  {date_str} {target}: NO VALID PREDICTIONS")
        return
    
    pred = filtered["y_pred"].to_numpy(float)
    true = filtered["y_true"].to_numpy(float)
    
    print(f"  {date_str} {target}:")
    for period in ["overall", "1_8", "9_16", "17_24"]:
        if period == "overall":
            sub_pred, sub_true = pred, true
        else:
            mask = filtered["period"] == period
            sub_pred = pred[mask]
            sub_true = true[mask]
        if len(sub_pred) == 0:
            continue
        s = smape(sub_pred, sub_true)
        mae = float(np.mean(np.abs(sub_pred - sub_true)))
        print(f"    {period:8s} n={len(sub_pred):3d}  MAE={mae:.2f}  SMAPE={s:.2f}%  Acc={100-s:.2f}%")

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "realtime"
    dates = ["2026-06-17", "2026-06-18", "2026-06-19"]
    print(f"=== TimeMixer Baseline: {target} ===")
    for d in dates:
        try:
            test_date(d, target)
        except Exception as e:
            print(f"  {d} {target}: FAILED - {e}")
