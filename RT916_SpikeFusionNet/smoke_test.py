"""Smoke test - just run baseline training for 1 epoch to verify pipeline."""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))
sys.path.insert(0, os.path.dirname(HERE))

os.environ["OPTIM_AMP"] = "0"
os.environ["OPTIM_NUM_WORKERS"] = "0"
os.environ["SPIKE_RT916_USE_V2"] = "0"
os.environ["SPIKE_RT916_DA_LINKAGE"] = "0"
os.environ["SPIKE_RT916_SMAPE_LOSS"] = "0"

import pandas as pd
import numpy as np

from rt916_spikefusionnet import core
from rt916_spikefusionnet.dataprocess import (
    enrich_selected_features,
    feature_engineer_solar_terms,
    process_features,
    split_excel_by_hours,
)

DATA = os.path.join(os.path.dirname(HERE), "data", "shandong_pmos_hourly.xlsx")
print(f"Data path: {DATA}")

core.RAW_DF_PATH = DATA
core.CONFIG["OUTPUT"] = "实时电价"
asof = pd.Timestamp("2026-06-18 15:00:00")
core._update_config("实时电价", [str(asof + pd.Timedelta(hours=9)), str(asof + pd.Timedelta(hours=16))])
core.CONFIG["EPOCHS"] = 1
core.CONFIG["BATCH_SIZE"] = 64
core.CONFIG["PATIENCE"] = 1
import pathlib
pathlib.Path(core.CONFIG["SAVE_ROOT_DIR"]).mkdir(parents=True, exist_ok=True)

df = pd.read_excel(DATA)
df = process_features(df)
df = feature_engineer_solar_terms(df)
df = enrich_selected_features(df, target_col="实时电价")
df["时刻"] = pd.to_datetime(df["时刻"])

start = asof - pd.DateOffset(months=2)
train = df[(df["时刻"] >= start) & (df["时刻"] <= asof)].copy()
_, df_9_16, _ = split_excel_by_hours(train)
print(f"9-16 train rows: {len(df_9_16)}, 范围: {df_9_16['时刻'].min()} ~ {df_9_16['时刻'].max()}")

t0 = time.time()
core.train_single_period("9-16点", df_9_16)
print(f"训练耗时: {time.time() - t0:.1f} 秒")
