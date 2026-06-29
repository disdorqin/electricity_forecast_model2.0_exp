#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature Audit for Shandong electricity price data."""
from __future__ import annotations
import argparse, sys, numpy as np, pandas as pd
from pathlib import Path

TARGET_DAYAHEAD = "日前电价"
TARGET_REALTIME = "实时电价"
TARGET_SPREAD = "价差(实时-日前)"

FORECAST_FEATURES = ["地方电厂总加预测值","联络线受电负荷预测值","风电总加预测值",
    "光伏总加预测值","核电总加预测值","自备机组总加预测值",
    "试验机组总加预测值","直调负荷预测值","竞价空间预测值",
    "新能源总加预测值"]
ACTUAL_FEATURES = ["地方电厂总加实际值","联络线受电负荷实际值","风电总加实际值",
    "光伏总加实际值","核电总加实际值","自备机组总加实际值",
    "试验机组总加实际值","直调负荷实际值","竞价空间实际值",
    "新能源总加实际值"]
ALL_FEATURES = FORECAST_FEATURES + ACTUAL_FEATURES
ALL_COLUMNS = [TARGET_DAYAHEAD, TARGET_REALTIME] + ALL_FEATURES
PERIOD_MAP = {"1_8": range(1, 9), "9_16": range(9, 17), "17_24": range(17, 25)}
PERIOD_LABELS = ["1_8", "9_16", "17_24"]
CV_THRESHOLD = 0.01

def load_data(data_path):
    data_path = Path(data_path)
    if not data_path.exists(): raise FileNotFoundError(f"Data file not found: {data_path}")
    for enc in ["gbk", "utf-8", "gb18030", "latin-1"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            if TARGET_DAYAHEAD in df.columns or "时刻" in df.columns:
                print(f"[INFO] Loaded with encoding: {enc}"); return df
        except: continue
    if data_path.suffix.lower() in (".xlsx", ".xls"): return pd.read_excel(data_path)
    raise RuntimeError(f"Cannot read: {data_path}")

def standardize_timestamp(df):
    out = df.copy()
    out["时刻"] = pd.to_datetime(out["时刻"], errors="coerce")
    out = out.dropna(subset=["时刻"])
    out["hour"] = out["时刻"].dt.hour
    out["hour_business"] = out["hour"].apply(lambda h: 24 if h == 0 else h)
    out["date"] = out["时刻"].dt.date
    out["business_day"] = out.apply(lambda r: (r["时刻"]-pd.Timedelta(days=1)).date() if r["hour"]==0 else r["date"], axis=1)
    def _ap(hb):
        for lb, rng in PERIOD_MAP.items():
            if hb in rng: return lb
        return "unknown"
    out["period"] = out["hour_business"].apply(_ap)
    return out

def audit_schema(df):
    n = len(df); schema = {}
    skip = {"时刻","hour","hour_business","date","business_day","period","year_month"}
    for col in df.columns:
        if col in skip: continue
        nn = df[col].notna().sum(); mr = 1.0-nn/n if n>0 else 0.0
        nu = int(df[col].nunique()); ur = nu/n if n>0 else 0.0
        if pd.api.types.is_numeric_dtype(df[col]):
            cd = pd.to_numeric(df[col], errors="coerce")
            std = float(cd.std()) if len(cd)>1 else 0.0
            mn = float(cd.mean()) if len(cd)>0 else 0.0
            cv = (std/abs(mn)) if abs(mn)>1e-10 else float("inf")
            zc = int((cd==0).sum()); zr = zc/n if n>0 else 0.0; lv = cv<CV_THRESHOLD
            schema[col] = {"dtype":str(df[col].dtype),"n_total":n,"n_non_null":int(nn),
                "missing_rate":mr,"n_unique":nu,"unique_ratio":ur,
                "mean":mn,"std":std,"cv":cv,
                "min":float(cd.min()),"q25":float(cd.quantile(0.25)),
                "q50":float(cd.median()),"q75":float(cd.quantile(0.75)),
                "max":float(cd.max()),"zero_ratio":zr,"low_variance":lv}
        else:
            schema[col] = {"dtype":str(df[col].dtype),"n_total":n,"n_non_null":int(nn),
                "missing_rate":mr,"n_unique":nu,"unique_ratio":ur,
                "mean":float("nan"),"std":float("nan"),"cv":float("nan"),
                "min":float("nan"),"q25":float("nan"),"q50":float("nan"),
                "q75":float("nan"),"max":float("nan"),"zero_ratio":float("nan"),"low_variance":False}
    return schema

def audit_period(df):
    recs = []; nc = [c for c in ALL_COLUMNS if c in df.columns]
    for pl in PERIOD_LABELS:
        sub = df.loc[df["period"]==pl]
        if len(sub)==0: continue
        for col in nc:
            cd = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(cd)==0: continue
            recs.append({"period":pl,"field":col,"n":len(cd),
                "mean":float(cd.mean()),"std":float(cd.std()),
                "min":float(cd.min()),"q25":float(cd.quantile(0.25)),
                "q50":float(cd.median()),"q75":float(cd.quantile(0.75)),
                "max":float(cd.max())})
    return pd.DataFrame(recs)

def audit_relation(df):
    out = df.copy()
    if TARGET_REALTIME in out.columns and TARGET_DAYAHEAD in out.columns:
        out[TARGET_SPREAD] = pd.to_numeric(out[TARGET_REALTIME],errors="coerce")-pd.to_numeric(out[TARGET_DAYAHEAD],errors="coerce")
    tgts = [t for t in [TARGET_DAYAHEAD,TARGET_REALTIME,TARGET_SPREAD] if t in out.columns]
    nc = [c for c in ALL_FEATURES if c in out.columns]; recs = []
    for feat in nc:
        for tgt in tgts:
            pair = out[[feat,tgt]].dropna()
            if len(pair)<10:
                recs.append({"feature":feat,"target":tgt,"n_pairs":len(pair),"pearson_r":float("nan"),"spearman_rho":float("nan")})
            else:
                recs.append({"feature":feat,"target":tgt,"n_pairs":len(pair),
                    "pearson_r":float(pair[feat].corr(pair[tgt])),
                    "spearman_rho":float(pair[feat].corr(pair[tgt],method="spearman"))})
    return pd.DataFrame(recs)

def audit_leakage(df):
    NL = chr(10); rows = []
    rows.append("# Leakage Risk Report"+NL+NL)
    rows.append("## Scenario: D-day predict D+1"+NL+NL)
    rows.append("- Dayahead: D+1 forecast features known on D"+NL)
    rows.append("- Realtime: only data before 14:00 on D allowed"+NL)
    rows.append("- Actual columns: D+1 actuals unknown"+NL+NL)
    rows.append("## Fully available"+NL+NL+"| Field | Note |"+NL+"|-------|------|"+NL)
    for f in FORECAST_FEATURES: rows.append(f"| {f} | D+1 forecast known on D |"+NL)
    rows.append(NL+"## Historical only"+NL+NL+"| Field | Note |"+NL+"|-------|------|"+NL)
    for f in ACTUAL_FEATURES: rows.append(f"| {f} | Historical actual only; D+1 unknown |"+NL)
    rows.append(NL+"## Look-ahead risk"+NL+NL+"| Field | Risk |"+NL+"|-------|------|"+NL)
    rows.append("| Realtime price | Only data before 14:00; post-14:00 causes look-ahead |"+NL)
    for f in ACTUAL_FEATURES: rows.append(f"| {f} | Same-day actuals cause look-ahead |"+NL)
    rows.append(NL+"## Forecast vs Actual Consistency"+NL+NL)
    rows.append("| Forecast | Actual | Identical% | Verdict |"+NL+"|----------|--------|------------|---------|"+NL)
    for p,a in zip(FORECAST_FEATURES,ACTUAL_FEATURES):
        if p in df.columns and a in df.columns:
            pd_ = pd.to_numeric(df[p],errors="coerce"); ad_ = pd.to_numeric(df[a],errors="coerce")
            both = pd_.notna() & ad_.notna()
            if both.sum()>0:
                r = int((pd_[both]==ad_[both]).sum())/both.sum()
                rs = f"{r:.2%}"; v = "HIGH" if r>0.5 else ("MED" if r>0.1 else "OK")
            else: rs,v = "N/A","-"
            rows.append(f"| {p} | {a} | {rs} | {v} |"+NL)
    rows.append(NL+"## Realtime cutoff"+NL+NL)
    if TARGET_REALTIME in df.columns:
        ha = df.groupby("hour")[TARGET_REALTIME].apply(lambda x: x.notna().sum())
        rows.append("| Hour | Count | Available |"+NL+"|------|-------|----------|"+NL)
        for hr in range(24):
            av = "Yes" if hr<=14 else "No (post 14:00)"
            rows.append(f"| {hr:2d}:00 | {ha.get(hr,0)} | {av} |"+NL)
    return "".join(rows)

def write_schema_report(schema, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    NL = chr(10); Ls = []
    Ls.append("# Feature Schema Report"+NL+NL)
    now_str = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    Ls.append("Generated: "+now_str+NL+NL)
    Ls.append("Total fields: "+str(len(schema))+NL+NL)
    dts = {}
    for c,i in schema.items(): dts.setdefault(i["dtype"],[]).append(c)
    Ls.append("## By Data Type"+NL+NL+"| Type | Count | Fields |"+NL+"|------|-------|--------|"+NL)
    for dt in sorted(dts.keys()):
        Ls.append("| "+dt+" | "+str(len(dts[dt]))+" | "+", ".join(dts[dt])+" |"+NL)
    Ls.append(NL+"## Field Details"+NL+NL)
    Ls.append("| Field | Type | NonNull | Miss% | Unique | Uniq% | Mean | Std | CV | Zero% | LowVar |"+NL)
    Ls.append("|-------|------|---------|-------|--------|-------|------|-----|-------|--------|"+NL)
    for c in sorted(schema.keys()):
        i = schema[c]
        miss_s = f"{i['missing_rate']:.2%}"
        uniq_s = f"{i['unique_ratio']:.2%}"
        mean_s = f"{i['mean']:.2f}"
        std_s = f"{i['std']:.2f}"
        cv_s = f"{i['cv']:.4f}"
        zero_s = f"{i['zero_ratio']:.2%}"
        lv_s = "YES" if i['low_variance'] else "no"
        parts = [c,str(i['dtype']),str(i['n_non_null']),miss_s,
                 str(i['n_unique']),uniq_s,mean_s,std_s,cv_s,zero_s,lv_s]
        Ls.append("|".join([""]+parts+[""])+"|"+NL)
    lv = [c for c,i in schema.items() if i.get("low_variance")]
    if lv:
        Ls.append(NL+"## Low Variance Fields"+NL+NL)
        for c in lv:
            Ls.append("- **"+c+"**: CV="+f"{schema[c]['cv']:.4f}"+", mean="+f"{schema[c]['mean']:.2f}"+NL)
    Path(path).write_text("".join(Ls), encoding="utf-8")
    print(f"[OUTPUT] Schema report: {path}")

def write_audit_csv(schema, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    for c,i in schema.items():
        recs.append({"field":c,"dtype":i["dtype"],"n_total":i["n_total"],
            "n_non_null":i["n_non_null"],"missing_rate":i["missing_rate"],
            "n_unique":i["n_unique"],"unique_ratio":i["unique_ratio"],
            "mean":i["mean"],"std":i["std"],"cv":i["cv"],
            "min":i["min"],"q25":i["q25"],"q50":i["q50"],
            "q75":i["q75"],"max":i["max"],"zero_ratio":i["zero_ratio"],
            "low_variance_flag":i["low_variance"]})
    pd.DataFrame(recs).to_csv(path,index=False,encoding="utf-8-sig")
    print(f"[OUTPUT] Audit CSV: {path}")

def write_period_report(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if len(df)>0: df.to_csv(path,index=False,encoding="utf-8-sig")
    print(f"[OUTPUT] Period report: {path}")

def write_relation_report(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if len(df)>0: df.to_csv(path,index=False,encoding="utf-8-sig")
    print(f"[OUTPUT] Relation report: {path}")

def write_leakage_report(report, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    print(f"[OUTPUT] Leakage report: {path}")

def analyze_special_months(df, schema, out_dir):
    df["year_month"] = df["时刻"].dt.strftime("%Y-%m")
    for month in ["2025-11", "2025-12"]:
        sub = df.loc[df["year_month"]==month]
        if len(sub)==0: print(f"[SKIP] {month} no data"); continue
        NL = chr(10); print(NL+"="*60)
        print(f"Special: {month} ({len(sub)} rows)"); print("="*60)
        md = Path(out_dir)/month; md.mkdir(parents=True, exist_ok=True)
        miss = []
        for c in ALL_COLUMNS:
            if c not in sub.columns: continue
            mc = int(sub[c].isna().sum())
            if mc > 0:
                miss.append({"field":c,"missing_count":mc,"missing_rate":mc/len(sub)})
                print(f"  [MISS] {c}: {mc}/{len(sub)} ({mc/len(sub):.2%})")
        if miss: pd.DataFrame(miss).to_csv(md/"missing_detail.csv",index=False,encoding="utf-8-sig")
        else: print("  No missing values")
        outl = []
        for c in ALL_COLUMNS:
            if c not in sub.columns: continue
            cd = pd.to_numeric(sub[c],errors="coerce").dropna()
            if len(cd)<10: continue
            z = np.abs((cd-cd.mean())/cd.std())
            n_out = int((z>3).sum())
            if n_out>0:
                outl.append({"field":c,"n_samples":len(cd),"n_outliers":n_out,"outlier_ratio":n_out/len(cd)})
                print(f"  [OUTLIER] {c}: {n_out} ({n_out/len(cd):.2%})")
        if outl: pd.DataFrame(outl).to_csv(md/"outlier_detail.csv",index=False,encoding="utf-8-sig")
        stats = []
        for c in ALL_COLUMNS:
            if c not in sub.columns: continue
            cd = pd.to_numeric(sub[c],errors="coerce").dropna()
            if len(cd)==0: continue
            stats.append({"field":c,"n":len(cd),"mean":float(cd.mean()),
                "std":float(cd.std()),"min":float(cd.min()),
                "q50":float(cd.median()),"max":float(cd.max())})
        if stats: pd.DataFrame(stats).to_csv(md/"monthly_stats.csv",index=False,encoding="utf-8-sig")
        print(NL+f"  {month} key means vs full:")
        for c in [TARGET_DAYAHEAD,TARGET_REALTIME]+FORECAST_FEATURES[:5]:
            if c not in sub.columns or c not in df.columns: continue
            sd = pd.to_numeric(sub[c],errors="coerce").dropna()
            fd = pd.to_numeric(df[c],errors="coerce").dropna()
            if len(sd)>0 and len(fd)>0:
                d = sd.mean()-fd.mean()
                print(f"    {c}: month={sd.mean():.2f} full={fd.mean():.2f} diff={d:.2f}")

def run_audit(data_path="data/shandong_pmos_hourly.csv",target="both",
              start_date=None,end_date=None,out_dir="reports/local/feature_audit"):
    print("="*60); print("FEATURE AUDIT - Shandong Electricity Price"); print("="*60)
    NL = chr(10)
    print(NL+"[1/6] Loading: "+data_path)
    df = load_data(data_path)
    print(f"  Rows: {len(df)}, Cols: {len(df.columns)}")
    print(NL+"[2/6] Standardizing timestamps")
    df = standardize_timestamp(df)
    print(f"  Processed: {len(df)} rows")
    tm_col = "时刻"
    print(f"  Range: {df[tm_col].min()} ~ {df[tm_col].max()}")
    if start_date: df = df[df[tm_col]>=pd.Timestamp(start_date)]
    if end_date: df = df[df[tm_col]<pd.Timestamp(end_date)+pd.Timedelta(days=1)]
    print(f"  After filter: {len(df)} rows")
    tc = []
    if target in ("dayahead","both"): tc.append(TARGET_DAYAHEAD)
    if target in ("realtime","both"): tc.append(TARGET_REALTIME)
    print(f"  Targets: {chr(44).join(tc)}")
    op = Path(out_dir)
    print(NL+"[3/6] Schema audit")
    schema = audit_schema(df)
    write_schema_report(schema,op/"feature_schema_report.md")
    write_audit_csv(schema,op/"feature_audit_report.csv")
    print(NL+"[4/6] Period statistics")
    write_period_report(audit_period(df),op/"feature_period_report.csv")
    print(NL+"[5/6] Correlation analysis")
    write_relation_report(audit_relation(df),op/"feature_relation_report.csv")
    print(NL+"[6/6] Leakage analysis")
    write_leakage_report(audit_leakage(df),op/"leakage_risk_report.md")
    print(NL+"[Bonus] Special months analysis")
    analyze_special_months(df,schema,op/"special_months")
    print(NL+"="*60+NL+"AUDIT COMPLETE"+NL+"="*60)
    print(NL+f"Output: {op.resolve()}")
    return {"schema_report":op/"feature_schema_report.md",
        "audit_csv":op/"feature_audit_report.csv",
        "period_report":op/"feature_period_report.csv",
        "relation_report":op/"feature_relation_report.csv",
        "leakage_report":op/"leakage_risk_report.md"}

def build_cli():
    p = argparse.ArgumentParser(description="Feature audit for Shandong electricity price")
    p.add_argument("--data-path",default="data/shandong_pmos_hourly.csv")
    p.add_argument("--target",default="both",choices=["dayahead","realtime","both"])
    p.add_argument("--start-date",default=None)
    p.add_argument("--end-date",default=None)
    p.add_argument("--out-dir",default="reports/local/feature_audit")
    return p

def main():
    args = build_cli().parse_args()
    run_audit(data_path=args.data_path,target=args.target,start_date=args.start_date,end_date=args.end_date,out_dir=args.out_dir)
    return 0

if __name__=="__main__":
    sys.exit(main())