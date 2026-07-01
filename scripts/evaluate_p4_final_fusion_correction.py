#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_p4_final_fusion_correction.py -- P4 Final Fusion + Correction.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from extreme.realtime_high_spike.apply_correction import CorrectionProfile, run_correction
from scripts.evaluate_realtime_spike_correction import compute_all_metrics
warnings.filterwarnings("ignore", category=FutureWarning)

PHASE2_CHAMPION = {"smape": 20.86, "severe": 63}
DEPLOY_GO = {"smape": 20.50, "severe": 63, "false_lift_rate": 0.10, "normal_hours_degradation": 0.50}

COMBOS = [
    {"key": "phase2_baseline", "label": "Phase2 Champion Baseline",          "risk_source": "phase2", "use_w2": False},
    {"key": "w2_phase2_risk",  "label": "W2 Quantile + Phase2 Correction",   "risk_source": "phase2", "use_w2": True},
    {"key": "phase2_w3_risk",  "label": "Phase2 Base + W3 ML Gate",          "risk_source": "w3",      "use_w2": False},
    {"key": "w2_w3_risk",      "label": "W2 Quantile + W3 ML Gate",          "risk_source": "w3",      "use_w2": True},
]
COMBO_KEYS = [c["key"] for c in COMBOS]
def load_canonical(path):
    df = pd.read_csv(path)
    required = {"business_day","hour_business","base_fused_pred","y_true",
                 "y_pred_lightgbm","y_pred_dayahead_proxy",
                 "y_pred_naive_lag1","y_pred_naive_lag7"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Canonical pack missing columns: {missing}")
    df["business_day"] = df["business_day"].astype(str)
    return df

def load_w2(path):
    df = pd.read_csv(path)
    if "pred_y" not in df.columns: raise ValueError("W2 missing pred_y")
    df["ds"] = pd.to_datetime(df["ds"])
    df["business_day"] = df["ds"].dt.strftime("%Y-%m-%d")
    df["hour_business"] = df["hour"].values
    r = df[["business_day","hour_business","pred_y"]].copy()
    r = r.drop_duplicates(subset=["business_day","hour_business"]).reset_index(drop=True)
    return r

def load_risk(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "high_spike_prob" not in df.columns: raise ValueError("Risk missing high_spike_prob")
    df["business_day"] = df["business_day"].astype(str)
    return df
def build_base_pred(combo, canonical, w2, keys):
    if not combo["use_w2"]:
        return keys.merge(canonical[["business_day","hour_business","base_fused_pred"]], on=["business_day","hour_business"], how="left")["base_fused_pred"]
    canonical["baseline_mean"] = (canonical["y_pred_dayahead_proxy"] + canonical["y_pred_naive_lag1"] + canonical["y_pred_naive_lag7"]) / 3.0
    merged = keys.merge(canonical[["business_day","hour_business","baseline_mean","y_pred_lightgbm"]], on=["business_day","hour_business"], how="left")
    merged = merged.merge(w2.rename(columns={"pred_y":"w2_pred"}), on=["business_day","hour_business"], how="left")
    merged["effective_lgbm"] = merged["w2_pred"].fillna(merged["y_pred_lightgbm"])
    cov = merged["w2_pred"].notna().sum(); tot = len(merged)
    if cov < tot: print(f"  [INFO] W2 coverage: {cov}/{tot} ({100*cov/tot:.1f}%)")
    merged["base_fused_pred"] = 0.9 * merged["effective_lgbm"] + 0.1 * merged["baseline_mean"]
    return merged["base_fused_pred"]

def write_pack(base_pred, y_true, keys, out_dir, label):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = keys[["business_day","hour_business"]].copy()
    p["base_fused_pred"] = base_pred.values; p["y_true"] = y_true.values
    path = out_dir / f"prediction_pack_{label}.csv"
    p.to_csv(path, index=False)
    print(f"  [INFO] Prediction pack: {path} ({len(p)} rows)")
    return path
def run_combo(ck, rpath, bp, yt, keys, prof, od):
    co = od / ck
    pp = write_pack(bp, yt, keys, co, ck)
    r = run_correction(prediction_pack_path=str(pp), risk_predictions_path=str(rpath), profile=prof)
    r.to_csv(co / "predictions.csv", index=False)
    if "hour_business" in r.columns:
        nb = len(r); r = r.drop_duplicates(subset=["business_day","hour_business"]).copy()
        if len(r) < nb: print(f"  [INFO] Dedup: {nb} -> {len(r)}")
    m = compute_all_metrics(r)
    m["combo"] = ck; m["n_timestamps"] = len(r)
    import json
    json.dump(m, open(co/"metrics.json","w",encoding="utf-8"), indent=2, ensure_ascii=False)
    _s = m.get("realtime_overall_smape_floor50")
    _se = m.get("severe_underestimate_count")
    _fl = m.get("false_lift_rate")
    _nd = m.get("normal_hours_degradation")
    print(f"  sMAPE={_fmt(_s)} severe={_se} flift={_fmt(_fl)} ndeg={_fmt(_nd)}")
    return m

def _fmt(v):
    if isinstance(v, float): return f"{v:.2f}"
    return str(v) if v is not None else "\u2014"

def assess(metrics):
    dc = {}
    for n, k, t in [
        ("sMAPE<=20.50", "realtime_overall_smape_floor50", DEPLOY_GO["smape"]),
        ("Severe<=63", "severe_underestimate_count", DEPLOY_GO["severe"]),
        ("False lift<=10%", "false_lift_rate", DEPLOY_GO["false_lift_rate"]),
        ("Normal degradation<=0.5", "normal_hours_degradation", DEPLOY_GO["normal_hours_degradation"]),
    ]:
        a = metrics.get(k)
        dc[n] = {"threshold": t, "actual": a, "met": a is not None and a <= t}
    adm = all(c["met"] for c in dc.values())
    si = PHASE2_CHAMPION["smape"] - metrics.get("realtime_overall_smape_floor50", 999) if metrics.get("realtime_overall_smape_floor50") is not None else -999
    sei = PHASE2_CHAMPION["severe"] - metrics.get("severe_underestimate_count", 999) if metrics.get("severe_underestimate_count") is not None else -999
    pg = si > 0 and sei >= 0
    v = "DEPLOY GO" if adm else ("PAPER GO" if pg else "NO-GO")
    return {"verdict": v, "all_deploy_criteria_met": adm, "paper_go": pg, "deploy_criteria": dc,
            "smape_improvement_vs_phase2": round(si, 2), "severe_improvement_vs_phase2": sei}

def build_table(am):
    h = "| Combo | sMAPE | Base sMAPE | Severe | High-spike MAE | False Lift | Normal Degrad | N | Verdict |"
    s = "|------|:-----:|:----------:|:------:|:--------------:|:----------:|:-------------:|:---:|:-------:|"
    r = [h, s]
    for ck in COMBO_KEYS:
        if ck not in am: continue
        m = am[ck]
        r.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            ck, _fmt(m.get("realtime_overall_smape_floor50")),
            _fmt(m.get("realtime_base_smape_floor50")),
            m.get("severe_underestimate_count", "\u2014"),
            _fmt(m.get("high_spike_mae")),
            _fmt(m.get("false_lift_rate")),
            _fmt(m.get("normal_hours_degradation")),
            m.get("n_timestamps", "\u2014"),
            m.get("_verdict", {}).get("verdict", "\u2014"),
        ))
    r.append("| phase2_champion (ref) | {} | \u2014 | {} | \u2014 | \u2014 | \u2014 | \u2014 | \u2014 |".format(
        PHASE2_CHAMPION["smape"], PHASE2_CHAMPION["severe"]))
    return "\n".join(r)

def main():
    import sys as _sys
    a = _sys.argv[1:]
    d = {}
    i = 0
    while i < len(a):
        if a[i].startswith("--"):
            k = a[i][2:]
            if i + 1 < len(a) and not a[i + 1].startswith("--"):
                d[k] = a[i + 1]; i += 2
            else:
                d[k] = True; i += 1
        else:
            i += 1
    cp = Path(d.get("canonical-pack", ""))
    w2p = Path(d.get("window2-csv", "")) if d.get("window2-csv") else None
    p2r = Path(d.get("risk-predictions", ""))
    w3r = Path(d.get("w3-risk", "")) if d.get("w3-risk") else p2r
    od = Path(d.get("out-dir", "reports/local/p4_final_fusion_correction"))
    for p in [cp, p2r, w3r]:
        if not p.exists(): _sys.exit(f"Error: {p} not found")
    od.mkdir(parents=True, exist_ok=True)

    print("\n" + 60 * "=")
    print("  P4 Final Fusion + Correction")
    print(60 * "=")
    canonical = load_canonical(cp)
    w2 = load_w2(w2p) if w2p and w2p.exists() else None
    p2r_df = load_risk(p2r); w3r_df = load_risk(w3r)
    fk = canonical[["business_day", "hour_business"]].drop_duplicates().reset_index(drop=True)
    fyt = canonical[["business_day", "hour_business", "y_true"]].drop_duplicates(subset=["business_day", "hour_business"])["y_true"].reset_index(drop=True)
    print(f"  Full period: {len(fk)} timestamps")
    if w2 is not None:
        wk = fk.merge(w2[["business_day", "hour_business"]], on=["business_day", "hour_business"], how="inner")
        wyt = fk.merge(canonical[["business_day", "hour_business", "y_true"]], on=["business_day", "hour_business"], how="left")
        wyt = wk.merge(wyt, on=["business_day", "hour_business"], how="left")["y_true"].reset_index(drop=True)
        print(f"  W2 coverage:   {len(wk)} timestamps ({w2['business_day'].nunique()} days)")
    else:
        wk = wyt = None

    prof = CorrectionProfile(name="medium", spike_prob_threshold=0.60, max_lift_ratio=0.35,
                              max_absolute_lift=350.0, protect_normal_hours=True, period_9_16_boost=1.15)
    ctr = [c for c in COMBOS if not c["use_w2"] or w2 is not None]
    am = {}
    ts = time.time()
    for c in ctr:
        ck = c["key"]; rp = w3r if c["risk_source"] == "w3" else p2r
        keys, yt, pd_ = (wk, wyt, "W2 period") if (c["use_w2"] and wk is not None) else (fk, fyt, "full period")
        print(f"\n  {'-' * 50}")
        print(f"  {c['label']} [{pd_}]")
        print(f"  {'-' * 50}")
        bp = build_base_pred(c, canonical, w2, keys)
        m = run_combo(ck, rp, bp, yt, keys, prof, od)
        v = assess(m); m["_verdict"] = v; m["_period"] = pd_; am[ck] = m
        print(f"  >> {v['verdict']}")

    print(f"\n{'=' * 60}\n  Comparison\n{'=' * 60}")
    tbl = build_table(am); print("\n" + tbl + "\n")

    (od / "comparison_table.md").write_text(
        "# P4 Final Fusion + Correction - Comparison\n\n"
        "DEPLOY GO: sMAPE <= {}, severe <= {}, false_lift <= {}, normal_degradation <= {}\n\n{}\n".format(
            DEPLOY_GO["smape"], DEPLOY_GO["severe"], DEPLOY_GO["false_lift_rate"],
            DEPLOY_GO["normal_hours_degradation"], tbl), encoding="utf-8")

    summary = {
        "script": "scripts/evaluate_p4_final_fusion_correction.py",
        "canonical_pack": str(cp), "window2_csv": str(w2p) if w2p else None,
        "phase2_risk": str(p2r), "w3_risk": str(w3r),
        "profile": {"name": "medium", "mode": "normal"},
        "deploy_go_thresholds": DEPLOY_GO, "phase2_champion_ref": PHASE2_CHAMPION,
        "combos": {k: {"metrics": {kk: vv for kk, vv in am[k].items() if not kk.startswith("_")},
                        "verdict": am[k].get("_verdict")} for k in am},
        "total_runtime_seconds": round(time.time() - ts, 1),
    }
    json.dump(summary, open(od / "comparison_summary.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"  Summary: {od / 'comparison_summary.json'}")
    print(f"  Runtime: {time.time() - ts:.0f}s")
    print("Done.")

if __name__ == "__main__":
    main()
