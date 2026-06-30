#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inventory_prediction_sources.py — P0 Prediction Source Inventory

Audits all available model prediction sources for the P0 window (Nov 2025 – Feb 2026).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT))

INVENTORY: list[dict] = []


def register(model: str, field: str, value):
    entry = next((e for e in INVENTORY if e["model_name"] == model), None)
    if entry is None:
        entry = {"model_name": model}
        INVENTORY.append(entry)
    entry[field] = value


def check_path(path_str: str) -> bool:
    return Path(path_str).exists()


# ── LightGBM ────────────────────────────────────────────────────────
lgbm_rt = "models/LightGBM/best_model_实时电价.pkl"
lgbm_da = "models/LightGBM/best_model_日前电价.pkl"
register("lightgbm", "has_training_entry", True)
register("lightgbm", "has_inference_entry", True)
register("lightgbm", "has_checkpoint", check_path(lgbm_rt))
register("lightgbm", "checkpoint_path", lgbm_rt)
register("lightgbm", "can_run_realtime", True)
register("lightgbm", "inference_script", "lightGBM/infer_fix.py")
register("lightgbm", "estimated_runtime", "~5 min for 4 months")
register("lightgbm", "requires_gpu", False)
register("lightgbm", "recommended_for_p0", True)
register("lightgbm", "blocker", None)
register("lightgbm", "notes", "ThreeStageLGBM with valley/solar/peak sub-models. Uses hour 1-24 logic, '减1秒' feature engineering.")

# ── TimesFM ──────────────────────────────────────────────────────────
register("timesfm", "has_training_entry", False)  # Uses pre-trained model
register("timesfm", "has_inference_entry", check_path("TimesFM/infer.py"))
register("timesfm", "has_checkpoint", False)
register("timesfm", "checkpoint_path", None)
register("timesfm", "can_run_realtime", True)
register("timesfm", "inference_script", "TimesFM/infer.py")
register("timesfm", "estimated_runtime", "~30 min for 4 months (legacy module)")
register("timesfm", "requires_gpu", True)
register("timesfm", "recommended_for_p0", False)
register("timesfm", "blocker", "Requires legacy TF module not in standard repo path. May need GPU.")
register("timesfm", "notes", "TimesFM v2.5 modules exist but inference pipeline uses legacy price_forecast_copy_分时段预测.py")

# ── TimeMixer ────────────────────────────────────────────────────────
register("timemixer", "has_training_entry", True)
register("timemixer", "has_inference_entry", False)
register("timemixer", "has_checkpoint", False)
register("timemixer", "checkpoint_path", None)
register("timemixer", "can_run_realtime", False)
register("timemixer", "inference_script", None)
register("timemixer", "estimated_runtime", "Unknown")
register("timemixer", "requires_gpu", True)
register("timemixer", "recommended_for_p0", False)
register("timemixer", "blocker", "No checkpoint found. TimeMixer outputs_baseline has pre-computed predictions but only for non-P0 dates.")
register("timemixer", "notes", "TimeMixer model defined but no inference script. outputs_baseline/ has predictions_raw.csv but outside P0 window.")

# ── SGDFNet ──────────────────────────────────────────────────────────
register("sgdfnet", "has_training_entry", True)
register("sgdfnet", "has_inference_entry", True)
register("sgdfnet", "has_checkpoint", False)
register("sgdfnet", "checkpoint_path", None)
register("sgdfnet", "can_run_realtime", True)
register("sgdfnet", "inference_script", "SGDFNet/pipeline.py")
register("sgdfnet", "estimated_runtime", "~15 min (inference only)")
register("sgdfnet", "requires_gpu", True)
register("sgdfnet", "recommended_for_p0", False)
register("sgdfnet", "blocker", "No checkpoint found. Config files exist but need trained weights.")
register("sgdfnet", "notes", "Has pipeline.py and production_api.py. Configs in SGDFNet/configs/ but no .pth/.ckpt found.")

# ── RT916 ────────────────────────────────────────────────────────────
register("rt916", "has_training_entry", True)
register("rt916", "has_inference_entry", True)
register("rt916", "has_checkpoint", False)
register("rt916", "checkpoint_path", None)
register("rt916", "can_run_realtime", True)
register("rt916", "inference_script", "RT916_SpikeFusionNet/model.py")
register("rt916", "estimated_runtime", "~2 hours for 4 months (GPU needed)")
register("rt916", "requires_gpu", True)
register("rt916", "recommended_for_p0", False)
register("rt916", "blocker", "No checkpoint found. Full run prohibited by Runner rules. Only selective inference allowed (max 3 dates).")
register("rt916", "notes", "Outputs exist for May-Jun 2026 only (outside P0 window).")

# ── Fusion ──────────────────────────────────────────────────────────
register("fusion", "has_training_entry", True)
register("fusion", "has_inference_entry", False)
register("fusion", "has_checkpoint", False)
register("fusion", "checkpoint_path", None)
register("fusion", "can_run_realtime", False)
register("fusion", "inference_script", None)
register("fusion", "estimated_runtime", "N/A")
register("fusion", "requires_gpu", False)
register("fusion", "recommended_for_p0", False)
register("fusion", "blocker", "Fusion adapters read pre-computed predictions; no standalone inference. Requires model predictions first.")
register("fusion", "notes", "Fusion adapters normalize LightGBM, TimesFM, etc. outputs. Can generate base_fused_pred from multi-model predictions.")


def write_report():
    out_dir = Path("reports/local/p0_full_run/inventory")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = out_dir / "prediction_source_inventory.json"
    json_path.write_text(
        json.dumps(INVENTORY, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown
    lines = [
        "# P0 Prediction Source Inventory",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        "| Model | Checkpoint | Inference Script | Can Run P0? | GPU Needed | Est. Time | Blocker |",
        "|-------|-----------|-----------------|-------------|-----------|-----------|---------|",
    ]
    for e in INVENTORY:
        ckpt = "✅" if e.get("has_checkpoint") else "❌"
        inf = "✅" if e.get("has_inference_entry") else "❌"
        run = "✅" if e.get("recommended_for_p0") else "❌"
        gpu = "✅" if e.get("requires_gpu") else "❌"
        time = e.get("estimated_runtime", "?")
        blocker = e.get("blocker", "") or "—"
        lines.append(f"| {e['model_name']} | {ckpt} | {inf} | {run} | {gpu} | {time} | {blocker} |")

    lines += [
        "",
        "## Details",
        "",
    ]
    for e in INVENTORY:
        lines.append(f"### {e['model_name']}")
        lines.append("")
        for k, v in e.items():
            if k == "model_name":
                continue
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    md_path = out_dir / "prediction_source_inventory.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Inventory written to {out_dir}")
    print(f"  JSON: {json_path.name}")
    print(f"  MD:   {md_path.name}")
    print()
    print("Models with checkpoints ready for P0 inference:")
    for e in INVENTORY:
        if e.get("has_checkpoint") and e.get("recommended_for_p0"):
            print(f"  ✅ {e['model_name']} — {e.get('inference_script', '?')}")


if __name__ == "__main__":
    write_report()
