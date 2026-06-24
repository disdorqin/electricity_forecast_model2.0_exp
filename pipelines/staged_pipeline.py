from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import joblib

from fusion.classifier_bridge import run_classifier_pipeline
from fusion.contracts import infer_period, standardize_prediction_table
from fusion.run_fixed_window_fusion import _apply_fixed_weights
from fusion.meta_learner_v3 import (
    augment_long_table_with_extras,
    fit_meta_learners_from_long_table,
    apply_meta_learners,
)
from fusion.dynamic_weights import (
    apply_dynamic_weights,
    evaluate_dynamic_weights,
    fit_dynamic_weights,
)
from fusion.spike_detector import SpikeDetector, SpikeDetectorConfig
from fusion.weights import fit_weights_from_long_table
from fusion.metrics import smape_floor50
from runners.registry import get_model_pipeline
from utils.daily_run_layout import build_daily_run_layout


RT916_MODEL_KEY = "rt916"
DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
REALTIME_MODELS = ["timesfm", "timemixer", "sgdfnet"]
FORMAL_DAYAHEAD_MODELS = ["lightgbm", "timesfm", "timemixer"]
FORMAL_REALTIME_MODELS = ["timesfm", "timemixer", "sgdfnet", RT916_MODEL_KEY]
logger = logging.getLogger(__name__)


def _resolve_run_date(args) -> str:
    run_date = args.date or args.start
    if not run_date:
        raise ValueError("staged pipelines require --date or --start")
    return pd.Timestamp(run_date).strftime("%Y-%m-%d")


def _resolve_daily_runs_root(args) -> Path:
    root = getattr(args, "daily_run_root", None) or "daily_runs"
    return Path(root)


def _resolve_models_for_target(target: str, stage_models: str) -> list[str]:
    raw = (stage_models or "formal").strip().lower()
    if raw == "formal":
        if target == "dayahead":
            return FORMAL_DAYAHEAD_MODELS.copy()
        if target == "realtime":
            return FORMAL_REALTIME_MODELS.copy()
    if raw == "all":
        if target == "dayahead":
            return DAYAHEAD_MODELS.copy()
        if target == "realtime":
            return REALTIME_MODELS.copy() + [RT916_MODEL_KEY]
    explicit = [item.strip().lower() for item in stage_models.split(",") if item.strip()]
    if explicit:
        return explicit
    if target == "dayahead":
        return DAYAHEAD_MODELS.copy()
    if target == "realtime":
        return REALTIME_MODELS.copy() + [RT916_MODEL_KEY]
    raise ValueError(f"Unsupported staged target: {target}")


def _target_column(target: str) -> str:
    if target == "dayahead":
        return "日前电价"
    if target == "realtime":
        return "实时电价"
    raise ValueError(f"Unsupported target: {target}")


def _read_truth_frame(data_path: str, target: str) -> pd.DataFrame:
    raw = pd.read_excel(data_path, engine="openpyxl")
    target_col = _target_column(target)
    if "时刻" not in raw.columns or target_col not in raw.columns:
        raise ValueError(f"Dataset missing required columns for {target}: 时刻, {target_col}")
    truth = raw[["时刻", target_col]].copy()
    truth["时刻"] = pd.to_datetime(truth["时刻"], errors="coerce")
    truth["真实值"] = pd.to_numeric(truth[target_col], errors="coerce")
    truth = truth.drop(columns=[target_col]).dropna(subset=["时刻", "真实值"])
    return truth.drop_duplicates(subset=["时刻"], keep="last").sort_values("时刻").reset_index(drop=True)


def _prediction_frame_from_result(result, model_name: str, target: str, run_date: str, source: str) -> pd.DataFrame:
    frame = result.frame.copy()
    ts_col = frame.columns[0]
    pred_col = frame.columns[1]
    out = pd.DataFrame(
        {
            "时刻": pd.to_datetime(frame[ts_col], errors="coerce"),
            "预测值": pd.to_numeric(frame[pred_col], errors="coerce"),
            "model_name": model_name,
            "target": target,
            "run_date": run_date,
            "source": source,
        }
    )
    return out.dropna(subset=["时刻", "预测值"]).sort_values("时刻").reset_index(drop=True)


def _attach_truth(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(truth_df, on="时刻", how="left")
    return merged.dropna(subset=["真实值"]).reset_index(drop=True)


def _attach_optional_truth(pred_df: pd.DataFrame, truth_df: pd.DataFrame) -> pd.DataFrame:
    merged = pred_df.merge(truth_df, on="时刻", how="left")
    return merged.reset_index(drop=True)


def _model_dir(layout, model_name: str) -> Path:
    model_dir = layout.model_outputs_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def _run_model_for_range(
    model_name: str, *, target: str, start: str, end: str,
    predict_date: str, args
):
    """Run a model's predict_range with explicit start/end for the date range.

    predict_date controls the training-window reference point.
    start/end control the prediction output range.
    """
    pipeline = get_model_pipeline(model_name)
    kwargs = {
        "target": target,
        "predict_date": predict_date,
        "start": start,
        "end": end,
        "data_path": args.data_path,
        "output_root": args.output_root,
        "training_months": args.training_months,
        "val_ratio": args.val_ratio,
        "use_predicted_temp": args.use_predicted_temp,
        "segment_count": args.segment_count,
        "seed": args.seed,
        "deterministic": args.deterministic,
    }
    return pipeline.predict_range(**kwargs)


def run_model_stage(args):
    """Efficient model stage: train once, predict for validation period + forecast day."""
    run_date = _resolve_run_date(args)
    targets = ["dayahead", "realtime"] if args.target == "both" else [args.target]
    root = _resolve_daily_runs_root(args)
    outputs: list[str] = []

    # ── Compute the validation window ──
    # validation-days controls the validation period length (default 30 days).
    # 720 hourly rows for stable SLSQP weight fitting,
    # DA training ~11 months, prediction horizon 1-30 days (acceptable for cyclical prices).
    run_ts = pd.Timestamp(run_date)
    training_months = int(getattr(args, "training_months", 12))
    val_days = max(int(getattr(args, "validation_days", 30) or 30), 1)
    val_start = run_ts - pd.Timedelta(days=val_days)
    val_end = run_ts - pd.Timedelta(days=1)          # last validation day

    val_start_str = val_start.strftime("%Y-%m-%d")
    val_end_str = val_end.strftime("%Y-%m-%d")
    forecast_str = run_ts.strftime("%Y-%m-%d")

    logger.info(
        "Val period: %s → %s (%d days) | Forecast: %s | DA training ~%d months",
        val_start_str, val_end_str, val_days, forecast_str, training_months,
    )

    # Track model execution status for summary reporting
    model_status: dict[str, str] = {}  # "{target}/{model}" -> "ok" | "skip" | "FAIL: reason"

    for target in targets:
        layout = build_daily_run_layout(root, run_date, target)
        truth_df = _read_truth_frame(args.data_path, target)
        requested_models = _resolve_models_for_target(target, getattr(args, "stage_models", "formal"))
        for model_name in requested_models:
            model_key = f"{target}/{model_name}"
            model_dir = _model_dir(layout, model_name)

            # ── Skip if outputs already exist (resume support) ──
            _forecast_csv = model_dir / "forecast_predictions.csv"
            _val_csv = model_dir / "val_predictions.csv"
            if _forecast_csv.exists() and _val_csv.exists():
                # Verify the files are non-empty (guard against empty files from prior failures)
                try:
                    _fc_head = pd.read_csv(_forecast_csv, nrows=1, encoding="utf-8-sig")
                    _vl_head = pd.read_csv(_val_csv, nrows=1, encoding="utf-8-sig")
                    if len(_fc_head) > 0 and len(_vl_head) > 0:
                        logger.info("SKIP %s/%s — outputs already exist", target, model_name)
                        model_status[model_key] = "skip (outputs exist)"
                        continue
                    else:
                        logger.info("REDO %s/%s — existing outputs are empty, re-running", target, model_name)
                except Exception:
                    logger.info("REDO %s/%s — existing outputs unreadable, re-running", target, model_name)

            # ── Single predict_range call for validation period ──
            val_df = pd.DataFrame(columns=["时刻", "预测值", "model_name", "target", "run_date", "source"])
            val_error = None
            try:
                val_result = _run_model_for_range(
                    model_name, target=target,
                    start=val_start_str, end=val_end_str,
                    predict_date=val_start_str,
                    args=args,
                )
                if val_result is not None:
                    val_df = _prediction_frame_from_result(val_result, model_name, target, run_date, "validation")
            except Exception as exc:  # noqa: BLE001
                val_error = str(exc)
                logger.error("FAILED validation %s/%s: %s", target, model_name, exc)

            # ── Single predict_range call for forecast day ──
            forecast_df = pd.DataFrame(columns=["时刻", "预测值", "model_name", "target", "run_date", "source"])
            fc_error = None
            try:
                fc_result = _run_model_for_range(
                    model_name, target=target,
                    start=forecast_str, end=forecast_str,
                    predict_date=forecast_str,
                    args=args,
                )
                if fc_result is not None:
                    forecast_df = _prediction_frame_from_result(fc_result, model_name, target, run_date, "forecast")
            except Exception as exc:  # noqa: BLE001
                fc_error = str(exc)
                logger.error("FAILED forecast %s/%s: %s", target, model_name, exc)

            if val_df.empty and forecast_df.empty:
                error_detail = val_error or fc_error or "both val and forecast produced no output"
                model_status[model_key] = f"FAIL: {error_detail}"
                continue

            val_path = model_dir / "val_predictions.csv"
            forecast_path = model_dir / "forecast_predictions.csv"

            if not val_df.empty:
                val_ready = _attach_truth(val_df, truth_df)
                val_ready.to_csv(val_path, index=False, encoding="utf-8-sig")
                outputs.append(str(val_path))
                logger.info("%s/%s val: %d rows", target, model_name, len(val_ready))

            if not forecast_df.empty:
                forecast_ready = _attach_optional_truth(forecast_df, truth_df)
                forecast_ready.to_csv(forecast_path, index=False, encoding="utf-8-sig")
                outputs.append(str(forecast_path))
                logger.info("%s/%s forecast: %d rows", target, model_name, len(forecast_ready))

            model_status[model_key] = "ok"

    # ── Summary: report which models succeeded/failed ──
    if model_status:
        ok_list = [k for k, v in model_status.items() if v == "ok" or v.startswith("skip")]
        fail_list = [(k, v) for k, v in model_status.items() if v.startswith("FAIL")]
        logger.info("=" * 60)
        logger.info("model_stage summary for %s:", run_date)
        for key in sorted(model_status):
            status = model_status[key]
            logger.info("  %-40s %s", key, status)
        if fail_list:
            logger.error("!! %d model(s) FAILED:", len(fail_list))
            for key, reason in fail_list:
                logger.error("   %s: %s", key, reason)
        logger.info("=" * 60)

    return outputs


def _collect_stage_predictions(layout, *, file_name: str) -> pd.DataFrame:
    if not layout.model_outputs_dir.exists():
        raise FileNotFoundError(f"Model outputs directory not found: {layout.model_outputs_dir}")
    frames: list[pd.DataFrame] = []
    for model_dir in sorted(layout.model_outputs_dir.iterdir()):
        candidate = model_dir / file_name
        if not candidate.exists():
            continue
        df = pd.read_csv(candidate, encoding="utf-8-sig")
        if not df.empty:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No stage prediction files found for {file_name} under {layout.model_outputs_dir}")
    return pd.concat(frames, ignore_index=True)


def _to_contract_long_table(
    df: pd.DataFrame,
    *,
    target: str,
    spike_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["ds"] = pd.to_datetime(out["时刻"], errors="coerce")
    out["target_day"] = out["ds"].dt.normalize().where(out["ds"].dt.hour != 0, out["ds"].dt.normalize() - pd.Timedelta(days=1))
    out["hour_business"] = out["ds"].dt.hour.replace({0: 24}).astype(int)
    out["period"] = out["hour_business"].map(infer_period)
    out["task"] = target
    out["y_true"] = pd.to_numeric(out["真实值"], errors="coerce")
    out["y_pred"] = pd.to_numeric(out["预测值"], errors="coerce")
    long_df = out[
        ["task", "model_name", "target_day", "ds", "period", "hour_business", "y_true", "y_pred"]
    ].copy()
    long_df["target_day"] = pd.to_datetime(long_df["target_day"]).dt.strftime("%Y-%m-%d")
    long_df = standardize_prediction_table(long_df.dropna(subset=["y_true", "y_pred"]))

    # 窗口1 尖峰融合: 把 spike_prob/is_spike 合并到 long_table，让下游学习器能用上。
    if spike_df is not None and not spike_df.empty:
        spike = spike_df.copy()
        if "ds" not in spike.columns and "时刻" in spike.columns:
            spike["ds"] = pd.to_datetime(spike["时刻"], errors="coerce")
        spike = spike[["ds", "spike_prob", "is_spike"]] if "spike_prob" in spike.columns else spike
        spike_cols = [c for c in ("spike_prob", "is_spike") if c in spike.columns]
        if "ds" in spike.columns and spike_cols:
            long_df = augment_long_table_with_extras(
                long_df, spike[["ds"] + spike_cols], join_keys=("ds",)
            )
    return long_df


def run_learner_stage(args):
    run_date = _resolve_run_date(args)
    targets = ["dayahead", "realtime"] if args.target == "both" else [args.target]
    root = _resolve_daily_runs_root(args)
    outputs: list[str] = []

    spike_detector = None
    raw_df = None
    try:
        raw_df = pd.read_excel(args.data_path, engine="openpyxl")
        raw_df["时刻"] = pd.to_datetime(raw_df["时刻"])
        spike_detector = SpikeDetector(SpikeDetectorConfig())
        spike_detector.fit(raw_df)
        logger.info("SpikeDetector trained on %d rows", len(raw_df))
    except Exception as exc:
        logger.warning("SpikeDetector training failed: %s", exc)

    for target in targets:
        layout = build_daily_run_layout(root, run_date, target)
        val_df = _collect_stage_predictions(layout, file_name="val_predictions.csv")
        raw_val_path = layout.learner_inputs_dir / "validation_predictions.csv"
        val_df.to_csv(raw_val_path, index=False, encoding="utf-8-sig")

        included_models = sorted(val_df["model_name"].dropna().unique().tolist()) if "model_name" in val_df.columns else []
        model_row_counts = {}
        if "model_name" in val_df.columns:
            for m in included_models:
                model_row_counts[m] = int((val_df["model_name"] == m).sum())
        logger.info(
            "learner_stage %s/%s: %d models included in fusion: %s",
            target, run_date, len(included_models),
            ", ".join(f"{m}({model_row_counts.get(m, 0)}rows)" for m in included_models),
        )
        if not included_models:
            raise RuntimeError(f"No model predictions found for {target}/{run_date} — cannot fit fusion weights")

        # 先生成 spike_predictions.csv（若 spike_detector 可用），然后用其 join 出来
        # 增强版的 long_table（含 spike_prob/is_spike）。
        spike_df_for_long: pd.DataFrame | None = None
        if spike_detector is not None and raw_df is not None:
            try:
                spike_probs = spike_detector.predict_spike_probability(raw_df)
                spike_labels = spike_detector.predict(raw_df, use_momentum=True)
                spike_df = raw_df[["时刻"]].copy()
                spike_df["spike_prob"] = spike_probs.values
                spike_df["is_spike"] = spike_labels.values
                spike_path = layout.learner_inputs_dir / "spike_predictions.csv"
                spike_df.to_csv(spike_path, index=False, encoding="utf-8-sig")
                outputs.append(str(spike_path))
                spike_df_for_long = spike_df
                logger.info("Spike predictions saved: %s", spike_path)
            except Exception as exc:
                logger.warning("Spike prediction failed: %s", exc)

        contract_df = _to_contract_long_table(val_df, target=target, spike_df=spike_df_for_long)
        contract_path = layout.learner_inputs_dir / "validation_long_table.csv"
        contract_df.to_csv(contract_path, index=False, encoding="utf-8-sig")
        spike_coverage = (
            float(contract_df["spike_prob"].notna().mean())
            if "spike_prob" in contract_df.columns else 0.0
        )
        logger.info(
            "learner_stage %s/%s: spike_prob coverage=%.2f%% in long table",
            target, run_date, spike_coverage * 100.0,
        )

        # ── Meta-learner (Ridge, 可自动用上 spike_prob/is_spike 特征) ──
        models_meta, report_df = fit_meta_learners_from_long_table(
            contract_df,
            alpha=1.0,
            cv_folds=3,
        )

        meta_learner_path = layout.learner_outputs_dir / "meta_learner.joblib"
        joblib.dump(models_meta, meta_learner_path)

        report_path = layout.learner_outputs_dir / "fit_report.csv"
        report_df.to_csv(report_path, index=False, encoding="utf-8-sig")
        outputs.extend([str(meta_learner_path), str(report_path)])

        # ── Period-aware dynamic weights (约束 SLSQP, 9-16 时段内 spike_prob 调控) ──
        try:
            dyn_result = fit_dynamic_weights(
                contract_df,
                lower_bound=float(getattr(args, "weight_lower_bound", -0.5)),
                upper_bound=float(getattr(args, "weight_upper_bound", 1.2)),
            )
            dyn_weights_path = layout.learner_outputs_dir / "dynamic_weights.csv"
            dyn_report_path = layout.learner_outputs_dir / "dynamic_weights_report.csv"
            # 序列化为 joblib: dict 形式方便 fuse_stage 直接载入
            dyn_joblib_path = layout.learner_outputs_dir / "dynamic_weights.joblib"
            joblib.dump(
                {
                    "weights": dyn_result.weights,
                    "spike_interpolation": dyn_result.spike_interpolation,
                },
                dyn_joblib_path,
            )
            # 同时给一个 CSV 给人看
            flat_rows: list[dict[str, object]] = []
            for (t, p), wmap in dyn_result.weights.items():
                for model_name, w in wmap.items():
                    flat_rows.append(
                        {
                            "task": t,
                            "period": p,
                            "model_name": model_name,
                            "weight": float(w),
                        }
                    )
            pd.DataFrame(flat_rows).to_csv(dyn_weights_path, index=False, encoding="utf-8-sig")
            dyn_result.report.to_csv(dyn_report_path, index=False, encoding="utf-8-sig")
            outputs.extend([str(dyn_weights_path), str(dyn_report_path), str(dyn_joblib_path)])

            # 在验证集上评估 dynamic weights（含 spike-aware 调控）
            try:
                metrics = evaluate_dynamic_weights(
                    contract_df,
                    dyn_result.weights,
                    spike_templates=dyn_result.spike_interpolation,
                )
                logger.info(
                    "  dynamic_weights[%s] val: overall=%.2f%% 1_8=%.2f%% 9_16=%.2f%% 17_24=%.2f%% (rows=%d)",
                    target,
                    metrics.get("smape_overall", float("nan")),
                    metrics.get("smape_1_8", float("nan")),
                    metrics.get("smape_9_16", float("nan")),
                    metrics.get("smape_17_24", float("nan")),
                    metrics.get("rows", 0),
                )
            except Exception as exc:
                logger.warning("evaluate_dynamic_weights failed: %s", exc)
        except Exception as exc:
            logger.warning("fit_dynamic_weights failed: %s", exc)
            # Fallback: 用 fit_weights_from_long_table 保底
            try:
                fallback_w_df, _ = fit_weights_from_long_table(
                    contract_df,
                    reg=0.1,
                    lower_bound=float(getattr(args, "weight_lower_bound", -0.5)),
                    upper_bound=float(getattr(args, "weight_upper_bound", 1.2)),
                )
                fallback_path = layout.learner_outputs_dir / "dynamic_weights.csv"
                fallback_w_df.to_csv(fallback_path, index=False, encoding="utf-8-sig")
                outputs.append(str(fallback_path))
            except Exception as exc2:
                logger.warning("Fallback weights also failed: %s", exc2)

        for key, seg in models_meta.items():
            row = report_df[(report_df["task"] == key[0]) & (report_df["period"] == key[1])]
            cv_val = float(row["cv_smape"].iloc[0]) if not row.empty else 0.0
            use_val = bool(row["use_learner"].iloc[0]) if not row.empty else False
            extras_val = (
                str(row["extra_features"].iloc[0]) if "extra_features" in row.columns and not row.empty else ""
            )
            logger.info(
                "  period=%s best_single=%s(%.2f%%) cv_smape=%.2f%% use_learner=%s extras=%s",
                key[1], seg.best_single_model, seg.best_single_smape, cv_val, use_val, extras_val,
            )

    return outputs


def _ensure_complete_period_weights(weights_df: pd.DataFrame, contract_df: pd.DataFrame, *, target: str) -> pd.DataFrame:
    model_names = sorted(contract_df["model_name"].dropna().astype(str).unique().tolist())
    if not model_names:
        return weights_df
    existing = {
        (str(row.task), str(row.period), str(row.model_name))
        for row in weights_df[["task", "period", "model_name"]].itertuples(index=False)
    }
    rows: list[dict[str, object]] = []
    default_weight = 1.0 / len(model_names)
    lower_bound = float(weights_df["weight_lower_bound"].iloc[0]) if "weight_lower_bound" in weights_df.columns and not weights_df.empty else -0.5
    upper_bound = float(weights_df["weight_upper_bound"].iloc[0]) if "weight_upper_bound" in weights_df.columns and not weights_df.empty else 1.2
    for period in ["1_8", "9_16", "17_24"]:
        period_has_weight = ((weights_df["task"] == target) & (weights_df["period"] == period)).any()
        if period_has_weight:
            continue
        for model_name in model_names:
            key = (target, period, model_name)
            if key in existing:
                continue
            rows.append(
                {
                    "task": target,
                    "period": period,
                    "model_name": model_name,
                    "weight": default_weight,
                    "sample_count": 0,
                    "weight_lower_bound": lower_bound,
                    "weight_upper_bound": upper_bound,
                }
            )
    if rows:
        weights_df = pd.concat([weights_df, pd.DataFrame(rows)], ignore_index=True)
    return weights_df.sort_values(["task", "period", "model_name"]).reset_index(drop=True)


def _build_forecast_long_with_truth(forecast_df: pd.DataFrame, *, target: str) -> pd.DataFrame:
    work = forecast_df.copy()
    work["时刻"] = pd.to_datetime(work["时刻"], errors="coerce")
    work["预测值"] = pd.to_numeric(work["预测值"], errors="coerce")
    work["真实值"] = pd.to_numeric(work["真实值"], errors="coerce")
    work = work.dropna(subset=["时刻", "预测值"])

    ds = pd.to_datetime(work["时刻"])
    hour_business = ds.dt.hour.replace({0: 24}).astype(int)
    return pd.DataFrame(
        {
            "task": target,
            "target_day": ds.dt.normalize().where(ds.dt.hour != 0, ds.dt.normalize() - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d"),
            "model_name": work["model_name"].astype(str),
            "ds": ds,
            "period": hour_business.map(infer_period),
            "hour_business": hour_business,
            "y_true": work["真实值"],
            "y_pred": work["预测值"],
        }
    )


def _load_spike_for_day(spike_path: Path, run_date: str) -> pd.DataFrame | None:
    """Load spike_predictions.csv and return the slice for `run_date` (+/-1 day slack)."""
    if not spike_path.exists():
        return None
    try:
        df = pd.read_csv(spike_path, encoding="utf-8-sig")
    except Exception:
        return None
    if df.empty:
        return None
    if "时刻" in df.columns:
        df["ds"] = pd.to_datetime(df["时刻"], errors="coerce")
    elif "ds" in df.columns:
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    else:
        return None
    run_ts = pd.Timestamp(run_date)
    mask = (df["ds"] >= run_ts - pd.Timedelta(days=1)) & (df["ds"] <= run_ts + pd.Timedelta(days=1))
    df = df[mask].copy()
    if "spike_prob" not in df.columns:
        return None
    return df


def run_fuse_stage(args):
    run_date = _resolve_run_date(args)
    targets = ["dayahead", "realtime"] if args.target == "both" else [args.target]
    root = _resolve_daily_runs_root(args)
    outputs: list[str] = []

    for target in targets:
        layout = build_daily_run_layout(root, run_date, target)
        forecast_df = _collect_stage_predictions(layout, file_name="forecast_predictions.csv")
        normalized = _build_forecast_long_with_truth(forecast_df, target=target)

        # ── 读 spike_predictions.csv 并 join 到 normalized (供 dynamic_weights 9-16 调控用) ──
        spike_path = layout.learner_inputs_dir / "spike_predictions.csv"
        spike_for_day = _load_spike_for_day(spike_path, run_date)
        if spike_for_day is not None and not spike_for_day.empty:
            normalized = augment_long_table_with_extras(
                normalized, spike_for_day[["ds", "spike_prob", "is_spike"]],
                join_keys=("ds",),
            )

        # ── (1) Window1 尖峰融合 dynamic weights (含 spike-aware 9-16 调控) ──
        dyn_joblib_path = layout.learner_outputs_dir / "dynamic_weights.joblib"
        dyn_csv_path = layout.learner_outputs_dir / "dynamic_weights.csv"
        fused = None
        if dyn_joblib_path.exists():
            try:
                dyn_artifact = joblib.load(dyn_joblib_path)
                weights_dict = dyn_artifact.get("weights", {})
                spike_templates = dyn_artifact.get("spike_interpolation", {})
                fused = apply_dynamic_weights(
                    normalized,
                    weights_dict,
                    spike_templates=spike_templates,
                    spike_col="spike_prob",
                )
                dyn_fused_path = layout.final_dir / "fused_dynamic.csv"
                fused.to_csv(dyn_fused_path, index=False, encoding="utf-8-sig")
                outputs.append(str(dyn_fused_path))
                logger.info(
                    "fuse_stage %s/%s: dynamic weights applied (Window1 spike-aware).",
                    target, run_date,
                )
            except Exception as exc:
                logger.warning("apply_dynamic_weights failed: %s", exc)
        elif dyn_csv_path.exists():
            # Fallback: 仅 CSV (旧格式，无 spike-aware 9-16 调控)
            try:
                dyn_weights_df = pd.read_csv(dyn_csv_path, encoding="utf-8-sig")
                val_df_for_completion = _collect_stage_predictions(layout, file_name="val_predictions.csv")
                try:
                    dyn_weights_df = _ensure_complete_period_weights(
                        dyn_weights_df,
                        _to_contract_long_table(val_df_for_completion, target=target),
                        target=target,
                    )
                except Exception:
                    pass
                fused = _apply_fixed_weights(
                    normalized, dyn_weights_df,
                    task=target, test_start=run_date, test_end=run_date,
                )
                dyn_fused_path = layout.final_dir / "fused_dynamic.csv"
                fused.to_csv(dyn_fused_path, index=False, encoding="utf-8-sig")
                outputs.append(str(dyn_fused_path))
                logger.info(
                    "fuse_stage %s/%s: dynamic weights (legacy csv) applied.",
                    target, run_date,
                )
            except Exception as exc:
                logger.warning("Fallback dynamic csv apply failed: %s", exc)

        # ── (2) v3 meta learner (Ridge per (task, period)) ──
        meta_learner_path = layout.learner_outputs_dir / "meta_learner.joblib"
        fused_meta = None
        if meta_learner_path.exists():
            models_meta = joblib.load(meta_learner_path)
            fused_meta = apply_meta_learners(
                normalized,
                models_meta,
                task=target,
                test_start=run_date,
                test_end=run_date,
            )
            meta_fused_path = layout.final_dir / "fused_meta_learner.csv"
            fused_meta.to_csv(meta_fused_path, index=False, encoding="utf-8-sig")
            outputs.append(str(meta_fused_path))
        else:
            logger.warning("Meta learner not found: %s", meta_learner_path)

        # ── (3) Default output = dynamic weights (Window1 首选) if available, else meta learner ──
        if fused is not None and "y_fused" in fused.columns:
            final = fused
        elif fused_meta is not None:
            final = fused_meta
        else:
            raise FileNotFoundError("No fusion artifacts (dynamic_weights or meta_learner) available.")

        output_path = layout.final_dir / "fused_predictions.csv"
        final.to_csv(output_path, index=False, encoding="utf-8-sig")
        outputs.append(str(output_path))

        if "y_true" in final.columns and "y_fused" in final.columns:
            valid = final.dropna(subset=["y_true", "y_fused"])
            if len(valid) > 0:
                smape_val = smape_floor50(valid["y_true"].values, valid["y_fused"].values)
                logger.info("fuse_stage %s/%s: SMAPE=%.2f%% Accuracy=%.2f%%", target, run_date, smape_val, 100 - smape_val)

        # 按 period + spike_regime 分别汇报 (window1 关心的 9-16 在这里)
        try:
            if "spike_regime" in final.columns and "y_true" in final.columns:
                valid = final.dropna(subset=["y_true", "y_fused"])
                if not valid.empty:
                    for period in ("1_8", "9_16", "17_24"):
                        sub = valid[valid["period"] == period]
                        if not sub.empty:
                            smp = smape_floor50(sub["y_true"].values, sub["y_fused"].values)
                            logger.info(
                                "fuse_stage %s/%s: per-period %s SMAPE=%.2f%% (n=%d)",
                                target, run_date, period, smp, len(sub),
                            )
                    for regime in ("neutral", "sgdfnet_heavy", "rt916_heavy"):
                        sub = valid[valid["spike_regime"] == regime]
                        if not sub.empty:
                            smp = smape_floor50(sub["y_true"].values, sub["y_fused"].values)
                            logger.info(
                                "fuse_stage %s/%s: spike_regime=%s SMAPE=%.2f%% (n=%d)",
                                target, run_date, regime, smp, len(sub),
                            )
        except Exception as exc:
            logger.warning("per-period reporting failed: %s", exc)

    return outputs


def run_classifier_stage(args):
    run_date = _resolve_run_date(args)
    if getattr(args, "target", "realtime") == "dayahead":
        return {"status": "skipped", "reason": "classifier_only_supports_realtime"}
    root = _resolve_daily_runs_root(args)
    layout = build_daily_run_layout(root, run_date, "realtime")

    compat_root = layout.target_dir / "compat_fusion"
    realtime_dir = compat_root / "realtime"
    realtime_dir.mkdir(parents=True, exist_ok=True)
    fused_src = layout.final_dir / "fused_predictions.csv"
    if not fused_src.exists():
        raise FileNotFoundError(f"Realtime fused predictions not found: {fused_src}")
    fused_dst = realtime_dir / "fused_predictions.csv"
    fused_dst.write_bytes(fused_src.read_bytes())

    project_root = Path(__file__).resolve().parents[1]
    default_clf_data = project_root / "ExtremPriceClf" / "data" / "260525.xlsx"
    clf_data_path = Path(args.clf_data) if args.clf_data else default_clf_data

    result = run_classifier_pipeline(
        fusion_work_dir=compat_root,
        project_root=project_root,
        start_date=run_date,
        end_date=run_date,
        clf_data_path=clf_data_path,
    )

    corrected_src = realtime_dir / "fused_predictions_corrected.csv"
    if corrected_src.exists():
        corrected_dst = layout.final_dir / "fused_predictions_corrected.csv"
        corrected_dst.write_bytes(corrected_src.read_bytes())
    return result


def run_full_pipeline(args):
    """One-command full pipeline: model_stage → learner_stage → fuse_stage → classifier_stage."""
    run_date = _resolve_run_date(args)
    logger.info("=" * 60)
    logger.info("FULL PIPELINE START — date=%s target=%s stage_models=%s", run_date, args.target, args.stage_models)
    logger.info("=" * 60)

    # Stage 1: Model predictions
    logger.info("── Stage 1/4: model_stage ──")
    model_outputs = run_model_stage(args)
    logger.info("model_stage produced %d output files", len(model_outputs))

    # Stage 2: Learn fusion weights
    logger.info("── Stage 2/4: learner_stage ──")
    learner_outputs = run_learner_stage(args)
    logger.info("learner_stage produced %d output files", len(learner_outputs))

    # Stage 3: Apply fusion weights
    logger.info("── Stage 3/4: fuse_stage ──")
    fuse_outputs = run_fuse_stage(args)
    logger.info("fuse_stage produced %d output files", len(fuse_outputs))

    # Stage 4: Classifier (only for realtime)
    logger.info("── Stage 4/4: classifier_stage ──")
    if args.target in ("both", "realtime"):
        classifier_result = run_classifier_stage(args)
        logger.info("classifier_stage result: %s", classifier_result)
    else:
        logger.info("classifier_stage skipped (target=dayahead only)")
        classifier_result = {"status": "skipped", "reason": "dayahead_only"}

    logger.info("=" * 60)
    logger.info("FULL PIPELINE COMPLETE — date=%s", run_date)
    logger.info("  model outputs: %d files", len(model_outputs))
    logger.info("  learner outputs: %d files", len(learner_outputs))
    logger.info("  fuse outputs: %d files", len(fuse_outputs))
    logger.info("  classifier: %s", classifier_result)
    logger.info("=" * 60)

    return {
        "model_stage": model_outputs,
        "learner_stage": learner_outputs,
        "fuse_stage": fuse_outputs,
        "classifier_stage": classifier_result,
    }
