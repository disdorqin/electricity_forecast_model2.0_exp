# -*- coding: utf-8 -*-
"""LightGBM 模型的 rolling-origin 适配器。

处理日前和实时两种电价模式。
日前模式使用新增的 daily_walk_forward 函数；
实时模式使用现有的 run_precision_simulation（已经是 daily_walk_forward）。
"""

from __future__ import annotations

import datetime
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

# 项目根
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class LightGBMRollingAdapter(BaseRollingAdapter):
    """LightGBM rolling-origin 适配器。"""

    model_name: str = "lightgbm"
    device_type: str = "cpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """对单个 fold 执行 LightGBM 训练+预测。

        日前模式使用 daily_walk_forward，实时模式沿用现有 pipeline。
        """
        training_months = kwargs.get("training_months", 12)
        val_ratio = kwargs.get("val_ratio", 0.2)

        result_df: pd.DataFrame | None = None

        if task == "dayahead":
            result_df = _run_da_daily_walk_forward(
                data_path=data_path,
                forecast_start=fold_spec.test_start.isoformat(),
                forecast_end=fold_spec.test_end.isoformat(),
                train_start=fold_spec.train_start.isoformat(),
                train_end=fold_spec.train_end.isoformat(),
                training_months=training_months,
                val_ratio=val_ratio,
            )
        else:  # realtime
            result_df = _run_rt_daily_walk_forward(
                data_path=data_path,
                forecast_start=fold_spec.test_start.isoformat(),
                forecast_end=fold_spec.test_end.isoformat(),
                training_months=training_months,
                val_ratio=val_ratio,
            )

        if result_df is None or result_df.empty:
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message="No predictions generated",
            )

        return FoldResult(
            fold_id=fold_spec.fold_id,
            model_name=self.model_name,
            task=task,
            fold_spec=fold_spec,
            predictions_df=result_df,
            success=True,
        )


# ---------------------------------------------------------------------------
# 底层包装函数
# ---------------------------------------------------------------------------


def _run_da_daily_walk_forward(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    train_start: str,
    train_end: str,
    training_months: int = 12,
    val_ratio: float = 0.2,
) -> pd.DataFrame | None:
    """日前电价的逐日 walk-forward 训练。

    参考实时模式 run_precision_simulation 的循环逻辑，
    但使用日前子模型（LGBMPowerPredictorDA + PowerInferenceDA）。
    """
    from lightGBM.train_da_fix import LGBMPowerPredictor as LGBMPowerPredictorDA
    from lightGBM.infer_da_fix import PowerInference as PowerInferenceDA
    from lightGBM.main_fix import _fit_dayahead_fixed_window

    predictor = LGBMPowerPredictorDA()
    inference = PowerInferenceDA(model_path=None)
    current_target_date = pd.to_datetime(forecast_start)
    end_target_date = pd.to_datetime(forecast_end)
    train_start_dt = pd.to_datetime(train_start)

    # 加载一次原始数据（用于特征工程中的滚动特征计算）
    raw_df = predictor.load_and_process_data(data_path)

    all_days_preds: list[pd.DataFrame] = []
    while current_target_date <= end_target_date:
        target_day_str = current_target_date.strftime("%Y-%m-%d")

        # cutoff = 预测日前一天 14:00（与 realtime 模式一致）
        decision_day_dt = current_target_date - pd.Timedelta(days=1)
        val_end_str = decision_day_dt.strftime("%Y-%m-%d 14:00:00")

        # 训练窗口起始 = 从 train_start 开始（expanding 窗口）
        effective_train_start = max(train_start_dt, pd.to_datetime(train_start))
        val_start_str = effective_train_start.strftime("%Y-%m-%d 01:00:00")

        best_res = None
        try:
            best_res = _fit_dayahead_fixed_window(
                predictor=predictor,
                data_path=data_path,
                history_start_date=val_start_str,
                history_end_date=val_end_str,
                raw_df=raw_df,
                val_ratio=val_ratio,
            )
            inference.model = best_res["model"]
            inference_start = current_target_date.strftime("%Y-%m-%d 01:00:00")
            inference_end = (current_target_date + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d 00:00:00"
            )
            day_result_df = inference.predict_range(
                data_path, inference_start, inference_end, target="日前电价", raw_df=raw_df
            )

            if day_result_df is not None and not day_result_df.empty:
                day_result_df["target_day"] = target_day_str
                all_days_preds.append(day_result_df)
        except Exception as e:
            logger.error(
                "[%s] dayahead daily walk-forward failed: %s",
                target_day_str,
                e,
                exc_info=True,
            )
        finally:
            if best_res is not None:
                del best_res
            gc.collect()

        current_target_date += datetime.timedelta(days=1)

    if all_days_preds:
        result = pd.concat(all_days_preds, axis=0)
        return result
    return None


# ── 3×10 block 策略：单次训练 + 逐日预测（正确计算滚动特征）──
def _run_lgbm_10day_block(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    train_start: str,
    train_end: str,
    target: str = "dayahead",
    training_months: int = 12,
    val_ratio: float = 0.2,
) -> pd.DataFrame | None:
    """3×10 true rolling: train once at cutoff, predict 10 days individually.

    For each 10-day block:
      - Train ONE model with history up to train_end (cutoff date)
      - Predict each day one at a time via predict_range (correct rolling features)
      - Return combined 10-day predictions

    This avoids the per-day retraining of the old daily_walk_forward,
    reducing from 30 trainings to 3 per 30-day validation window.
    """
    if target == "dayahead":
        return _run_da_10day_block(
            data_path, forecast_start, forecast_end,
            train_start, train_end, training_months, val_ratio,
        )
    else:
        # realtime: the existing pipeline is already per-day walk-forward
        # For 3×10, we train at cutoff and predict 10 days
        from lightGBM.main_fix import run_lgbm_pipeline
        return run_lgbm_pipeline(
            data_path=data_path,
            forecast_start=forecast_start,
            forecast_end=forecast_end,
            target="实时电价",
            use_predicted_temp=False,
            training_months=training_months,
            val_ratio=val_ratio,
        )


def _run_da_10day_block(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    train_start: str,
    train_end: str,
    training_months: int = 12,
    val_ratio: float = 0.2,
) -> pd.DataFrame | None:
    """日前电价 3×10 block: train once, predict 10 days individually."""
    from lightGBM.train_da_fix import LGBMPowerPredictor as LGBMPowerPredictorDA
    from lightGBM.infer_da_fix import PowerInference as PowerInferenceDA
    from lightGBM.main_fix import _fit_dayahead_fixed_window

    predictor = LGBMPowerPredictorDA()
    inference = PowerInferenceDA(model_path=None)

    train_end_dt = pd.to_datetime(train_end)
    forecast_start_dt = pd.to_datetime(forecast_start)
    forecast_end_dt = pd.to_datetime(forecast_end)
    train_start_dt = pd.to_datetime(train_start)

    # Load raw data once for feature engineering context
    raw_df = predictor.load_and_process_data(data_path)

    # ── Train ONCE at block cutoff ──
    # Cutoff is train_end + " 14:00" (same convention as per-day)
    val_end_str = train_end_dt.strftime("%Y-%m-%d 14:00:00")
    val_start_str = train_start_dt.strftime("%Y-%m-%d 01:00:00")

    best_res = _fit_dayahead_fixed_window(
        predictor=predictor,
        data_path=data_path,
        history_start_date=val_start_str,
        history_end_date=val_end_str,
        raw_df=raw_df,
        val_ratio=val_ratio,
    )
    inference.model = best_res["model"]

    # ── Predict each day individually ──
    all_days_preds: list[pd.DataFrame] = []
    current_date = forecast_start_dt
    while current_date <= forecast_end_dt:
        target_day_str = current_date.strftime("%Y-%m-%d")
        try:
            inference_start = current_date.strftime("%Y-%m-%d 01:00:00")
            inference_end = (current_date + datetime.timedelta(days=1)).strftime(
                "%Y-%m-%d 00:00:00"
            )
            day_result_df = inference.predict_range(
                data_path, inference_start, inference_end,
                target="日前电价", raw_df=raw_df,
            )

            if day_result_df is not None and not day_result_df.empty:
                day_result_df["target_day"] = target_day_str
                all_days_preds.append(day_result_df)
        except Exception as e:
            logger.error(
                "[lgbm_10day_block] %s prediction failed: %s",
                target_day_str, e, exc_info=True,
            )

        current_date += datetime.timedelta(days=1)

    del best_res
    gc.collect()

    if all_days_preds:
        return pd.concat(all_days_preds, axis=0)
    return None


def _run_rt_daily_walk_forward(
    data_path: str,
    forecast_start: str,
    forecast_end: str,
    training_months: int = 12,
    val_ratio: float = 0.2,
) -> pd.DataFrame | None:
    """实时电价的 daily_walk_forward（直接使用现有 pipeline）。

    现有的 run_lgbm_pipeline(target="实时电价") 已经是 daily_walk_forward。
    """
    from lightGBM.main_fix import run_lgbm_pipeline

    result = run_lgbm_pipeline(
        data_path=data_path,
        forecast_start=forecast_start,
        forecast_end=forecast_end,
        target="实时电价",
        use_predicted_temp=False,
        training_months=training_months,
        val_ratio=val_ratio,
    )
    return result
