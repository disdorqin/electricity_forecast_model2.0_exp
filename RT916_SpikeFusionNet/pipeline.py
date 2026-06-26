from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd

from pipelines.base import BaseModelPipeline, PredictionResult
from utils.io import ensure_prediction_frame, ensure_runtime_dirs


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rt916_spikefusionnet import core  # noqa: E402


TARGET_MAP = {
    "dayahead": "日前电价",
    "realtime": "实时电价",
}


class ModelPipeline(BaseModelPipeline):
    model_name = "rt916"
    device_type = "gpu"

    def train(self, target: str = "realtime", **kwargs):
        _dp = kwargs.get("data_path")
        if _dp:
            core.RAW_DF_PATH = os.path.abspath(_dp)
        start_end = self._resolve_start_end(kwargs)
        return core.train_interface(target=TARGET_MAP[target], start_end_list=start_end, mod="all")

    def predict(self, **kwargs) -> PredictionResult:
        return self.predict_range(**kwargs)

    def predict_range(self, target: str, **kwargs) -> PredictionResult:
        # Disable AMP during RT916 inference — model weights saved in BFloat16
        # cause "Unsupported dtype BFloat16" when converted to numpy.
        # Training (separate path) still benefits from AMP.
        os.environ["OPTIM_AMP"] = "0"
        os.environ["SPIKE_TRAIN_MONTHS"] = str(int(kwargs.get("training_months", 12)))
        # Override frozen RAW_DF_PATH so the model works on other machines / paths
        _dp = kwargs.get("data_path")
        if _dp:
            core.RAW_DF_PATH = os.path.abspath(_dp)
        start_end = self._resolve_start_end(kwargs)
        if target == "realtime":
            # RT916 realtime must first produce DA predictions, then inject them into RT.
            result = core.run_joint_da_rt_daily_backtest(
                start_end_list=start_end,
                mod="all",
                asof_hour=15,
            )
        else:
            result = core.run_daily_asof_backtest(
                target=TARGET_MAP[target],
                start_end_list=start_end,
                mod="all",
                asof_hour=15,
                retrain_daily=False,
            )
        prediction_col = "预测日前电价" if target == "dayahead" else "预测实时电价"
        if result is None or (isinstance(result, pd.DataFrame) and result.empty):
            raise ValueError(
                f"RT916 produced no predictions for target={target} "
                f"[{start_end[0]} to {start_end[1]}]. "
                f"Possible causes: insufficient training data, core returned empty DataFrame."
            )
        normalized = ensure_prediction_frame(result, prediction_col)
        output_root = ensure_runtime_dirs(Path(kwargs.get("output_root", "outputs/unified_runs")) / self.model_name / target)
        output_path = output_root / "predictions.csv"
        normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
        return PredictionResult(model_name=self.model_name, target=target, output_path=output_path, frame=normalized)

    def online_predict_range(
        self,
        target: str,
        *,
        fold_specs: list,
        checkpoint_root: str,
        online_epochs: int = 3,
        online_lr: float | None = None,
        **kwargs,
    ) -> list:
        """Run online walk-forward prediction for validation tap.

        Parameters
        ----------
        target : str
            "dayahead" or "realtime".
        fold_specs : list[FoldSpec]
            Fold specifications defining train/test windows.
        checkpoint_root : str
            Root directory for storing/loading model checkpoints.
        online_epochs : int
            Number of fine-tuning epochs per block.
        online_lr : float | None
            Learning rate for online update.

        Returns
        -------
        list[pd.DataFrame]
            One DataFrame per fold/block with predictions.
        """
        # Disable AMP during inference (same as predict_range)
        os.environ["OPTIM_AMP"] = "0"
        os.environ["OPTIM_NUM_WORKERS"] = "0"

        _dp = kwargs.get("data_path")
        if _dp:
            core.RAW_DF_PATH = os.path.abspath(_dp)

        # Read raw data and run feature engineering pipeline
        df_raw = core._load_raw_df()
        df_raw = core.process_features(df_raw)
        df_raw = core.feature_engineer_solar_terms(df_raw)

        cn_target = TARGET_MAP[target]
        df_raw = core.enrich_selected_features(df_raw, target_col=cn_target)

        if target == "realtime":
            # RT needs DA linkage features
            if bool(int(os.getenv("SPIKE_RT916_DA_LINKAGE", "1"))):
                core.CONFIG["ENABLE_DA_LINKAGE"] = True
                df_raw = core.enrich_da_linkage_features(df_raw, da_pred_series=None)
            else:
                core.CONFIG["ENABLE_DA_LINKAGE"] = False

        df_raw["时刻"] = pd.to_datetime(df_raw["时刻"])

        if target == "realtime":
            # For RT, must first produce DA predictions via joint flow
            results = core.run_online_walk_forward_joint(
                df_raw=df_raw,
                fold_specs=fold_specs,
                checkpoint_root=checkpoint_root,
                online_epochs=online_epochs,
                online_lr=online_lr,
            )
            return results.get("rt_predictions", [])
        else:
            # DA: configure and run directly
            test_start = str(pd.Timestamp(fold_specs[0].test_start))
            test_end = str(pd.Timestamp(fold_specs[-1].test_end))
            core._update_config(cn_target, [test_start, test_end])
            core.CONFIG["ENABLE_DA_LINKAGE"] = False

            return core.run_online_walk_forward_rt916(
                target=cn_target,
                df_raw=df_raw,
                fold_specs=fold_specs,
                checkpoint_root=checkpoint_root,
                online_epochs=online_epochs,
                online_lr=online_lr,
            )

    @staticmethod
    def _resolve_start_end(kwargs: dict) -> list[str]:
        start = kwargs.get("start")
        end = kwargs.get("end")
        if start and end:
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)
            if start_ts.hour == 0 and start_ts.minute == 0 and start_ts.second == 0:
                start_ts = start_ts.normalize() + pd.Timedelta(hours=1)
            if end_ts.hour == 0 and end_ts.minute == 0 and end_ts.second == 0:
                end_ts = end_ts.normalize() + pd.Timedelta(days=1)
            return [start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")]
        predict_date = pd.Timestamp(kwargs.get("predict_date"))
        start_ts = predict_date.normalize() + pd.Timedelta(hours=1)
        end_ts = predict_date.normalize() + pd.Timedelta(days=1)
        return [start_ts.strftime("%Y-%m-%d %H:%M:%S"), end_ts.strftime("%Y-%m-%d %H:%M:%S")]
