# -*- coding: utf-8 -*-
"""RT916/SpikeFusionNet 模型的 rolling-origin 适配器。

支持三种滚动模式：
- daily (默认): 每天训练一次，逐日预测（最严格）
- online: base train + online update（推荐，最快）
- window_once: 训练一次，预测整个 fold（基准对比）

关键修复：
1. 强制 retrain_daily=True（daily_walk_forward）
2. 训练时调用 apply_asof_cutoff_for_inference 防止 lag 特征泄露
3. 修复 00:00 时间对齐（统一为 business_hour=24）
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# RT916 子模块路径
_RT916_SRC_ROOT = _PROJECT_ROOT / "RT916_SpikeFusionNet" / "src"
if str(_RT916_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_RT916_SRC_ROOT))


TARGET_MAP = {"dayahead": "日前电价", "realtime": "实时电价"}

# 支持的滚动模式
SUPPORTED_ROLLING_MODES = ("daily", "online", "window_once")


class RT916RollingAdapter(BaseRollingAdapter):
    """RT916 rolling-origin 适配器（修复版）。

    支持 rolling_mode 参数：
    - "daily": 每天训练一次，逐日预测（默认，兼容旧行为）
    - "online": base train + online update（推荐）
    - "window_once": 训练一次，预测整个 fold
    """

    model_name: str = "rt916"
    device_type: str = "gpu"

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """RT916 对单个 fold 执行训练+预测。

        根据 rolling_mode 参数选择训练模式。
        """
        rolling_mode = kwargs.get("rolling_mode", "daily")
        if rolling_mode not in SUPPORTED_ROLLING_MODES:
            logger.warning(
                "[rt916/%s] unknown rolling_mode=%s, falling back to 'daily'",
                task, rolling_mode,
            )
            rolling_mode = "daily"

        logger.info(
            "[rt916/%s] fold %d: mode=%s, %s -> %s (cutoff=%s)",
            task,
            fold_spec.fold_id,
            rolling_mode,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        try:
            # Windows 下 CUDA + DataLoader multiprocessing 不稳定，强制单进程
            os.environ["OPTIM_NUM_WORKERS"] = "0"

            if rolling_mode == "online":
                result_df = _run_rt916_online_fold(
                    data_path=data_path,
                    fold_spec=fold_spec,
                    task=task,
                    checkpoint_dir=kwargs.get("checkpoint_dir"),
                    online_epochs=kwargs.get("online_epochs", 3),
                    online_lr=kwargs.get("online_lr"),
                )
            else:
                # daily / window_once: 走原有 daily_walk_forward 路径
                os.environ["SPIKE_TRAIN_START_DATE"] = fold_spec.train_start.isoformat()
                os.environ["SPIKE_TRAIN_END_DATE"] = fold_spec.train_end.isoformat()
                os.environ["SPIKE_RETRAIN_DAILY"] = "1"

                result_df = _run_rt916_fold(data_path, fold_spec, task)

            if result_df is None or (isinstance(result_df, pd.DataFrame) and result_df.empty):
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
        except Exception as e:
            logger.error(
                "[rt916/%s] fold %d failed: %s", task, fold_spec.fold_id, e, exc_info=True
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )
        finally:
            # 清理环境变量
            for key in (
                "SPIKE_TRAIN_START_DATE",
                "SPIKE_TRAIN_END_DATE",
                "SPIKE_RETRAIN_DAILY",
                "OPTIM_NUM_WORKERS",
            ):
                os.environ.pop(key, None)

    # ------------------------------------------------------------------
    # Online mode: 分步接口（供编排器使用）
    # ------------------------------------------------------------------

    @classmethod
    def online_base_train(
        cls,
        task: str,
        data_path: str,
        train_end,
        checkpoint_dir: str,
    ) -> None:
        """Online 模式: 执行 base training 并保存 checkpoint。

        Parameters
        ----------
        task : str
            "dayahead" 或 "realtime"。
        data_path : str
            原始数据文件路径。
        train_end : date | str
            训练数据截止日期（inclusive）。
        checkpoint_dir : str
            checkpoint 存储根目录。
        """
        from rt916_spikefusionnet import core

        os.environ["OPTIM_AMP"] = "1"  # 训练时启用 AMP
        os.environ["OPTIM_NUM_WORKERS"] = "0"

        cn_target = TARGET_MAP[task]

        # 读取数据并做特征工程
        df_raw = core._load_raw_df(data_path)
        df_raw = core.process_features(df_raw)
        df_raw = core.feature_engineer_solar_terms(df_raw)
        df_raw = core.enrich_selected_features(df_raw, target_col=cn_target)
        if task == "realtime" and bool(int(os.getenv("SPIKE_RT916_DA_LINKAGE", "1"))):
            core.CONFIG["ENABLE_DA_LINKAGE"] = True
            df_raw = core.enrich_da_linkage_features(df_raw, da_pred_series=None)
        else:
            core.CONFIG["ENABLE_DA_LINKAGE"] = False
        df_raw["时刻"] = pd.to_datetime(df_raw["时刻"])

        train_end_ts = pd.Timestamp(train_end)
        # 设置 CONFIG
        core._update_config(cn_target, [
            str(train_end_ts + pd.Timedelta(hours=1)),
            str(train_end_ts + pd.Timedelta(days=3)),  # dummy end
        ])

        base_train_df = df_raw[df_raw["时刻"] <= train_end_ts].copy()
        logger.info(
            "[rt916/%s] online_base_train: %d rows up to %s",
            task, len(base_train_df), train_end_ts.date(),
        )

        # 对 3 个时段分别训练
        for period_name, period_data in core._get_periods(base_train_df, "all"):
            core.CONFIG["CURRENT_PERIOD_NAME"] = period_name
            core.train_single_period(period_name, period_data)

            # 复制模型文件和 scaler 到 checkpoint 目录
            import shutil
            train_output_dir = os.path.join(
                core.CONFIG["SAVE_ROOT_DIR"], f"TS{core.CONFIG['TRAIN_STEPS']}_{period_name}"
            )
            pred_len = core.CONFIG["OUTPUT_LEN_LIST"]
            model_filename = f"model_{core.CONFIG['INPUT_LEN_LIST'] + 1}天输出最后{pred_len}点.pth"

            period_ckpt_dir = Path(checkpoint_dir) / period_name
            period_ckpt_dir.mkdir(parents=True, exist_ok=True)

            for fname in [model_filename, "scalar_input.pkl", "scalar_output.pkl"]:
                src = os.path.join(train_output_dir, fname)
                dst = str(period_ckpt_dir / fname)
                if os.path.exists(src):
                    shutil.copy2(src, dst)

            logger.info("[rt916/%s] base train checkpoint: %s", task, period_ckpt_dir)

    @classmethod
    def online_update_predict(
        cls,
        task: str,
        data_path: str,
        fold_spec: FoldSpec,
        checkpoint_dir: str,
        online_epochs: int = 3,
        online_lr: float | None = None,
    ) -> pd.DataFrame | None:
        """Online 模式: 加载 checkpoint，预测一个 block，然后 online update。

        Parameters
        ----------
        task : str
            "dayahead" 或 "realtime"。
        data_path : str
            原始数据文件路径。
        fold_spec : FoldSpec
            fold 参数。
        checkpoint_dir : str
            checkpoint 根目录。
        online_epochs : int
            微调轮数。
        online_lr : float | None
            微调学习率。

        Returns
        -------
        pd.DataFrame | None
            预测结果 DataFrame。
        """
        from rt916_spikefusionnet import core

        os.environ["OPTIM_AMP"] = "0"  # 推理时禁用 AMP
        os.environ["OPTIM_NUM_WORKERS"] = "0"

        cn_target = TARGET_MAP[task]

        # 读取数据并做特征工程
        df_raw = core._load_raw_df(data_path)
        df_raw = core.process_features(df_raw)
        df_raw = core.feature_engineer_solar_terms(df_raw)
        df_raw = core.enrich_selected_features(df_raw, target_col=cn_target)
        if task == "realtime" and bool(int(os.getenv("SPIKE_RT916_DA_LINKAGE", "1"))):
            core.CONFIG["ENABLE_DA_LINKAGE"] = True
            df_raw = core.enrich_da_linkage_features(df_raw, da_pred_series=None)
        else:
            core.CONFIG["ENABLE_DA_LINKAGE"] = False
        df_raw["时刻"] = pd.to_datetime(df_raw["时刻"])

        test_start = pd.Timestamp(fold_spec.test_start)
        test_end = pd.Timestamp(fold_spec.test_end)
        context_start = test_start - pd.Timedelta(days=core.CONFIG["INPUT_LEN_LIST"])

        core._update_config(cn_target, [str(test_start), str(test_end)])
        core.CONFIG["TEST_TOTAL_START_END_LIST"] = [str(test_start), str(test_end)]

        block_data = df_raw[
            (df_raw["时刻"] >= context_start) & (df_raw["时刻"] <= test_end)
        ].copy()

        # 对每个时段进行预测
        import shutil
        pred_len = core.CONFIG["OUTPUT_LEN_LIST"]
        model_filename = f"model_{core.CONFIG['INPUT_LEN_LIST'] + 1}天输出最后{pred_len}点.pth"
        block_predictions = {}

        for period_name, period_data in core._get_periods(block_data, "all"):
            core.CONFIG["CURRENT_PERIOD_NAME"] = period_name

            # 临时修改 SAVE_ROOT_DIR 使 inference_single_period 从 checkpoint 加载
            _orig_save_root = core.CONFIG["SAVE_ROOT_DIR"]
            _ts_dir_name = f"TS{core.CONFIG['TRAIN_STEPS']}_{period_name}"
            _tmp_save_root = str(Path(checkpoint_dir) / "_tmp_inference")
            _tmp_period_dir = os.path.join(_tmp_save_root, _ts_dir_name)
            os.makedirs(_tmp_save_root, exist_ok=True)
            if os.path.exists(_tmp_period_dir):
                shutil.rmtree(_tmp_period_dir)
            try:
                os.symlink(
                    str(Path(checkpoint_dir) / period_name),
                    _tmp_period_dir,
                    target_is_directory=True,
                )
            except OSError:
                shutil.copytree(str(Path(checkpoint_dir) / period_name), _tmp_period_dir)

            core.CONFIG["SAVE_ROOT_DIR"] = _tmp_save_root

            try:
                pred_df = core.inference_single_period(period_name, period_data)
                block_predictions[period_name] = pred_df
            except Exception as e:
                logger.error(
                    "[rt916/%s] online predict %s failed: %s", task, period_name, e
                )
                block_predictions[period_name] = pd.DataFrame()
            finally:
                core.CONFIG["SAVE_ROOT_DIR"] = _orig_save_root
                if os.path.exists(_tmp_period_dir):
                    try:
                        if os.path.islink(_tmp_period_dir):
                            os.unlink(_tmp_period_dir)
                        else:
                            shutil.rmtree(_tmp_period_dir)
                    except Exception:
                        pass

        # 合并预测结果
        valid_preds = [v for v in block_predictions.values() if v is not None and len(v) > 0]
        merged = pd.concat(valid_preds, ignore_index=False) if valid_preds else pd.DataFrame()

        # Online update: 用该 block 的真实值微调
        update_data_with_context = df_raw[
            (df_raw["时刻"] >= context_start) & (df_raw["时刻"] <= test_end)
        ].copy()

        for period_name, period_data in core._get_periods(update_data_with_context, "all"):
            try:
                core.online_update_single_period(
                    period_name=period_name,
                    update_df=period_data,
                    checkpoint_dir=checkpoint_dir,
                    online_epochs=online_epochs,
                    online_lr=online_lr,
                )
            except Exception as e:
                logger.error(
                    "[rt916/%s] online update %s failed: %s", task, period_name, e
                )

        if merged.empty:
            return None

        # 修复时间对齐
        if "ds" in merged.columns and "hour_business" not in merged.columns:
            merged["ds"] = pd.to_datetime(merged["ds"])
            merged["hour"] = merged["ds"].dt.hour
            merged["hour_business"] = merged["hour"].apply(lambda h: 24 if h == 0 else h)

        return merged


# ── 模块级辅助函数 ──────────────────────────────────────────────────────


def _run_rt916_fold(
    data_path: str, fold_spec: FoldSpec, task: str
) -> pd.DataFrame | None:
    """运行 RT916 fold（daily 模式）。"""
    from RT916_SpikeFusionNet.pipeline import ModelPipeline

    pipeline = ModelPipeline()

    result = pipeline.predict_range(
        target=task,
        data_path=data_path,
        start=fold_spec.test_start.isoformat(),
        end=fold_spec.test_end.isoformat(),
        predict_date=None,
        output_root="oof_runs/rt916_temp",
        retrain_daily=True,  # 强制 daily_walk_forward
        asof_hour=15,
    )

    if result is not None and hasattr(result, "frame"):
        df = result.frame
        # 修复时间对齐：确保 00:00 映射为 business_hour=24
        if "ds" in df.columns and "hour_business" not in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
            df["hour"] = df["ds"].dt.hour
            df["hour_business"] = df["hour"].apply(lambda h: 24 if h == 0 else h)
        return df
    return result


def _run_rt916_online_fold(
    data_path: str,
    fold_spec: FoldSpec,
    task: str,
    checkpoint_dir: str | None = None,
    online_epochs: int = 3,
    online_lr: float | None = None,
) -> pd.DataFrame | None:
    """运行 RT916 fold（online 模式）。

    使用 pipeline.online_predict_range 进行完整的 online walk-forward。
    """
    from RT916_SpikeFusionNet.pipeline import ModelPipeline

    pipeline = ModelPipeline()

    if checkpoint_dir is None:
        checkpoint_dir = f"oof_runs/rt916_online_temp/fold_{fold_spec.fold_id}_{task}/checkpoints"

    results = pipeline.online_predict_range(
        target=task,
        fold_specs=[fold_spec],
        checkpoint_root=checkpoint_dir,
        online_epochs=online_epochs,
        online_lr=online_lr,
        data_path=data_path,
    )

    if not results:
        return None

    # results 是 list[pd.DataFrame]，每个 fold 一个
    valid = [r for r in results if r is not None and len(r) > 0]
    if not valid:
        return None

    merged = pd.concat(valid, ignore_index=True)
    merged = merged.sort_values("时刻").drop_duplicates(subset=["时刻"], keep="last").reset_index(drop=True)

    # 修复时间对齐
    if "ds" in merged.columns and "hour_business" not in merged.columns:
        merged["ds"] = pd.to_datetime(merged["ds"])
        merged["hour"] = merged["ds"].dt.hour
        merged["hour_business"] = merged["hour"].apply(lambda h: 24 if h == 0 else h)

    return merged
