# -*- coding: utf-8 -*-
"""TimesFM 模型的 rolling-origin 适配器。

TimesFM 是纯预训练模型，无训练环节。
本适配器确保 cutoff-safe 推理：上下文截止于 train_end，缓存按 train_end 隔离。

Phase 3 新增：
- 逐日推理模式（daily）：每天使用 cutoff=d-1，更接近生产场景。
- 30 天批量逐日推理 + 聚合到 10 block。
- Block 推理模式（block）：原有行为，作为 daily 的 fallback。
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from rolling_oof.adapters.base import BaseRollingAdapter
from rolling_oof.contracts import FoldResult, FoldSpec

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


class TimesFMRollingAdapter(BaseRollingAdapter):
    """TimesFM rolling-origin 适配器（cutoff-safe 推理）。

    支持两种推理模式：
    - "daily": 逐日推理，每天使用 d-1 作为 cutoff（更接近真实生产场景）。
    - "block":  逐块推理，整个 block 共用一个 cutoff（原有行为）。
    """

    model_name: str = "timesfm"
    device_type: str = "gpu"

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """TimesFM 对单个 fold 执行推理。

        Parameters
        ----------
        inference_mode : str
            "daily"（默认）逐日推理；"block" 整块推理。
            daily 失败时自动 fallback 到 block。
        output_dir : str | None
            逐日缓存的输出目录。默认在 data_path 同级 timesfm_cache/daily_inference。
        """
        inference_mode: str = kwargs.pop("inference_mode", "daily")
        output_dir: str | None = kwargs.pop("output_dir", None)

        logger.info(
            "[timesfm/%s] fold %d: mode=%s, %s -> %s (cutoff=%s)",
            task,
            fold_spec.fold_id,
            inference_mode,
            fold_spec.test_start,
            fold_spec.test_end,
            fold_spec.train_end,
        )

        # ---- daily 模式 ----
        if inference_mode == "daily":
            try:
                return self._daily_fold_predict(
                    task, fold_spec, data_path, output_dir, **kwargs,
                )
            except Exception as e:
                logger.warning(
                    "[timesfm/%s] fold %d daily inference failed: %s. "
                    "Falling back to block mode.",
                    task, fold_spec.fold_id, e, exc_info=True,
                )
                # fall through to block mode

        # ---- block 模式（默认 / fallback）----
        return self._block_fold_predict(task, fold_spec, data_path, **kwargs)

    # ------------------------------------------------------------------
    # Daily 模式内部实现
    # ------------------------------------------------------------------

    def _daily_fold_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        output_dir: str | None,
        **kwargs,
    ) -> FoldResult:
        """逐日推理：对 fold 内每一天 d 使用 cutoff=d-1 进行预测。"""
        if output_dir is None:
            output_dir = str(
                Path(data_path).parent / "timesfm_cache" / "daily_inference"
            )
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        all_frames: list[pd.DataFrame] = []
        current = fold_spec.test_start
        while current <= fold_spec.test_end:
            d = current
            cutoff = d - timedelta(days=1)

            # 检查逐日缓存
            cache_file = Path(output_dir) / f"daily_{d.isoformat()}.csv"
            if cache_file.exists():
                try:
                    cached = pd.read_csv(cache_file)
                    if not cached.empty:
                        logger.info(
                            "[timesfm/daily] cache hit: %s", cache_file.name,
                        )
                        all_frames.append(cached)
                        current += timedelta(days=1)
                        continue
                except Exception:
                    logger.warning(
                        "[timesfm/daily] cache corrupted for %s, regenerating", d,
                    )

            # cutoff-safe 缓存目录（每天 cutoff 不同 → 目录不同）
            day_cache_dir = _get_cutoff_safe_cache_dir(
                data_path, task, cutoff.isoformat(),
            )

            result_df = _predict_oof_fold(
                data_path=data_path,
                start_date=d.isoformat(),
                end_date=d.isoformat(),
                target=task,
                cache_dir=day_cache_dir,
                cutoff_date=cutoff.isoformat(),
                segment_count=kwargs.get("segment_count", 3),
                seed=kwargs.get("seed", 42),
                deterministic=kwargs.get("deterministic", True),
            )

            if result_df is not None and not result_df.empty:
                # 标记逐日推理元数据
                result_df["tap_source"] = "direct_inference_daily"
                result_df["source_confidence"] = 0.90
                result_df["predict_day"] = d.isoformat()
                result_df["cutoff_date_daily"] = cutoff.isoformat()

                # 写入逐日缓存
                result_df.to_csv(cache_file, index=False)
                all_frames.append(result_df)
            else:
                logger.warning("[timesfm/daily] no prediction for day %s", d)

            current += timedelta(days=1)

        if not all_frames:
            raise RuntimeError(
                f"No daily predictions generated for fold {fold_spec.fold_id}"
            )

        combined = pd.concat(all_frames, ignore_index=True)
        return FoldResult(
            fold_id=fold_spec.fold_id,
            model_name=self.model_name,
            task=task,
            fold_spec=fold_spec,
            predictions_df=combined,
            success=True,
        )

    # ------------------------------------------------------------------
    # Block 模式内部实现（原有行为）
    # ------------------------------------------------------------------

    def _block_fold_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """整块推理：整个 fold 共用 train_end 作为 cutoff（原有逻辑）。"""
        try:
            cache_dir = _get_cutoff_safe_cache_dir(
                data_path, task, fold_spec.train_end.isoformat(),
            )

            result_df = _predict_oof_fold(
                data_path=data_path,
                start_date=fold_spec.test_start.isoformat(),
                end_date=fold_spec.test_end.isoformat(),
                target=task,
                cache_dir=cache_dir,
                cutoff_date=fold_spec.train_end.isoformat(),
                segment_count=kwargs.get("segment_count", 3),
                seed=kwargs.get("seed", 42),
                deterministic=kwargs.get("deterministic", True),
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

            # 标记 block 推理元数据
            result_df["tap_source"] = "direct_inference_block"
            result_df["source_confidence"] = 0.85

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
                "[timesfm/%s] fold %d failed: %s",
                task, fold_spec.fold_id, e, exc_info=True,
            )
            return FoldResult(
                fold_id=fold_spec.fold_id,
                model_name=self.model_name,
                task=task,
                fold_spec=fold_spec,
                success=False,
                error_message=str(e),
            )

    # ------------------------------------------------------------------
    # 高级批量方法
    # ------------------------------------------------------------------

    def daily_inference_30day(
        self,
        task: str,
        data_path: str,
        predict_date: str,
        output_dir: str,
        **kwargs,
    ) -> pd.DataFrame:
        """对 D-30 .. D-1 共 30 天逐日推理。

        Parameters
        ----------
        task : str
            "dayahead" 或 "realtime"。
        data_path : str
            原始数据文件路径。
        predict_date : str
            D（目标月第一天），ISO 格式日期字符串。
        output_dir : str
            逐日缓存目录。

        Returns
        -------
        pd.DataFrame
            30 天的逐日预测结果合并。
            包含 tap_source / source_confidence / predict_day 列。
        """
        D = pd.Timestamp(predict_date).date()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        target_tag = f"timesfm_{task}"
        all_frames: list[pd.DataFrame] = []
        for offset in range(30, 0, -1):  # D-30, D-30+1, ..., D-1
            d = D - timedelta(days=offset)
            cutoff = d - timedelta(days=1)

            # P0-5: Unified cache key — consistent with _predict_oof_fold internal cache
            day_cache_dir = _get_cutoff_safe_cache_dir(
                data_path, task, cutoff.isoformat(),
            )
            cache_file = out / f"daily_{target_tag}_{d.isoformat()}_cutoff_{cutoff.isoformat()}.csv"
            if cache_file.exists() and cache_file.stat().st_size > 0:
                try:
                    cached = pd.read_csv(cache_file)
                    if not cached.empty:
                        logger.info("[timesfm/daily30] cache hit: %s", cache_file.name)
                        all_frames.append(cached)
                        continue
                except Exception:
                    pass

            # Also check internal cutoff-safe cache
            internal_cache = Path(day_cache_dir) / f"predictions_{d.isoformat()}_{d.isoformat()}.csv"
            if internal_cache.exists() and internal_cache.stat().st_size > 0:
                try:
                    cached = pd.read_csv(internal_cache)
                    if not cached.empty:
                        logger.info("[timesfm/daily30] internal cache hit for %s", d)
                        cached["tap_source"] = "direct_inference_daily"
                        cached["source_confidence"] = 0.90
                        cached["predict_day"] = d.isoformat()
                        cached["cutoff_date_daily"] = cutoff.isoformat()
                        cached["daily_inference_day"] = d.isoformat()
                        cached.to_csv(cache_file, index=False)
                        all_frames.append(cached)
                        continue
                except Exception:
                    pass
            result_df = _predict_oof_fold(
                data_path=data_path,
                start_date=d.isoformat(),
                end_date=d.isoformat(),
                target=task,
                cache_dir=day_cache_dir,
                cutoff_date=cutoff.isoformat(),
                segment_count=kwargs.get("segment_count", 3),
                seed=kwargs.get("seed", 42),
                deterministic=kwargs.get("deterministic", True),
            )

            if result_df is not None and not result_df.empty:
                result_df["tap_source"] = "direct_inference_daily"
                result_df["source_confidence"] = 0.90
                result_df["predict_day"] = d.isoformat()
                result_df["cutoff_date_daily"] = cutoff.isoformat()
                result_df.to_csv(cache_file, index=False)
                all_frames.append(result_df)
            else:
                logger.warning("[timesfm/daily30] no prediction for %s", d)

        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames, ignore_index=True)

    def aggregate_daily_to_blocks(
        self,
        daily_df: pd.DataFrame,
        fold_specs: list[FoldSpec],
    ) -> pd.DataFrame:
        """将 30 天逐日预测映射到 10 个 block（每 block 3 天）。

        Block 划分：
            Block 0: D-30, D-29, D-28
            Block 1: D-27, D-26, D-25
            ...
            Block 9: D-3,  D-2,  D-1

        Parameters
        ----------
        daily_df : pd.DataFrame
            daily_inference_30day 返回的 30 天预测。
        fold_specs : list[FoldSpec]
            10 个 FoldSpec，每个覆盖 3 天。

        Returns
        -------
        pd.DataFrame
            原始 daily_df 增加 tap_block_id / age_block / horizon_day 列，
            并按 fold_specs 顺序排列。
        """
        if daily_df.empty:
            return daily_df

        df = daily_df.copy()

        # 确保有可解析的日期列
        if "predict_day" in df.columns:
            df["_predict_day_dt"] = pd.to_datetime(df["predict_day"]).dt.date
        elif "ds" in df.columns:
            df["_predict_day_dt"] = pd.to_datetime(df["ds"]).dt.normalize().dt.date
        else:
            logger.error("[timesfm/aggregate] no date column for block mapping")
            return df

        # 从 fold_specs 推断 D（目标月第一天）
        # D = max(test_end) + 1 day
        D = max(fs.test_end for fs in fold_specs) + timedelta(days=1)

        # 建立日期 → block_id 映射
        date_to_block: dict = {}
        for fs in fold_specs:
            current = fs.test_start
            while current <= fs.test_end:
                k = (D - current).days  # 1..30
                block_id = (30 - k) // 3
                date_to_block[current] = block_id
                current += timedelta(days=1)

        df["tap_block_id"] = df["_predict_day_dt"].map(date_to_block)
        df["age_block"] = df["tap_block_id"].apply(
            lambda b: 9 - b if pd.notna(b) else None
        )
        df["horizon_day"] = df["_predict_day_dt"].apply(
            lambda d: (D - d).days if d is not None else None
        )
        df.drop(columns=["_predict_day_dt"], inplace=True)

        # 按 block 排序
        df.sort_values(
            ["tap_block_id", "predict_day" if "predict_day" in df.columns else "ds"],
            inplace=True,
        )
        return df.reset_index(drop=True)

    def block_inference_10fold(
        self,
        task: str,
        data_path: str,
        fold_specs: list[FoldSpec],
        output_dir: str,
        **kwargs,
    ) -> pd.DataFrame:
        """10 个 block 逐块推理（原有行为）。

        Returns
        -------
        pd.DataFrame
            10 个 block 的预测合并，带 tap_block_id / tap_source 标记。
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_frames: list[pd.DataFrame] = []
        for fs in fold_specs:
            cache_file = out / f"block_{fs.fold_id}_{fs.test_start}_{fs.test_end}.csv"
            if cache_file.exists():
                try:
                    cached = pd.read_csv(cache_file)
                    if not cached.empty:
                        logger.info("[timesfm/block10] cache hit: fold %d", fs.fold_id)
                        all_frames.append(cached)
                        continue
                except Exception:
                    pass

            cache_dir = _get_cutoff_safe_cache_dir(
                data_path, task, fs.train_end.isoformat(),
            )
            result_df = _predict_oof_fold(
                data_path=data_path,
                start_date=fs.test_start.isoformat(),
                end_date=fs.test_end.isoformat(),
                target=task,
                cache_dir=cache_dir,
                cutoff_date=fs.train_end.isoformat(),
                segment_count=kwargs.get("segment_count", 3),
                seed=kwargs.get("seed", 42),
                deterministic=kwargs.get("deterministic", True),
            )

            if result_df is not None and not result_df.empty:
                result_df["tap_source"] = "direct_inference_block"
                result_df["source_confidence"] = 0.85
                result_df["tap_block_id"] = fs.fold_id
                result_df["age_block"] = 9 - fs.fold_id
                result_df.to_csv(cache_file, index=False)
                all_frames.append(result_df)
            else:
                logger.warning("[timesfm/block10] no prediction for fold %d", fs.fold_id)

        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 底层函数
# ---------------------------------------------------------------------------


def _get_cutoff_safe_cache_dir(
    data_path: str, task: str, train_end: str
) -> str:
    """生成包含 train_end 的 cutoff-safe 缓存目录。

    不同 train_end 使用不同的缓存 key，防止缓存污染。
    """
    # 使用文件名+task+train_end 生成唯一 key
    data_name = Path(data_path).stem
    cache_key = f"{data_name}_{task}_cutoff_{train_end}"
    return str(Path(data_path).parent / "timesfm_cache" / cache_key)


def _predict_oof_fold(
    data_path: str,
    start_date: str,
    end_date: str,
    target: str,
    cache_dir: str,
    cutoff_date: str,
    segment_count: int = 3,
    seed: int = 42,
    deterministic: bool = True,
) -> pd.DataFrame | None:
    """对单个 fold 执行 TimesFM 预测。

    Cutoff-safe 策略：
    1. 加载原始数据
    2. 将 cutoff_date 之后的目标列置为 NaN（防止 TimesFM 上下文看到未来数据）
    3. 写入临时 cutoff-safe 文件
    4. 用该文件进行推理
    5. 记录缓存 manifest
    """
    import json
    from datetime import datetime as dt

    cutoff_dt = pd.Timestamp(cutoff_date) + pd.Timedelta(days=1)  # cutoff 次日 00:00
    cache_file = Path(cache_dir) / f"predictions_{start_date}_{end_date}.csv"
    manifest_file = Path(cache_dir) / "cache_manifest.json"

    # 检查现有缓存
    if cache_file.exists():
        try:
            cached = pd.read_csv(cache_file)
            logger.info("[timesfm] Using cached predictions: %s", cache_file)
            return cached
        except Exception:
            logger.warning("[timesfm] Cache corrupted, regenerating")

    # 构造 cutoff-safe 数据视图
    raw_df = _load_raw_data(data_path)

    # 目标列 candidates 查找（兼容多种命名）
    target_col_candidates = {
        "dayahead": ["day_ahead_clearing_price", "日前电价", "日前出清价"],
        "realtime": ["realtime_price", "real_time_clearing_price", "实时电价"],
    }
    candidates = target_col_candidates.get(target, [])
    target_col = None
    for cand in candidates:
        if cand in raw_df.columns:
            target_col = cand
            break
    if target_col is None:
        logger.error(
            "[timesfm] Cannot find target column for %s. Candidates: %s. Columns: %s",
            target, candidates, list(raw_df.columns)[:20],
        )
        return None

    # 将 cutoff 之后的目标值置为 NaN
    if "ds" in raw_df.columns:
        raw_df["ds"] = pd.to_datetime(raw_df["ds"])
        cutoff_mask = raw_df["ds"] >= cutoff_dt
        raw_df.loc[cutoff_mask, target_col] = None
        logger.info(
            "[timesfm] Masked %d rows after cutoff %s (col=%s)",
            cutoff_mask.sum(),
            cutoff_dt,
            target_col,
        )

    # 写入临时文件
    cutoff_data_path = Path(cache_dir) / f"cutoff_safe_{Path(data_path).name}"
    cutoff_data_path.parent.mkdir(parents=True, exist_ok=True)
    if str(data_path).endswith(".xlsx"):
        raw_df.to_excel(str(cutoff_data_path), index=False)
    else:
        raw_df.to_csv(str(cutoff_data_path), index=False)

    # 调用 TimesFM 推理
    from TimesFM.infer import predict_price_for_range

    try:
        result_df = predict_price_for_range(
            data_path=str(cutoff_data_path),
            start_date=start_date,
            end_date=end_date,
            target=target,
            segment_count=segment_count,
            seed=seed,
            deterministic=deterministic,
        )

        if result_df is not None and not result_df.empty:
            result_df["model_name"] = "timesfm"
            result_df["task"] = target

            # 添加 fold 元数据
            result_df["train_end"] = cutoff_date
            result_df["source"] = "pretrained_inference"
            result_df["run_mode"] = "rolling_origin"

            # 缓存结果
            result_df.to_csv(cache_file, index=False)

            # 保存 manifest
            manifest = {
                "cache_file": str(cache_file),
                "cutoff_data_path": str(cutoff_data_path),
                "cutoff_date": cutoff_date,
                "start_date": start_date,
                "end_date": end_date,
                "target": target,
                "generated_at": dt.now().isoformat(),
                "rows_masked": int(cutoff_mask.sum()) if "cutoff_mask" in dir() else 0,
            }
            manifest_file.parent.mkdir(parents=True, exist_ok=True)
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)

        return result_df

    except Exception as e:
        logger.error("[timesfm] prediction failed: %s", e, exc_info=True)
        return None


def _load_raw_data(data_path: str) -> pd.DataFrame:
    """加载原始数据。支持 .csv / .xlsx / .xls，CSV 自动尝试 utf-8-sig → utf-8 → gbk → gb18030。"""
    path = str(data_path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pd.read_csv(path)
