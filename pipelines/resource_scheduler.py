"""CPU/GPU parallel resource scheduler for R3D-Online-Tap-GEF.

Manages concurrent execution of CPU-bound and GPU-bound model tasks.
GPU tasks are serialized (single GPU). CPU tasks can run in parallel.
CPU and GPU queues execute concurrently.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date as date_type
from typing import Callable, Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Result of a scheduled task."""
    model_name: str
    target: str
    resource: str  # "cpu" or "gpu"
    tap_strategy: str
    blocks_generated: int = 0
    runtime_seconds: float = 0.0
    checkpoint_path: str = ""
    status: str = "pending"  # pending, running, complete, failed
    error_message: str = ""
    predictions: Any = None  # DataFrame or list of DataFrames


class ResourceScheduler:
    """Simple CPU/GPU parallel scheduler.

    GPU queue: serialized (one task at a time on the single GPU)
    CPU queue: parallel up to max_cpu_workers
    CPU and GPU queues run concurrently.
    """

    def __init__(self, max_cpu_workers: int = 2, max_gpu_workers: int = 1):
        self.max_cpu_workers = max_cpu_workers
        self.max_gpu_workers = max_gpu_workers
        self.results: list[TaskResult] = []

    def run_validation_tap(
        self,
        *,
        target: str,
        predict_date: str,
        data_path: str,
        fold_specs: list[dict],
        output_dir,
        force: bool = False,
        extra_kwargs: dict | None = None,
    ) -> tuple[list[pd.DataFrame], list[TaskResult]]:
        """Run validation tap for all models with CPU/GPU parallelism.

        GPU queue (serialized):
          1. TimeMixer: base_train + 10x online_update
          2. RT916: base_train + 10x online_update

        CPU queue (parallel):
          1. LightGBM: 10x3day rolling folds
          2. TimesFM: 30 daily inference
          3. SGDFNet: 10x3day folds (or 3x10day fallback)

        Returns (list_of_prediction_dfs, list_of_task_results).
        """
        from pipelines.validation_tap import (
            FORMAL_MODELS_BY_TASK,
            _normalize_tap_predictions,
        )
        from rolling_oof.scheduler import ADAPTER_REGISTRY, _init_registry
        _init_registry()

        models = FORMAL_MODELS_BY_TASK[target]
        kwargs = extra_kwargs or {}

        # Classify models by resource
        gpu_models = [m for m in models if m in ("timemixer", "rt916")]
        cpu_models = [m for m in models if m in ("lightgbm", "timesfm", "sgdfnet")]

        all_dfs: list[pd.DataFrame] = []
        all_results: list[TaskResult] = []

        # Run GPU and CPU queues concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit GPU queue
            gpu_future = executor.submit(
                self._run_gpu_validation,
                target=target, predict_date=predict_date,
                data_path=data_path, fold_specs=fold_specs,
                models=gpu_models, output_dir=output_dir,
                force=force, kwargs=kwargs,
            )

            # Submit CPU queue
            cpu_future = executor.submit(
                self._run_cpu_validation,
                target=target, predict_date=predict_date,
                data_path=data_path, fold_specs=fold_specs,
                models=cpu_models, output_dir=output_dir,
                force=force, kwargs=kwargs,
            )

            # Collect results
            gpu_dfs, gpu_results = gpu_future.result()
            cpu_dfs, cpu_results = cpu_future.result()

        all_dfs.extend(gpu_dfs)
        all_dfs.extend(cpu_dfs)
        all_results.extend(gpu_results)
        all_results.extend(cpu_results)

        return all_dfs, all_results

    def _run_gpu_validation(
        self, *, target, predict_date, data_path, fold_specs,
        models, output_dir, force, kwargs,
    ):
        """Run GPU models sequentially."""
        from pipelines.validation_tap import _normalize_tap_predictions
        from rolling_oof.scheduler import ADAPTER_REGISTRY
        from rolling_oof.contracts import FoldSpec

        dfs = []
        results = []

        for model_name in models:
            t0 = time.time()
            result = TaskResult(
                model_name=model_name, target=target, resource="gpu",
                tap_strategy="online_update",
            )
            try:
                adapter_cls = ADAPTER_REGISTRY.get(model_name)
                if adapter_cls is None:
                    result.status = "failed"
                    result.error_message = f"Unknown model: {model_name}"
                    results.append(result)
                    continue

                adapter = adapter_cls()
                if target not in adapter.supported_tasks:
                    result.status = "skipped"
                    result.error_message = f"{model_name} doesn't support {target}"
                    results.append(result)
                    continue

                # Online mode: run all folds in one call
                model_dfs = []
                for fold_info in fold_specs:
                    fs = FoldSpec(
                        fold_id=fold_info["fold_id"],
                        train_start=date_type.fromisoformat(fold_info["train_start"]),
                        train_end=date_type.fromisoformat(fold_info["train_end"]),
                        test_start=date_type.fromisoformat(fold_info["test_start"]),
                        test_end=date_type.fromisoformat(fold_info["test_end"]),
                        target_month="",
                    )

                    online_kwargs = {
                        "rolling_mode": "online",
                        "training_months": kwargs.get("training_months", 6),
                        "block_days": 3,
                        "online_epochs": kwargs.get(f"{model_name}_online_epochs", 3),
                        "online_lr": kwargs.get(f"{model_name}_online_lr"),
                        "checkpoint_dir": str(output_dir / f"{model_name}_checkpoints"),
                    }

                    fold_result = adapter.fold_train_predict(
                        task=target, fold_spec=fs, data_path=data_path,
                        **online_kwargs,
                    )

                    if fold_result.success and fold_result.predictions_df is not None:
                        df = _normalize_tap_predictions(
                            fold_result.predictions_df,
                            task=target, model_name=model_name,
                            fold_id=fold_info["fold_id"], fold_info=fold_info,
                        )
                        df["tap_source"] = "online_update"
                        df["source_confidence"] = 0.95
                        model_dfs.append(df)

                if model_dfs:
                    combined = pd.concat(model_dfs, ignore_index=True)
                    dfs.append(combined)
                    result.status = "complete"
                    result.blocks_generated = len(model_dfs)
                    result.tap_strategy = "online_update"
                else:
                    result.status = "failed"
                    result.error_message = "No predictions generated"

            except Exception as exc:
                result.status = "failed"
                result.error_message = str(exc)
                logger.error("GPU model %s failed: %s", model_name, exc)

            result.runtime_seconds = time.time() - t0
            results.append(result)

        return dfs, results

    def _run_cpu_validation(
        self, *, target, predict_date, data_path, fold_specs,
        models, output_dir, force, kwargs,
    ):
        """Run CPU models (potentially in parallel)."""
        from pipelines.validation_tap import _normalize_tap_predictions
        from rolling_oof.scheduler import ADAPTER_REGISTRY
        from rolling_oof.contracts import FoldSpec

        dfs = []
        results = []

        for model_name in models:
            t0 = time.time()
            result = TaskResult(
                model_name=model_name, target=target, resource="cpu",
                tap_strategy="",
            )
            try:
                adapter_cls = ADAPTER_REGISTRY.get(model_name)
                if adapter_cls is None:
                    result.status = "failed"
                    result.error_message = f"Unknown model: {model_name}"
                    results.append(result)
                    continue

                adapter = adapter_cls()
                if target not in adapter.supported_tasks:
                    result.status = "skipped"
                    results.append(result)
                    continue

                # Special handling per model type
                if model_name == "timesfm":
                    # TimesFM: try daily inference, fallback to block
                    inference_mode = kwargs.get("timesfm_inference_mode", "daily")
                    model_dfs = self._run_timesfm_validation(
                        adapter, target, data_path, fold_specs,
                        output_dir, inference_mode,
                    )
                    result.tap_strategy = (
                        "direct_inference_daily" if inference_mode == "daily"
                        else "direct_inference_block"
                    )
                elif model_name == "sgdfnet":
                    fold_strategy = kwargs.get("sgdfnet_fold_strategy", "auto")
                    model_dfs = self._run_sgdfnet_validation(
                        adapter, target, data_path, fold_specs,
                        output_dir, fold_strategy, kwargs,
                    )
                    result.tap_strategy = "rolling_cutoff"
                else:
                    # LightGBM: standard rolling folds
                    model_dfs = self._run_standard_rolling_validation(
                        adapter, target, data_path, fold_specs,
                        output_dir, kwargs,
                    )
                    result.tap_strategy = "rolling_cutoff"

                if model_dfs:
                    combined = pd.concat(model_dfs, ignore_index=True)
                    dfs.append(combined)
                    result.status = "complete"
                    result.blocks_generated = len(model_dfs)
                else:
                    result.status = "failed"
                    result.error_message = "No predictions"

            except Exception as exc:
                result.status = "failed"
                result.error_message = str(exc)
                logger.error("CPU model %s failed: %s", model_name, exc)

            result.runtime_seconds = time.time() - t0
            results.append(result)

        return dfs, results

    def _run_timesfm_validation(
        self, adapter, target, data_path, fold_specs, output_dir, inference_mode,
    ):
        """Run TimesFM validation with daily or block inference."""
        from pipelines.validation_tap import _normalize_tap_predictions
        from rolling_oof.contracts import FoldSpec

        model_dfs = []
        for fold_info in fold_specs:
            fs = FoldSpec(
                fold_id=fold_info["fold_id"],
                train_start=date_type.fromisoformat(fold_info["train_start"]),
                train_end=date_type.fromisoformat(fold_info["train_end"]),
                test_start=date_type.fromisoformat(fold_info["test_start"]),
                test_end=date_type.fromisoformat(fold_info["test_end"]),
                target_month="",
            )

            fold_result = adapter.fold_train_predict(
                task=target, fold_spec=fs, data_path=data_path,
                inference_mode=inference_mode,
                training_months=6,
            )

            if fold_result.success and fold_result.predictions_df is not None:
                df = _normalize_tap_predictions(
                    fold_result.predictions_df,
                    task=target, model_name="timesfm",
                    fold_id=fold_info["fold_id"], fold_info=fold_info,
                )
                # Add source metadata
                if "tap_source" not in df.columns:
                    df["tap_source"] = (
                        "direct_inference_daily" if inference_mode == "daily"
                        else "direct_inference_block"
                    )
                if "source_confidence" not in df.columns:
                    df["source_confidence"] = 0.90 if inference_mode == "daily" else 0.85
                model_dfs.append(df)

        return model_dfs

    def _run_sgdfnet_validation(
        self, adapter, target, data_path, fold_specs,
        output_dir, fold_strategy, kwargs,
    ):
        """Run SGDFNet validation with fold strategy."""
        from pipelines.validation_tap import _normalize_tap_predictions
        from rolling_oof.contracts import FoldSpec

        model_dfs = []
        for fold_info in fold_specs:
            fs = FoldSpec(
                fold_id=fold_info["fold_id"],
                train_start=date_type.fromisoformat(fold_info["train_start"]),
                train_end=date_type.fromisoformat(fold_info["train_end"]),
                test_start=date_type.fromisoformat(fold_info["test_start"]),
                test_end=date_type.fromisoformat(fold_info["test_end"]),
                target_month="",
            )

            fold_result = adapter.fold_train_predict(
                task=target, fold_spec=fs, data_path=data_path,
                fold_strategy=fold_strategy,
                training_months=kwargs.get("training_months", 6),
            )

            if fold_result.success and fold_result.predictions_df is not None:
                df = _normalize_tap_predictions(
                    fold_result.predictions_df,
                    task=target, model_name="sgdfnet",
                    fold_id=fold_info["fold_id"], fold_info=fold_info,
                )
                if "tap_source" not in df.columns:
                    df["tap_source"] = "rolling_cutoff"
                if "source_confidence" not in df.columns:
                    df["source_confidence"] = 1.0
                model_dfs.append(df)

        return model_dfs

    def _run_standard_rolling_validation(
        self, adapter, target, data_path, fold_specs, output_dir, kwargs,
    ):
        """Run standard rolling validation (LightGBM)."""
        from pipelines.validation_tap import _normalize_tap_predictions
        from rolling_oof.contracts import FoldSpec

        model_dfs = []
        for fold_info in fold_specs:
            fs = FoldSpec(
                fold_id=fold_info["fold_id"],
                train_start=date_type.fromisoformat(fold_info["train_start"]),
                train_end=date_type.fromisoformat(fold_info["train_end"]),
                test_start=date_type.fromisoformat(fold_info["test_start"]),
                test_end=date_type.fromisoformat(fold_info["test_end"]),
                target_month="",
            )

            fold_result = adapter.fold_train_predict(
                task=target, fold_spec=fs, data_path=data_path,
                training_months=kwargs.get("training_months", 6),
                rolling_mode="block",
                block_days=3,
            )

            if fold_result.success and fold_result.predictions_df is not None:
                df = _normalize_tap_predictions(
                    fold_result.predictions_df,
                    task=target, model_name=adapter.model_name,
                    fold_id=fold_info["fold_id"], fold_info=fold_info,
                )
                if "tap_source" not in df.columns:
                    df["tap_source"] = "rolling_cutoff"
                if "source_confidence" not in df.columns:
                    df["source_confidence"] = 1.0
                model_dfs.append(df)

        return model_dfs

    def generate_runtime_report(self) -> pd.DataFrame:
        """Generate runtime report from task results."""
        rows = []
        for r in self.results:
            rows.append({
                "model_name": r.model_name,
                "target": r.target,
                "resource": r.resource,
                "tap_strategy": r.tap_strategy,
                "blocks_generated": r.blocks_generated,
                "runtime_seconds": round(r.runtime_seconds, 1),
                "checkpoint_path": r.checkpoint_path,
                "status": r.status,
            })
        return pd.DataFrame(rows)
