# -*- coding: utf-8 -*-
"""BaseRollingAdapter —— 所有模型 rolling-origin 适配器的抽象基类。

与 pipelines/base.py 的 BaseModelPipeline 独立并存，不修改其行为。
"""

from __future__ import annotations

import abc
from typing import Optional

import pandas as pd

from rolling_oof.contracts import FoldResult, FoldSpec


class BaseRollingAdapter(abc.ABC):
    """rolling-origin 模型适配器抽象基类。

    每个模型实现 fold_train_predict() 方法，满足统一协议：
    1. 训练数据: data[train_start, train_end] (inclusive)
    2. 验证集: 训练数据的时间顺序尾部，严禁随机打乱
    3. 预测: data[test_start, test_end] (inclusive)
    4. 特征截止: 所有 lag/rolling 特征以 train_end 时刻为截止点
    5. 返回: FoldResult，其中 predictions_df 为标准 long-table

    Attributes
    ----------
    model_name : str
        模型名称（与 registry 中的 key 一致）。
    device_type : str
        "cpu" 或 "gpu"。
    """

    model_name: str = ""
    device_type: str = "cpu"
    supported_tasks: tuple[str, ...] = ("dayahead", "realtime")

    @abc.abstractmethod
    def fold_train_predict(
        self,
        task: str,
        fold_spec: FoldSpec,
        data_path: str,
        **kwargs,
    ) -> FoldResult:
        """核心方法：对单个 fold 执行训练+预测。

        Parameters
        ----------
        task : str
            "dayahead" 或 "realtime"。
        fold_spec : FoldSpec
            fold 参数，包含 train_start/end 和 test_start/end。
        data_path : str
            原始数据文件路径。
        **kwargs
            额外参数传递给底层模型（如 training_months, val_ratio 等）。

        Returns
        -------
        FoldResult
        """
        ...

    # ------------------------------------------------------------------
    # 通用辅助方法（子类可重写或直接使用）
    # ------------------------------------------------------------------

    def _load_data(self, data_path: str) -> pd.DataFrame:
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

    def _ensure_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """确保关键列使用正确的类型。"""
        if "ds" in df.columns:
            df["ds"] = pd.to_datetime(df["ds"])
        if "target_day" in df.columns:
            df["target_day"] = pd.to_datetime(df["target_day"]).dt.strftime("%Y-%m-%d")
        return df

    def _format_result(
        self,
        df: pd.DataFrame,
        task: str,
        fold_spec: FoldSpec,
        metrics: Optional[dict] = None,
    ) -> FoldResult:
        """将原始预测 DataFrame 封装为 FoldResult。"""
        return FoldResult(
            fold_id=fold_spec.fold_id,
            model_name=self.model_name,
            task=task,
            fold_spec=fold_spec,
            predictions_df=df,
            train_metrics=metrics or {},
            success=True,
        )
