# -*- coding: utf-8 -*-
"""OOF 运行目录布局管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OofRunLayout:
    """定义 oof_runs/ 下的目录结构。

    目录结构:
        oof_runs/
          {pool_id}/
            manifest.json
            protocol_audit.json
            oof_long_table.csv
            folds/
              fold_0/
                fold_spec.json
                audits/
                  {model}__{task}__audit.json
                {model}/
                  {task}/
                    fold_{id}_{task}_long.csv
                    fold_{id}_{task}_raw.csv
              ...
            escort/
              escort_{date}_{task}.csv
              escort_{date}_long.csv
    """

    pool_root: Path
    pool_id: str

    # --- 属性 ---

    @property
    def manifest_path(self) -> Path:
        return self.pool_root / "manifest.json"

    @property
    def audit_path(self) -> Path:
        return self.pool_root / "protocol_audit.json"

    @property
    def long_table_path(self) -> Path:
        return self.pool_root / "oof_long_table.csv"

    @property
    def folds_dir(self) -> Path:
        return self.pool_root / "folds"

    @property
    def escort_dir(self) -> Path:
        return self.pool_root / "escort"

    # --- fold 级别路径 ---

    def fold_dir(self, fold_id: int) -> Path:
        return self.folds_dir / f"fold_{fold_id}"

    def fold_spec_path(self, fold_id: int) -> Path:
        return self.fold_dir(fold_id) / "fold_spec.json"

    def fold_audits_dir(self, fold_id: int) -> Path:
        return self.fold_dir(fold_id) / "audits"

    def fold_audit_path(self, fold_id: int, model_name: str, task: str) -> Path:
        return self.fold_audits_dir(fold_id) / f"{model_name}__{task}__audit.json"

    def model_task_dir(self, fold_id: int, model_name: str, task: str) -> Path:
        return self.fold_dir(fold_id) / model_name / task

    def fold_long_path(self, fold_id: int, model_name: str, task: str) -> Path:
        return self.model_task_dir(fold_id, model_name, task) / f"fold_{fold_id}_{task}_long.csv"

    def fold_raw_path(self, fold_id: int, model_name: str, task: str) -> Path:
        return self.model_task_dir(fold_id, model_name, task) / f"fold_{fold_id}_{task}_raw.csv"

    # --- escort 路径 ---

    def escort_task_path(self, target_date: str, task: str) -> Path:
        return self.escort_dir / f"escort_{target_date}_{task}.csv"

    def escort_long_path(self, target_date: str) -> Path:
        return self.escort_dir / f"escort_{target_date}_long.csv"

    # --- 目录创建 ---

    def ensure_dirs(self, fold_id: int = -1) -> None:
        """确保所有必要目录存在。"""
        self.pool_root.mkdir(parents=True, exist_ok=True)
        self.folds_dir.mkdir(parents=True, exist_ok=True)
        self.escort_dir.mkdir(parents=True, exist_ok=True)
        if fold_id >= 0:
            self.fold_dir(fold_id).mkdir(parents=True, exist_ok=True)
            self.fold_audits_dir(fold_id).mkdir(parents=True, exist_ok=True)
