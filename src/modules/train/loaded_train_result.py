#!/usr/bin/env python3
"""
学習結果HDF5（現行フォーマット: history/belle をフル保存）を読み出すスクリプト。

使い方:
    python read.py [path/to/train_result.h5]

引数未指定時は以下を既定とします:
    outputs/experiments/plasma_lumos_1h/sinusoidal/additive/train_result.h5

出力:
    - current_epoch / patience / belle_epoch
    - history の各エポックに対する validate の合算Pearson（accuracy）
    - belle エポックの同要約
"""

from __future__ import annotations

from dataclasses import dataclass

import h5py

from src.modules.train.train_result import EpochSummary, TrainingHistory


@dataclass
class LoadedTrainResult:
    """学習結果HDF5の読み取り結果を保持するデータクラス。"""

    path: str
    current_epoch: int
    patience: int
    belle_epoch: int
    history: TrainingHistory
    belle: EpochSummary

    @classmethod
    def load(cls, file_path: str) -> "LoadedTrainResult":
        """HDF5 ファイルから学習結果を読み込み、`LoadedTrainResult` を返す。"""
        with h5py.File(file_path, "r") as f:
            root = f["result"]

            current_epoch = int(root.attrs.get("current_epoch", 0))
            patience = int(root.attrs.get("patience", 0))

            belle_epoch: int = int(root.attrs["belle_epoch"])  # type: ignore[assignment]
            history = TrainingHistory.from_hdf5_group(root["history"])  # type: ignore[arg-type]
            belle: EpochSummary = EpochSummary.from_hdf5_group(root["belle"])  # type: ignore[arg-type]

        return cls(
            path=file_path,
            current_epoch=current_epoch,
            patience=patience,
            belle_epoch=belle_epoch,
            history=history,
            belle=belle,
        )
