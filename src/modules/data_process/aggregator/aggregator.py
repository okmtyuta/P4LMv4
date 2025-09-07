"""
系列表現を固定長ベクトルへ集約する基本 Aggregator。

- 現在は単純平均（mean）のみを提供。
"""

from typing import Literal

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein

AggregateMethod = Literal["mean"]


class Aggregator(DataProcess):
    """集約処理を行う `DataProcess` 実装（平均のみ）。"""

    def __init__(self, method: AggregateMethod) -> None:
        """集約方式を指定して初期化する。"""
        self._method: AggregateMethod = method

    @property
    def dim_factor(self) -> int:
        """出力次元は入力と同一（＝1倍）。"""
        return 1

    def _act(self, protein: Protein) -> Protein:
        """集約を実行する。現在は平均のみ対応。"""
        if self._method == "mean":
            processed = protein.get_processed()
            mean = self._mean(input=processed)
            return protein.set_processed(processed=mean)

        raise ValueError(f"Unknown aggregate method: {self._method}")

    def _mean(self, input: torch.Tensor) -> torch.Tensor:
        """系列次元に沿った単純平均を返す。"""
        return torch.mean(input=input, dim=0)
