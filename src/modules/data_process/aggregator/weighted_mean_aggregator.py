"""
内容スコアと位置バイアスを組み合わせ、softmax 重みで平均をとる Aggregator。

- 内容スコア: 内積（特徴と学習ベクトル）を温度でスケーリング。
- 位置バイアス: シーケンス位置ごとの学習スカラ（`max_length` 超は末尾値を再利用）。
- 重みは長さ方向で softmax 正規化し、加重平均を求めます。
"""

from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class WeightedMeanAggregator(DataProcess):
    """内容×位置スコアで重み付け平均を行う集約器。"""

    def __init__(self, dim: int, max_length: int) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        self._D = int(dim)
        self._Lmax = int(max_length)

        # 平均に近い初期挙動：v~0, b=0, tau 大きめ
        self._v = nn.Parameter(torch.zeros(self._D, dtype=torch.float32))
        self._b = nn.Parameter(torch.zeros(self._Lmax, dtype=torch.float32))
        self._log_tau = nn.Parameter(torch.tensor(2.3025851, dtype=torch.float32))  # log(10) ≈ 2.3026

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習対象パラメータを返す。"""
        return [self._v, self._b, self._log_tau]

    @property
    def dim_factor(self) -> int:
        """出力次元は D（=1倍）。"""
        return 1

    def _act(self, protein: Protein) -> Protein:
        """重み付け平均を計算し、(D,) ベクトルへ集約する。"""
        x = protein.get_processed()  # (L, D)
        L, D = x.shape
        if D != self._D:
            raise ValueError(f"dimension mismatch: expected {self._D}, got {D}")

        device = x.device
        dtype = x.dtype if x.dtype.is_floating_point else torch.float32

        # 内容スコア (L,)
        v = self._v.to(device=device, dtype=dtype)
        content = x @ v  # (L,)

        # 位置バイアス (L,)
        idx = torch.arange(L, device=device)
        idx = torch.clamp(idx, max=self._Lmax - 1)
        b = self._b.to(device=device, dtype=dtype)
        pos_bias = b[idx]

        # 温度（正値）。大きいほど平均に近い。
        tau = torch.nn.functional.softplus(self._log_tau).to(device=device, dtype=dtype)

        scores = (content / tau) + pos_bias  # (L,)
        weights = torch.softmax(scores, dim=0)  # (L,)

        y = torch.sum(weights[:, None] * x, dim=0)  # (D,)
        return protein.set_processed(processed=y)
