from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class WeightedMeanAggregator(DataProcess):
    """内容×位置の学習スコアで重み付け平均を行う集約器。

    - 入力: processed (L, D)
    - 出力: (D,)
    - 重み: w = softmax( (X @ v) / tau + b[pos] ) over 長さ次元
      * v ∈ R^D: 内容ベースの投影ベクトル（学習）
      * b ∈ R^{max_length}: 位置バイアス（学習, L>max_length は末尾再利用）
      * tau > 0: 温度（学習, softplus で正値化）
    初期化は平均に近い挙動（v≈0, b=0, tau=10）から開始。
    """

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
        return [self._v, self._b, self._log_tau]

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
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
