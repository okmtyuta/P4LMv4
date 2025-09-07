"""
学習クエリを用いた注意プーリング集約。

- 各クエリごとに長さ方向の softmax 重みを計算し、加重平均した (D,) を得ます。
- これを K 個連結して (K·D,) を出力します。
"""

from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class AttentionPoolingAggregator(DataProcess):
    """学習クエリによる注意プーリング集約。"""

    def __init__(self, dim: int, num_queries: int, temperature: float) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self._D = int(dim)
        self._K = int(num_queries)
        self._tau = float(temperature)

        # 平均に近い初期挙動: 小さな乱数で初期化（≈一様重み）
        self._Q = nn.Parameter(0.01 * torch.randn(self._K, self._D, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習クエリ行列 Q を返す。"""
        return [self._Q]

    @property
    def dim_factor(self) -> int:  # 出力は K×D
        """出力次元は K 倍。"""
        return self._K

    def _act(self, protein: Protein) -> Protein:
        """注意プーリングを適用して (K·D,) を出力する。"""
        x = protein.get_processed()  # (L, D)
        if x.ndim != 2:
            raise ValueError("processed must be a 2D tensor (L, D)")
        L, D = x.shape
        if D != self._D:
            raise ValueError(f"dimension mismatch: expected {self._D}, got {D}")

        if L == 0:
            # 安全策: 空系列ならゼロを返す
            return protein.set_processed(processed=torch.zeros(self._K * D, dtype=x.dtype, device=x.device))

        Q = self._Q.to(device=x.device, dtype=x.dtype)  # (K, D)

        # スコア (L, K)
        # scores[p,k] = x[p,:] @ Q[k,:]^T / tau
        scores = (x @ Q.T) / self._tau

        # 長さ方向に正規化（各クエリごとに和が1）
        weights = torch.softmax(scores, dim=0)  # (L, K)

        # 集約: Y = weights^T @ x -> (K, D)
        Y = weights.T @ x  # (K, D)

        out = Y.reshape(self._K * D)
        return protein.set_processed(processed=out)
