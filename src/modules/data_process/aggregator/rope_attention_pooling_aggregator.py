"""
RoPE整合型 注意プーリング Aggregator。

目的:
- RoPE により位相を持つ系列表現に対し、クエリも同位相で回転させて内積を取り、
  絶対位置に不変な内容ベースの重み付けを実現する。

挙動:
- クエリ行列 `Q`（K×D）を各位置 p の角度で回転し `Q_p = R(θ_p)·Q` を作る。
- スコア `scores[p,k] = <x_p, Q_p[k]> / τ` を計算し、長さ方向 softmax で正規化。
- 加重平均 `Y[k] = Σ_p w[p,k]·x_p` を計算し、(K·D,) に連結して出力。
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class RoPEAttentionPoolingAggregator(DataProcess):
    """RoPE 位相と整合する注意プーリング集約。"""

    def __init__(self, dim: int, num_queries: int, theta_base: float, reversed: bool, temperature: float) -> None:
        """特徴次元・クエリ数・RoPE基底・向き・温度を指定して初期化する。"""
        if dim <= 0:
            raise ValueError("dim must be positive")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        if theta_base <= 0.0:
            raise ValueError("theta_base must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        self._D = int(dim)
        self._K = int(num_queries)
        self._theta_base = float(theta_base)
        self._reversed = bool(reversed)
        self._tau = float(temperature)

        # 初期は平均に近い挙動: 小さな乱数で初期化
        self._Q = nn.Parameter(0.01 * torch.randn(self._K, self._D, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習対象パラメータ `Q` を返す。"""
        return [self._Q]

    @property
    def dim_factor(self) -> int:
        """出力は K×D に連結するため K 倍。"""
        return self._K

    def _positions(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """1..L または L..1 の位置テンソルを返す。"""
        if self._reversed:
            return torch.arange(length, 0, -1, dtype=dtype, device=device)
        return torch.arange(1, length + 1, dtype=dtype, device=device)

    def _cos_sin(
        self, length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE の cos/sin 角度行列を生成（形状: (L, half)）。"""
        half = dim // 2
        positions = self._positions(length=length, device=device, dtype=dtype)  # (L,)
        p_col = positions[:, None]
        m = torch.arange(0, half, dtype=dtype, device=device)
        inv_freq = self._theta_base ** (-2.0 * m / float(dim))  # (half,)
        angles = p_col * inv_freq[None, :]
        return torch.cos(angles), torch.sin(angles)

    def _rotate_queries(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """各位置の角度でクエリを回転する。

        入力:
        - q: (K, D)
        - cos/sin: (L, half)

        出力: (L, K, D)
        """
        K, D = q.shape
        half = D // 2
        q_even = q[:, 0 : 2 * half : 2]  # (K, half)
        q_odd = q[:, 1 : 2 * half : 2]  # (K, half)

        # 位置次元 L を持つ回転
        rot_even = q_even[None, :, :] * cos[:, None, :] - q_odd[None, :, :] * sin[:, None, :]
        rot_odd = q_even[None, :, :] * sin[:, None, :] + q_odd[None, :, :] * cos[:, None, :]

        out = q[None, :, :].repeat(cos.shape[0], 1, 1)  # (L, K, D) 複製
        out[:, :, 0 : 2 * half : 2] = rot_even
        out[:, :, 1 : 2 * half : 2] = rot_odd
        return out

    def _act(self, protein: Protein) -> Protein:
        """RoPE 整合注意プーリングを適用して (K·D,) を返す。"""
        x = protein.get_processed()  # (L, D)
        if x.ndim != 2:
            raise ValueError("processed must be a 2D tensor (L, D)")
        L, D = x.shape
        if D != self._D:
            raise ValueError(f"dimension mismatch: expected {self._D}, got {D}")
        if L == 0:
            return protein.set_processed(processed=torch.zeros(self._K * D, dtype=x.dtype, device=x.device))

        dtype = x.dtype if x.dtype.is_floating_point else torch.float32
        device = x.device

        cos, sin = self._cos_sin(length=L, dim=D, device=device, dtype=dtype)
        q = self._Q.to(device=device, dtype=dtype)
        q_rot = self._rotate_queries(q=q, cos=cos, sin=sin)  # (L, K, D)

        # スコア: s[p,k] = <x[p], q_rot[p,k]> / tau
        scores = torch.einsum("ld,lkd->lk", x.to(dtype), q_rot) / self._tau  # (L, K)
        weights = torch.softmax(scores, dim=0)  # (L, K)

        # 集約: (K, D)
        y = torch.einsum("lk,ld->kd", weights, x)  # (K, D)
        out = y.reshape(self._K * D)
        return protein.set_processed(processed=out)
