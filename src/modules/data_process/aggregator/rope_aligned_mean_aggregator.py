"""
RoPE 位相整合平均 Aggregator。

- RoPE の角度に基づき、各位置ベクトルを共通基準へ「逆回転」してから平均します。
- これにより、同一パターンが配列内のどこに現れても（RoPE 前提の）位相差を打ち消しつつ集約できます。

前提:
- 入力は (L, D) の 2 次元テンソル。
- RoPE に準拠した 2 次元ペアごとの回転を仮定し、奇数最終チャネルは無変換で保持します。
"""

from __future__ import annotations

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class RoPEAlignedMeanAggregator(DataProcess):
    """RoPE の位相を打ち消してから平均する集約器。

    使いどころ:
    - 事前に RoPE を適用した系列表現を、位相ずれの影響を減らして固定長化したい場合。
    - RoPE を適用していない場合でも、明示的な位置に応じた回転（−角度）を掛ける集約として機能します。
    """

    def __init__(self, theta_base: float, reversed: bool) -> None:
        """RoPE 基底 `theta_base` と、位置カウントの向き `reversed` を指定する。

        - `theta_base`: RoPE の基底周波数（一般に 10000.0 など）。
        - `reversed`: 位置を L..1 と数える場合は True、1..L なら False。
        """
        if theta_base <= 0.0:
            raise ValueError("theta_base must be positive")
        self._theta_base = float(theta_base)
        self._reversed = bool(reversed)

    @property
    def dim_factor(self) -> int:
        """出力次元は D（1倍）。"""
        return 1

    def _positions(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """1..L もしくは L..1 の位置ベクトルを返す。"""
        if self._reversed:
            return torch.arange(length, 0, -1, dtype=dtype, device=device)
        return torch.arange(1, length + 1, dtype=dtype, device=device)

    def _cos_sin(
        self, length: int, dim: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE の cos/sin 行列を生成する。

        - 角度行列は形状 (L, half)。
        - `inv_freq = base^{-2m/D}` を用いる標準形。
        """
        half = dim // 2
        positions = self._positions(length=length, device=device, dtype=dtype)  # (L,)
        p_col = positions[:, None]  # (L, 1)
        m = torch.arange(0, half, dtype=dtype, device=device)
        inv_freq = self._theta_base ** (-2.0 * m / float(dim))  # (half,)
        angles = p_col * inv_freq[None, :]  # (L, half)
        return torch.cos(angles), torch.sin(angles)

    def _inverse_rotate(self, x: torch.Tensor) -> torch.Tensor:
        """RoPE の逆回転（角度に符号反転）を位置ごとに適用する。

        入力/出力: (L, D)
        """
        L, D = x.shape
        if D < 2:
            return x  # 最低限の安全策（D<2 は回転対象なし）

        cos, sin = self._cos_sin(length=L, dim=D, device=x.device, dtype=x.dtype)  # (L, half)
        half = D // 2

        # 偶数/奇数チャネルをペアとして取り出す
        even_part = x[:, 0 : 2 * half : 2]
        odd_part = x[:, 1 : 2 * half : 2]

        # 逆回転 R(-θ) を適用
        # [e'; o'] = [ e*cos + o*sin, -e*sin + o*cos ]
        inv_even = even_part * cos + odd_part * sin
        inv_odd = -even_part * sin + odd_part * cos

        out = x.clone()
        out[:, 0 : 2 * half : 2] = inv_even
        out[:, 1 : 2 * half : 2] = inv_odd
        return out

    def _act(self, protein: Protein) -> Protein:
        """位相整合（逆回転）→ 列方向平均で (D,) を返す。"""
        x = protein.get_processed()  # (L, D)
        if x.ndim != 2:
            raise ValueError("processed must be a 2D tensor (L, D)")

        L, D = x.shape
        if L == 0:
            # 空系列への安全策
            return protein.set_processed(processed=torch.zeros(D, dtype=x.dtype, device=x.device))

        x_aligned = self._inverse_rotate(x)
        y = torch.mean(x_aligned, dim=0)  # (D,)
        return protein.set_processed(processed=y)
