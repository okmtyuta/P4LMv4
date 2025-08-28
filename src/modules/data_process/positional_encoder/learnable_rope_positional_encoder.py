from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class _BaseLearnableRoPEPositionalEncoder(DataProcess):
    """Learnable Rotary Positional Embedding (RoPE).

    - 周波数ベクトル `inv_freq` を学習（ペアごとに1つ）。
    - 入力は `Protein.get_processed()` を読み、逐次合成に対応。
    - 偶数ペアごとに2次元回転を適用、奇数次元末尾はそのまま。
    """

    def __init__(self, dim: int, theta_base: float) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._D = int(dim)
        self._half = self._D // 2

        # RoPE の初期周波数 inv_freq = base^{-2m/D} を対数で保持（>0の安定確保）
        if self._half > 0:
            m = torch.arange(0, self._half, dtype=torch.float32)
            init_inv = float(theta_base) ** (-2.0 * m / float(self._D))
            init_log_inv = torch.log(init_inv)
        else:
            init_log_inv = torch.empty(0, dtype=torch.float32)

        self._log_inv_freq = nn.Parameter(init_log_inv)  # (half,)

    @property
    def dim_factor(self) -> int:
        return 1

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        return [self._log_inv_freq]

    def _positions(self, length: int, reversed: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if reversed:
            return torch.arange(length, 0, -1, dtype=dtype, device=device)
        return torch.arange(1, length + 1, dtype=dtype, device=device)

    def _cos_sin(
        self, length: int, reversed: bool, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._half == 0:
            cos = torch.ones((length, 0), dtype=dtype, device=device)
            sin = torch.zeros((length, 0), dtype=dtype, device=device)
            return cos, sin

        p = self._positions(length=length, reversed=reversed, device=device, dtype=dtype)  # (L,)
        p_col = p[:, None]  # (L, 1)

        inv_freq = torch.exp(self._log_inv_freq).to(device=device, dtype=dtype)  # (half,)
        angles = p_col * inv_freq[None, :]  # (L, half)
        return torch.cos(angles), torch.sin(angles)

    def _apply_rope(self, x: torch.Tensor, reversed: bool) -> torch.Tensor:
        L, D = x.shape
        if D != self._D:
            raise ValueError(f"dimension mismatch: expected {self._D}, got {D}")

        cos, sin = self._cos_sin(length=L, reversed=reversed, device=x.device, dtype=x.dtype)
        half = self._half
        if half == 0:
            return x

        x_even = x[:, 0 : 2 * half : 2]
        x_odd = x[:, 1 : 2 * half : 2]

        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = x.clone()
        out[:, 0 : 2 * half : 2] = rot_even
        out[:, 1 : 2 * half : 2] = rot_odd
        return out


class LearnableRoPEPositionalEncoder(_BaseLearnableRoPEPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        x = protein.get_processed()  # (L, D)
        out = self._apply_rope(x, False)
        return protein.set_processed(processed=out)


class ReversedLearnableRoPEPositionalEncoder(_BaseLearnableRoPEPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        x = protein.get_processed()
        out = self._apply_rope(x, True)
        return protein.set_processed(processed=out)


class BidirectionalLearnableRoPEPositionalEncoder(_BaseLearnableRoPEPositionalEncoder):
    @property
    def dim_factor(self) -> int:  # 2D へ拡張
        return 2

    def _act(self, protein: Protein) -> Protein:
        x = protein.get_processed()
        out_n = self._apply_rope(x, False)
        out_r = self._apply_rope(x, True)
        out = torch.cat([out_n, out_r], dim=1)
        return protein.set_processed(processed=out)
