import math
from typing import Final

import torch
from torch import nn


class SimpleGatedSineDynamics(nn.Module):
    """時間埋め込み(sin/cos) + ゲート付きMLP + 減衰項のシンプルな連続力学。

    dy/dt = W2·SiLU(W1·y + V·phi(t)) - lambda·y

    - `phi(t)` は `omega` を対数均等配置した複数周波数の `sin, cos` 連結。
    - 減衰 `lambda` は ReLU により非負に制約して安定性を確保。
    - FLOATER の `dim` と一致するように初期化してください。
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        num_freqs: int,
        omega_min: float,
        omega_max: float,
        damping: float,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if hidden <= 0:
            raise ValueError("hidden must be positive")
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive")
        if omega_min <= 0.0 or omega_max <= 0.0 or omega_min >= omega_max:
            raise ValueError("0 < omega_min < omega_max must hold")

        self._dim: Final[int] = dim
        self._hidden: Final[int] = hidden
        self._num_freqs: Final[int] = num_freqs

        omegas = torch.logspace(
            start=math.log10(omega_min), end=math.log10(omega_max), steps=num_freqs, dtype=torch.float32
        )
        self._omegas: torch.Tensor
        self.register_buffer("_omegas", omegas)

        self._fc_y = nn.Linear(dim, hidden, bias=True)
        self._fc_t = nn.Linear(2 * num_freqs, hidden, bias=False)
        self._fc_out = nn.Linear(hidden, dim, bias=True)
        self._damping = nn.Parameter(torch.tensor(float(damping), dtype=torch.float32))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self._fc_y.weight)
        nn.init.zeros_(self._fc_y.bias)
        nn.init.xavier_uniform_(self._fc_t.weight)
        nn.init.xavier_uniform_(self._fc_out.weight)
        nn.init.zeros_(self._fc_out.bias)

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # t: () or (1,), returns (2*num_freqs,)
        t = t.to(dtype=self._omegas.dtype)
        wt = self._omegas * t
        return torch.cat([torch.sin(wt), torch.cos(wt)], dim=0)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        te = self._time_embed(t)
        z = self._fc_y(y) + self._fc_t(te)
        h = torch.nn.functional.silu(z)
        lam = torch.relu(self._damping)
        dy = self._fc_out(h) - lam * y
        return dy


__all__ = [
    "SimpleGatedSineDynamics",
]
