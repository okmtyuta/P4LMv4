from typing import List

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class FloaterPositionalEncoder(DataProcess):
    """FLOATER: 連続力学系による位置エンコーディング（torchdiffeq で ODE を数値積分）。

    - 連続状態 p(t) ∈ R^D を ODE: dp/dt = f(t, p; θ) で定義し、離散位置 i に対して
      t_i = i·Δt の解 p(t_i) を加算的に注入: out[i] = x[i] + p(t_i)。
    - f はユーザーが指定する `nn.Module`（`forward(t, y)` 形式）。
    - ODE 解法は `torchdiffeq` の `odeint` または `odeint_adjoint` を利用。

    Args:
        dim: 特徴次元 D。
        dynamics: `forward(t, y)->dy/dt` を満たす NN。
        delta_t: Δt (>0)。
        method: 例 `"dopri5"`, `"rk4"`, `"adams"` など。
        rtol: 相対誤差許容。
        atol: 絶対誤差許容。
        use_adjoint: True なら `odeint_adjoint` を使用。
    """

    def __init__(
        self,
        dim: int,
        dynamics: nn.Module,
        delta_t: float,
        method: str = "dopri5",
        rtol: float = 1e-3,
        atol: float = 1e-6,
        use_adjoint: bool = False,
    ) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        if delta_t <= 0.0:
            raise ValueError("delta_t must be > 0")
        if rtol <= 0.0 or atol <= 0.0:
            raise ValueError("rtol/atol must be > 0")

        self._D = dim
        self._dt = delta_t
        self._method = method
        self._rtol = rtol
        self._atol = atol
        self._use_adj = use_adjoint

        # torchdiffeq が期待するシグネチャ (t, y) -> dy/dt の NN をそのまま受け取る
        self._dynamics: nn.Module = dynamics
        # 学習可能な初期状態 p(0)
        self._p0 = nn.Parameter(torch.zeros(self._D, dtype=torch.float32))

    @property
    def dim_factor(self) -> int:
        return 1

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        # dynamics 側のパラメータも一緒に最適化できるよう expose
        return list(self._dynamics.parameters()) + [self._p0]

    def _solve_states(self, L: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # t: [0, Δt, 2Δt, ..., LΔt] の (L+1,)。最初の 0 は p(0) 用。
        t = torch.arange(0, L + 1, dtype=dtype, device=device) * self._dt
        y0 = self._p0.to(device=device, dtype=dtype)

        # dynamics を同じデバイスへ（訓練時は事前に .to(...) しておくのが理想だが安全側で同期）
        # dynamics を同じデバイス/型へ移動（in-place）
        self._dynamics.to(device=device, dtype=dtype)

        solver = odeint_adjoint if self._use_adj else odeint
        # y: (L+1, D)
        y = solver(
            self._dynamics,
            y0,
            t,
            method=self._method,
            rtol=self._rtol,
            atol=self._atol,
        )
        # 先頭を除いた (L, D) を返す
        return y[1:, :]

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()
        L, D = reps.shape
        if D != self._D:
            raise ValueError(f"dimension mismatch: expected {self._D}, got {D}")

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        pos = self._solve_states(L=L, device=device, dtype=dtype)  # (L, D)
        out = reps + pos
        return protein.set_processed(processed=out)
