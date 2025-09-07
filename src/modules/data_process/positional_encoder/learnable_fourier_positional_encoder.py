"""
学習可能なフーリエ基底を用いてスカラーゲートを生成し、各位置の特徴に乗算するエンコーダ。
"""

import math
from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class LearnableFourierPositionalEncoder(DataProcess):
    """学習可能な周波数・位相から時間特徴を作り、線形投影でスカラーを得て乗算する。"""

    def __init__(self, num_bases: int, min_period: float, max_period: float, projection_scale: float) -> None:
        """基底数と期間範囲、出力スケールを受け取り初期化する。"""
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if min_period <= 0 or max_period <= 0:
            raise ValueError("periods must be positive")
        if not (min_period < max_period):
            raise ValueError("min_period must be < max_period")

        self._K = int(num_bases)
        self._scale = float(projection_scale)

        # Initialize frequencies using log-spaced periods between [min_period, max_period]
        periods = torch.logspace(
            start=math.log10(min_period), end=math.log10(max_period), steps=self._K, dtype=torch.float32
        )
        init_w = (2.0 * math.pi) / periods  # angular frequencies
        init_log_w = torch.log(init_w)  # store in log-space for stability/positivity

        # Learnable parameters
        self._log_w = nn.Parameter(init_log_w)  # (K,)
        self._phase = nn.Parameter(torch.zeros(self._K, dtype=torch.float32))  # (K,)
        self._proj = nn.Parameter(torch.randn(2 * self._K, dtype=torch.float32) * 0.01)  # (2K,)

    # Optional helper to expose parameters for optimizers if統合する場合
    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習対象パラメータ（周波数・位相・射影）を返す。"""
        return [self._log_w, self._phase, self._proj]

    @property
    def dim_factor(self) -> int:
        """出力次元は D（=1倍）。"""
        return 1

    def _act(self, protein: Protein) -> Protein:
        """位置ごとにゲートを計算し、(L, D) に乗算する。"""
        reps = protein.get_representations()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        # Positions 1..L
        p = torch.arange(1, L + 1, dtype=dtype, device=device)  # (L,)

        # Frequencies and phases (ensure on same device/dtype)
        w = torch.exp(self._log_w).to(device=device, dtype=dtype)  # (K,)
        phi = self._phase.to(device=device, dtype=dtype)  # (K,)

        # Features Φ(p): (L, 2K)
        angles = p[:, None] * w[None, :] + phi[None, :]  # (L, K)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (L, 2K)

        # Scalar gate per position: g(p) = 1 + s * Φ(p) · v
        v = self._proj.to(device=device, dtype=dtype)  # (2K,)
        gate = 1.0 + self._scale * (feats @ v)  # (L,)

        out = reps * gate[:, None]
        return protein.set_processed(processed=out)
