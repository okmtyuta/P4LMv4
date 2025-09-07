"""
絶対位置ごとに学習可能な加算/スケールを適用するエンコーダ。
"""

from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class LearnableAbsolutePositionalAdder(DataProcess):
    """位置 p ごとに学習スカラのバイアス b[p] を加える。"""

    def __init__(self, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._max_len = int(max_length)
        self._bias = nn.Parameter(torch.zeros(self._max_len, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習対象のバイアスベクトルを返す。"""
        return [self._bias]

    @property
    def dim_factor(self) -> int:
        """出力次元は D（=1倍）。"""
        return 1

    def _act(self, protein: Protein) -> Protein:
        """各位置にバイアスを加算する。"""
        reps = protein.get_processed()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        idx = torch.arange(L, device=device)
        idx = torch.clamp(idx, max=self._max_len - 1)

        b = self._bias.to(device=device, dtype=dtype)  # (max_len,)
        pos_bias = b[idx]  # (L,)

        out = reps + pos_bias[:, None]
        return protein.set_processed(processed=out)


class LearnableAbsolutePositionalScaler(DataProcess):
    """位置 p ごとにゲート g[p]=1+s[p] を掛ける（s[p] は学習スカラ）。"""

    def __init__(self, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._max_len = int(max_length)
        self._scale = nn.Parameter(torch.zeros(self._max_len, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        """学習対象のスケールベクトルを返す。"""
        return [self._scale]

    @property
    def dim_factor(self) -> int:
        """出力次元は D（=1倍）。"""
        return 1

    def _act(self, protein: Protein) -> Protein:
        """各位置にスケールを乗算する。"""
        reps = protein.get_processed()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        idx = torch.arange(L, device=device)
        idx = torch.clamp(idx, max=self._max_len - 1)

        s = self._scale.to(device=device, dtype=dtype)  # (max_len,)
        gate = 1.0 + s[idx]  # (L,)

        out = reps * gate[:, None]
        return protein.set_processed(processed=out)
