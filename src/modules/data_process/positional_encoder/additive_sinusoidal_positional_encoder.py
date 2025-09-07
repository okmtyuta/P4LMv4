"""
サイン波に基づく加法的な位置エンコード（通常/逆順/双方向）。

- 位置テンソルの生成は `base_sinusoidal__positional_encoder._BaseSinusoidalPositionalEncoder` を利用します。
- 入力との結合は加算（additive）。
"""

import torch

from src.modules.data_process.positional_encoder.base_sinusoidal__positional_encoder import (
    _BaseSinusoidalPositionalEncoder,
)
from src.modules.protein.protein import Protein


class AdditiveSinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    """通常順序の加法的サイン波位置エンコーダ。"""

    def __init__(self, a: float, b: float, gamma: float) -> None:
        """加法結合を用いるエンコーダとして初期化する。"""
        super().__init__(a=a, b=b, gamma=gamma)

    def _act(self, protein: Protein) -> Protein:
        """処理済テンソルへ位置テンソルを加算する。"""
        reps = protein.get_processed()
        L, D = reps.shape
        pos = self._positional_tensor(length=L, dim=D, reversed=False)
        out = reps + pos
        return protein.set_processed(processed=out)


class ReversedAdditiveSinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    """逆順の加法的サイン波位置エンコーダ。"""

    def __init__(self, a: float, b: float, gamma: float) -> None:
        """加法結合を用いるエンコーダとして初期化する。"""
        super().__init__(a=a, b=b, gamma=gamma)

    def _act(self, protein: Protein) -> Protein:
        """処理済テンソルへ逆順位置テンソルを加算する。"""
        reps = protein.get_processed()
        L, D = reps.shape
        pos = self._positional_tensor(length=L, dim=D, reversed=True)
        out = reps + pos
        return protein.set_processed(processed=out)


class BidirectionalAdditiveSinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    """通常/逆順を連結して 2D に拡張する加法版。"""

    @property
    def dim_factor(self) -> int:
        """出力次元は 2 倍。"""
        return 2

    def __init__(self, a: float, b: float, gamma: float) -> None:
        """加法結合を用いるエンコーダとして初期化する。"""
        super().__init__(a=a, b=b, gamma=gamma)

    def _act(self, protein: Protein) -> Protein:
        """通常/逆順の位置テンソルをそれぞれ加算し、チャネル方向に連結する。"""
        reps = protein.get_processed()
        L, D = reps.shape
        pos_normal = self._positional_tensor(length=L, dim=D, reversed=False)
        pos_reversed = self._positional_tensor(length=L, dim=D, reversed=True)

        reps_cat = torch.cat([reps, reps], dim=1)
        pos_cat = torch.cat([pos_normal, pos_reversed], dim=1)
        out = reps_cat + pos_cat
        return protein.set_processed(processed=out)
