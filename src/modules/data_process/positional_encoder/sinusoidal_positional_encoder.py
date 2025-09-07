"""
サイン波に基づく簡易位置エンコード（双方向/反転も提供）。
"""

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class SinusoidalPositionalEncoderCache:
    """(length, dim, reversed) をキーとしたテンソルキャッシュ。"""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, bool], torch.Tensor] = {}

    def read(self, length: int, dim: int, reversed: bool) -> torch.Tensor | None:
        return self._cache.get((length, dim, reversed))

    def set(self, length: int, dim: int, reversed: bool, value: torch.Tensor) -> None:
        self._cache[(length, dim, reversed)] = value


class _BaseSinusoidalPositionalEncoder(DataProcess):
    """サイン波位置エンコードの共通実装。"""

    def __init__(self, a: float, b: float, gamma: float) -> None:
        self._a = a
        self._b = b
        self._gamma = gamma
        self._cache = SinusoidalPositionalEncoderCache()

    @property
    def dim_factor(self) -> int:
        return 1

    # ベクトル化された位置エンコード行列の生成
    def _positional_tensor(self, length: int, dim: int, reversed: bool = False) -> torch.Tensor:
        cached = self._cache.read(length=length, dim=dim, reversed=reversed)
        if cached is not None:
            return cached

        # 位置 [1..length]（逆順が指定された場合は length..1）
        positions = self._positions(length=length, reversed=reversed)
        p_col = positions[:, None]  # (L, 1)

        # 次元インデックス [1..dim]
        i = torch.arange(1, dim + 1, dtype=torch.float32)  # (D,)
        even_mask = i.remainder(2) == 0  # True where i is even

        # 分母の事前計算（iは1始まり）
        den_even = self._a ** ((i - 1.0) / float(dim))  # (D,)
        den_odd = self._a ** ((i - 2.0) / float(dim))  # (D,)

        # ブロードキャスト計算 (L, D)
        even_vals = torch.sin(p_col / den_even) ** self._b + self._gamma
        odd_vals = torch.sin(p_col / den_odd) ** self._b + self._gamma
        vals = torch.where(even_mask[None, :], even_vals, odd_vals)

        self._cache.set(length=length, dim=dim, reversed=reversed, value=vals)
        return vals

    def _positions(self, length: int, reversed: bool) -> torch.Tensor:
        """長さ `length` の位置ベクトルを生成する。

        - `reversed` が True の場合: `[length, ..., 1]`
        - `reversed` が False の場合: `[1, ..., length]`
        """
        if reversed:
            return torch.arange(length, 0, -1, dtype=torch.float32)
        return torch.arange(1, length + 1, dtype=torch.float32)


class SinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()
        L, D = reps.shape
        pos = self._positional_tensor(length=L, dim=D, reversed=False)
        out = reps * pos
        return protein.set_processed(processed=out)


class ReversedSinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()
        L, D = reps.shape
        pos = self._positional_tensor(length=L, dim=D, reversed=True)
        out = reps * pos
        return protein.set_processed(processed=out)


class BidirectionalSinusoidalPositionalEncoder(_BaseSinusoidalPositionalEncoder):
    @property
    def dim_factor(self) -> int:  # 2D へ拡張
        return 2

    def __init__(self, a: float, b: float, gamma: float) -> None:
        super().__init__(a=a, b=b, gamma=gamma)
        self._normal = SinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        self._reversed = ReversedSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()
        L, D = reps.shape
        pos_normal = self._positional_tensor(length=L, dim=D, reversed=False)
        pos_reversed = self._positional_tensor(length=L, dim=D, reversed=True)

        # (L, 2D) に拡張
        reps_cat = torch.cat([reps, reps], dim=1)
        pos_cat = torch.cat([pos_normal, pos_reversed], dim=1)

        out = reps_cat * pos_cat

        return protein.set_processed(processed=out)
