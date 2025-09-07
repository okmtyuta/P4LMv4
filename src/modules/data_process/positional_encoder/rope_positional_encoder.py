"""
RoPE（Rotary Positional Embedding）による位置エンコード。

- 偶数ペアごとに2次元回転を適用し、奇数次元末尾はそのまま保持します。
- 位置は 1..L（反転時は L..1）。
"""

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class RoPEAnglesCache:
    """(length, dim, reversed) をキーに (cos, sin) を保持する簡易キャッシュ。"""

    def __init__(self) -> None:
        """空キャッシュで初期化する。"""
        self._cache: dict[tuple[int, int, bool], tuple[torch.Tensor, torch.Tensor]] = {}

    def read(self, length: int, dim: int, reversed: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        """キャッシュ読み出し（未登録なら None）。"""
        return self._cache.get((length, dim, reversed))

    def set(
        self,
        length: int,
        dim: int,
        reversed: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        """キャッシュへ (cos, sin) を登録。"""
        self._cache[(length, dim, reversed)] = (cos, sin)


class _BaseRoPEPositionalEncoder(DataProcess):
    """(L, D) 表現へ RoPE を適用する基底クラス。"""

    def __init__(self, theta_base: float) -> None:
        """RoPE の基底周波数を設定する。"""
        self._theta_base = float(theta_base)
        self._cache = RoPEAnglesCache()

    @property
    def dim_factor(self) -> int:
        """出力次元は D（=1倍）。"""
        return 1

    def _positions(self, length: int, reversed: bool) -> torch.Tensor:
        """1..L（反転時は L..1）の位置ベクトルを返す。"""
        if reversed:
            return torch.arange(length, 0, -1, dtype=torch.float32)
        return torch.arange(1, length + 1, dtype=torch.float32)

    def _cos_sin(self, length: int, dim: int, reversed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """RoPE 用の cos/sin 行列を生成（キャッシュ利用）。"""
        cached = self._cache.read(length=length, dim=dim, reversed=reversed)
        if cached is not None:
            return cached

        L, D = length, dim
        half = D // 2
        if half == 0:
            # D == 0 or 1 の場合にも対応
            cos = torch.ones((L, 0), dtype=torch.float32)
            sin = torch.zeros((L, 0), dtype=torch.float32)
            self._cache.set(length, dim, reversed, cos, sin)
            return cos, sin

        # 位置 (1..L) または (L..1)
        positions = self._positions(length=L, reversed=reversed)  # (L,)
        p_col = positions[:, None]  # (L, 1)

        # 周波数（ペア単位）。inv_freq = base^{-2m/D}
        m = torch.arange(0, half, dtype=torch.float32)  # (half,)
        inv_freq = self._theta_base ** (-2.0 * m / float(D))  # (half,)

        angles = p_col * inv_freq[None, :]  # (L, half)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self._cache.set(length, dim, reversed, cos, sin)
        return cos, sin

    def _apply_rope(self, reps: torch.Tensor, reversed: bool) -> torch.Tensor:
        """RoPE による回転を適用して同形状のテンソルを返す。"""
        L, D = reps.shape
        cos, sin = self._cos_sin(length=L, dim=D, reversed=reversed)  # (L, half)
        half = D // 2
        if half == 0:
            return reps

        x_even = reps[:, 0 : 2 * half : 2]
        x_odd = reps[:, 1 : 2 * half : 2]

        # 回転: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = reps.clone()
        out[:, 0 : 2 * half : 2] = rot_even
        out[:, 1 : 2 * half : 2] = rot_odd
        # 奇数次元の最終チャネルはそのまま保持
        return out


class RoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    """通常順序の RoPE を適用する。"""

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()
        out = self._apply_rope(reps, False)
        return protein.set_processed(processed=out)


class ReversedRoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    """逆順序の RoPE を適用する。"""

    def _act(self, protein: Protein) -> Protein:
        """処理済テンソルへ逆順 RoPE 回転を適用する。"""
        reps = protein.get_processed()
        out = self._apply_rope(reps, True)
        return protein.set_processed(processed=out)


class BidirectionalRoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    """通常/逆順の2系列を連結して 2D に拡張する。"""

    @property
    def dim_factor(self) -> int:  # 2D へ拡張
        """出力次元は 2 倍。"""
        return 2

    def _act(self, protein: Protein) -> Protein:
        """通常/逆順の両 RoPE を適用し、チャネル方向に連結する。"""
        reps = protein.get_representations()
        out_normal = self._apply_rope(reps, False)
        out_reversed = self._apply_rope(reps, True)
        out = torch.cat([out_normal, out_reversed], dim=1)
        return protein.set_processed(processed=out)
