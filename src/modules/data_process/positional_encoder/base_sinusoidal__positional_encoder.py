"""
サイン波位置エンコードの共通基底（位置テンソル生成）。

- 位置テンソルの計算のみを提供します。
- 入力との結合（加法/乗法）は各サブクラスの `_act` 内で行ってください。
"""

import torch

from src.modules.data_process.data_process import DataProcess


class SinusoidalPositionTensorCache:
    """(length, dim, reversed) をキーとした位置テンソルのキャッシュ。"""

    def __init__(self) -> None:
        """空キャッシュで初期化する。"""
        self._cache: dict[tuple[int, int, bool], torch.Tensor] = {}

    def read(self, length: int, dim: int, reversed: bool) -> torch.Tensor | None:
        """キーに一致する値を取得（未登録なら None）。"""
        return self._cache.get((length, dim, reversed))

    def set(self, length: int, dim: int, reversed: bool, value: torch.Tensor) -> None:
        """キャッシュへ値を登録。"""
        self._cache[(length, dim, reversed)] = value


class _BaseSinusoidalPositionalEncoder(DataProcess):
    """サイン波位置エンコードの共通実装（結合はサブクラスに委譲）。"""

    def __init__(self, a: float, b: float, gamma: float) -> None:
        """ハイパーパラメータを設定する。"""
        self._a = a
        self._b = b
        self._gamma = gamma
        self._cache = SinusoidalPositionTensorCache()

    @property
    def dim_factor(self) -> int:
        """出力次元は D（1倍）。"""
        return 1

    # ベクトル化された位置エンコード行列の生成
    def _positional_tensor(self, length: int, dim: int, reversed: bool = False) -> torch.Tensor:
        """位置エンコード行列を生成（キャッシュ利用）。"""
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

    # 入力との結合処理は各サブクラスの `_act` で実装する。
