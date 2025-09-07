import random
import secrets
from typing import Iterable, Optional, Self

import polars as pl

from src.modules.container.serializable_container_list import SerializableContainerList
from src.modules.protein.protein import Protein
from src.modules.protein.protein_types import ProteinProps


class ProteinList(SerializableContainerList[Protein]):
    """Protein を格納するシーケンスコンテナ（HDF5/辞書I/O対応）。"""

    def __init__(self, iterable: Iterable[Protein]) -> None:
        """Protein の列で初期化する。"""
        super().__init__(iterable)

    @classmethod
    def from_csv(cls, path: str) -> "ProteinList":
        """CSV から ProteinList を構築する（カラム 'index' を key に使用）。"""
        df = pl.read_csv(path)

        proteins = []
        for row in df.iter_rows(named=True):
            # index を key に採用
            key = str(row["index"])

            # 'seq' を含むすべてのカラムを props へ（index は除外）
            props: ProteinProps = {k: v for k, v in row.items() if k != "index"}

            # 直接 Protein を生成
            protein = Protein(key=key, props=props, representations=None, processed=None, predicted={})
            proteins.append(protein)

        return cls(proteins)

    def shuffle(self, seed: Optional[int] = None) -> Self:
        """要素順を破壊的にシャッフルする。

        - インプレースで順序を変更します。
        - `seed` が `None` の場合は内部でランダムに決定します（`secrets.randbits(64)`）。

        Args:
            seed: 乱数シード。`None` 可（未指定時は内部で自動決定）。

        Returns:
            self（メソッドチェーン可能）。
        """
        if seed is None:
            seed = secrets.randbits(64)
        rng = random.Random(seed)
        rng.shuffle(self._data)

        return self
