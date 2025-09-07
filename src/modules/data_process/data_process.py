from abc import ABC, abstractmethod

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class DataProcess(ABC):
    """データ処理ステップの基底クラス。

    - `dim_factor`: 入力特徴次元 D に対して出力特徴次元が何倍になるかの係数。
      例) 双方向PEや3分割集約などは >1 を設定。既定は 1（不変）。
    - `map_dim(D)`: 入力次元 D を受け取り出力次元を返す。既定は `D * dim_factor`。
    """

    @property
    def dim_factor(self) -> int:
        """特徴次元の倍率を返す（既定は 1）。"""
        raise NotImplementedError

    def map_dim(self, dim: int) -> int:
        """`dim_factor` を用いて出力次元を見積もる。"""
        return int(dim) * int(self.dim_factor)

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        """各 Protein に対して処理を適用し、新しいリストで置換する。"""
        proteins = [self._act(protein=protein) for protein in protein_list]
        return protein_list.replace(iterable=proteins)

    @abstractmethod
    def _act(self, protein: Protein) -> Protein:
        """サブクラスが 1 件の Protein に対する処理を実装する。"""
        raise NotImplementedError
