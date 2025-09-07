"""
複数の DataProcess を順に適用するパイプラインコンテナ。
"""

from src.modules.container.sequence_container import SequenceContainer
from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein_list import ProteinList


class DataProcessList(SequenceContainer[DataProcess]):
    """DataProcess の列を保持し、順方向へ適用する。"""

    def __init__(self, iterable: list[DataProcess]):
        """処理列で初期化する。"""
        super().__init__(iterable=iterable)

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        """各プロセスを順に適用して ProteinList を更新する。"""
        for process in self._data:
            protein_list = process(protein_list=protein_list)
        return protein_list

    def output_dim(self, input_dim: int) -> int:
        """入力特徴次元から、このパイプライン通過後の出力次元を推定する。

        各 DataProcess の `map_dim` を順に適用するだけの単純な推定。
        集約器などで長さ次元 L を畳み込んでも、本メソッドは特徴次元 D のみを追跡する。
        """
        dim = int(input_dim)
        for proc in self._data:
            dim = proc.map_dim(dim)
        return dim
