"""
前処理パイプラインの先頭で、生の表現を `processed` に設定する初期化器。
"""

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class Initializer(DataProcess):
    """`representations` を `processed` にコピーして以降の処理に備える。"""

    @property
    def dim_factor(self) -> int:
        """出力次元は D（1倍）。"""
        return 1

    def _act(self, protein: Protein):
        """生の表現をそのまま `processed` に設定する。"""
        reps = protein.get_representations()
        return protein.set_processed(processed=reps)
