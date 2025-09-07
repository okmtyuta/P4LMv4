"""
ESM 系言語モデルの共通ラッパ。
"""

from src.modules.extract.language._language import _Language
from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.extract.language.esm.esm_types import ESMModelName
from src.modules.protein.protein_list import ProteinList


class _ESMLanguage(_Language):
    """ESM 変換器で `ProteinList` に表現を設定する言語モデル。"""

    def __init__(self, model_name: ESMModelName):
        """指定モデル名の ESM 変換器を構築する。"""
        super().__init__()
        self._converter = ESMConverter(model_name=model_name)

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        """与えられた `ProteinList` に対し、ESM 表現を設定して返す。"""
        self._set_representations(protein_list=protein_list)
        return protein_list

    def _set_representations(self, protein_list: ProteinList) -> ProteinList:
        """各 Protein の `seq` から (L, D) 表現を得て `representations` に格納。"""
        # 空のリストの場合は何もしない
        if len(protein_list) == 0:
            return protein_list

        # ProteinListから配列を取得（propsのseqから）
        seqs = [protein.seq for protein in protein_list]

        sequence_representations = self._converter(seqs=seqs)

        # 各Proteinに特徴量を設定
        for i, protein in enumerate(protein_list):
            representations = sequence_representations[i]
            protein.set_representations(representations=representations)

        return protein_list
