"""
言語モデル共通インタフェース。
"""

import abc

from src.modules.protein.protein_list import ProteinList


class _Language(metaclass=abc.ABCMeta):
    """ProteinList に表現を設定する抽象基底。"""

    name = "_language"

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        """実装クラスは `protein_list` を参照・更新して返す。"""
        raise NotImplementedError()
