import abc

from src.modules.protein.protein_list import ProteinList


class _Language(metaclass=abc.ABCMeta):
    name = "_language"

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        raise NotImplementedError()
