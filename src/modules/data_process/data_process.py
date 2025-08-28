from abc import ABC, abstractmethod

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class DataProcess(ABC):
    def __call__(self, protein_list: ProteinList) -> ProteinList:
        proteins = [self._act(protein=protein) for protein in protein_list]
        return protein_list.replace(iterable=proteins)

    @abstractmethod
    def _act(self, protein: Protein) -> Protein:
        raise NotImplementedError
