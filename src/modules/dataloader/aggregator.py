from typing import Literal

import torch

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList

AggregateMethod = Literal["mean"]


class Aggregator:
    def __init__(self, method: AggregateMethod) -> None:
        self._method: AggregateMethod = method

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        proteins = [self._act(protein=protein) for protein in protein_list]
        return ProteinList(proteins=proteins)

    def _act(self, protein: Protein):
        if self._method == "mean":
            protein.processed = protein.representations

            if protein.processed is None:
                raise Exception

            mean = self._mean(input=protein.processed)
            return protein.set_processed(processed=mean)

        raise Exception

    def _mean(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(input=input, dim=0)
