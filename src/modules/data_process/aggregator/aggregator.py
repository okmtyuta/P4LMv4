from typing import Literal

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein

AggregateMethod = Literal["mean"]


class Aggregator(DataProcess):
    def __init__(self, method: AggregateMethod) -> None:
        self._method: AggregateMethod = method

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
        if self._method == "mean":
            processed = protein.get_processed()
            mean = self._mean(input=processed)
            return protein.set_processed(processed=mean)

        raise ValueError(f"Unknown aggregate method: {self._method}")

    def _mean(self, input: torch.Tensor) -> torch.Tensor:
        return torch.mean(input=input, dim=0)
