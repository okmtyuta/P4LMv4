from typing import Optional, TypedDict

import torch

from src.modules.protein.protein_list import ProteinList


class DataloaderStateSource(TypedDict):
    protein_list: ProteinList
    batch_size: int
    cacheable: bool


UsableDataBatch = tuple[torch.Tensor, torch.Tensor, ProteinList]


class DataBatch:
    def __init__(self, protein_list: ProteinList, cacheable: bool = True):
        self._protein_list = protein_list
        self._cache: Optional[UsableDataBatch] = None
        self._cacheable = cacheable

    def use(self):
        if self._cacheable and self._cache is not None:
            return self._cache

        inputs = []
        labels = []

        for protein in self._protein_list:
            input = protein.processed
            inputs.append(input)

            label = [protein.read_props(key) for key in ["rt"]]
            labels.append(label)

        usable = (
            torch.stack(inputs).to(torch.float32),
            torch.Tensor(labels).to(torch.float32),
            self._protein_list,
        )

        if self._cacheable:
            self._cache = usable

        return usable


class Dataloader:
    def __init__(self, protein_list: ProteinList):
        self._protein_list = protein_list
        self._batches: Optional[list[DataBatch]] = None

    def __len__(self):
        return len(self._protein_list)

    @property
    def batches(self):
        return self._generate_batch()

    def _generate_batch(self):
        if self._batches is not None:
            return self._batches

        protein_lists = self._protein_list.split_equal(n=100)

        batches = [DataBatch(protein_list=protein_list) for protein_list in protein_lists]

        self._batches = batches
        return batches
