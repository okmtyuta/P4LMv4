from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from src.modules.data_process.data_process_list import DataProcessList
from src.modules.protein.protein_list import ProteinList


@dataclass
class DataloaderConfig:
    protein_list: ProteinList
    input_props: list[str]
    output_props: list[str]
    batch_size: int
    cacheable: bool
    process_list: DataProcessList

    def with_protein_list(self, protein_list: ProteinList) -> "DataloaderConfig":
        return DataloaderConfig(
            protein_list=protein_list,
            input_props=self.input_props,
            output_props=self.output_props,
            batch_size=self.batch_size,
            cacheable=self.cacheable,
            process_list=self.process_list,
        )


UsableDataBatch = tuple[torch.Tensor, torch.Tensor, ProteinList]


class DataBatch:
    def __init__(self, config: DataloaderConfig) -> None:
        self._config = config
        self._cache: Optional[UsableDataBatch] = None

    def __len__(self) -> int:
        return len(self._config.protein_list)

    def use(self) -> UsableDataBatch:
        if self._config.cacheable and self._cache is not None:
            return self._cache

        inputs = []
        labels = []

        process_list = self._config.process_list(protein_list=self._config.protein_list)

        for protein in process_list:
            processed = protein.get_processed()
            input_props = torch.Tensor([protein.read_props(key) for key in self._config.input_props])
            input_tensor = torch.cat([processed, input_props], dim=0)
            inputs.append(input_tensor)

            label = [protein.read_props(key) for key in self._config.output_props]
            labels.append(label)

        usable: UsableDataBatch = (
            torch.stack(inputs).to(torch.float32),
            torch.Tensor(labels).to(torch.float32),
            self._config.protein_list,
        )

        if self._config.cacheable:
            self._cache = usable

        return usable


class Dataloader:
    def __init__(self, config: DataloaderConfig) -> None:
        self._config = config
        self._batches: Optional[list[DataBatch]] = None

    @property
    def input_props(self):
        return self._config.input_props

    @property
    def output_props(self):
        return self._config.output_props

    @property
    def batches(self):
        if self._batches is None:
            self._batches = self._generate_batches()

        return self._batches

    def _generate_batches(self) -> list[DataBatch]:
        protein_lists: list[ProteinList] = self._config.protein_list.split_by_size(self._config.batch_size)
        batches: list[DataBatch] = []

        for protein_list in protein_lists:
            config = self._config.with_protein_list(protein_list=protein_list)
            batches.append(DataBatch(config=config))

        return batches

    def split_by_ratio(self, ratios: list[float]) -> list["Dataloader"]:
        """内部の ProteinList を比率で分割し、それぞれ独立の Dataloader を返す。"""
        if any(r <= 0 for r in ratios):
            raise ValueError("All ratios must be positive")

        parts: list[ProteinList] = self._config.protein_list.split_by_ratio(ratios)
        loaders: list[Dataloader] = []
        for plist in parts:
            cfg = self._config.with_protein_list(plist)
            loaders.append(Dataloader(cfg))
        return loaders
