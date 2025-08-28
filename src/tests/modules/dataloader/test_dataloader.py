#!/usr/bin/env python3
"""Dataloader の分割・基本APIのテスト"""

import torch

from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinProps


def _make_protein(idx: int, L: int, D: int) -> Protein:
    reps = torch.full((L, D), float(idx), dtype=torch.float32)
    props: ProteinProps = {"seq": "A" * L, "feat1": float(idx), "label1": float(100 + idx)}
    return Protein(key=f"p{idx}", props=props, representations=reps)


class TestDataloader:
    def test_batch_splitting_and_types(self):
        N, L, D = 10, 3, 4
        proteins = [_make_protein(idx=i, L=L, D=D) for i in range(1, N + 1)]
        plist = ProteinList(proteins)

        config = DataloaderConfig(
            protein_list=plist,
            input_props=["feat1"],
            output_props=["label1"],
            batch_size=4,  # -> [4, 4, 2]
            cacheable=True,
            process_list=DataProcessList([Initializer(), Aggregator("mean")]),
        )

        loader = Dataloader(config=config)

        # バッチ数と各バッチのサイズ
        assert len(loader) == 3
        sizes = [len(b) for b in loader]
        assert sizes == [4, 4, 2]

        # インデックス取得とスライスの型
        first = loader[0]
        assert hasattr(first, "use") and callable(first.use)

        tail = loader[1:]
        assert isinstance(tail, Dataloader)
        assert len(tail) == 2

        # 各バッチで use() が動作し、形状が正しいこと
        for bsz, batch in zip(sizes, loader):
            x, y, back = batch.use()
            assert x.shape == (bsz, D + 1)  # processed(D) + input_props(1)
            assert y.shape == (bsz, 1)
            assert isinstance(back, ProteinList)
