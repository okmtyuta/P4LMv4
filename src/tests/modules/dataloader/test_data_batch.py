#!/usr/bin/env python3
"""DataBatch の基本動作とキャッシュ挙動のテスト"""

import torch

from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.dataloader.dataloader import DataBatch, DataloaderConfig
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinProps


def _make_protein(idx: int, L: int, D: int) -> Protein:
    reps = torch.full((L, D), float(idx), dtype=torch.float32)
    props: ProteinProps = {"seq": "A" * L, "feat1": float(idx), "label1": float(10 + idx)}
    return Protein(key=f"p{idx}", props=props, representations=reps)


class TestDataBatch:
    def test_use_and_shapes_with_cache(self):
        L, D = 4, 3
        proteins = [
            _make_protein(idx=1, L=L, D=D),
            _make_protein(idx=2, L=L, D=D),
        ]
        plist = ProteinList(proteins)

        config = DataloaderConfig(
            protein_list=plist,
            input_props=["feat1"],
            output_props=["label1"],
            batch_size=8,
            cacheable=True,
            process_list=DataProcessList([Initializer(), Aggregator("mean")]),
        )

        batch = DataBatch(config=config)
        x, y, back = batch.use()

        assert x.shape == (2, D + 1)
        assert y.shape == (2, 1)
        assert back is plist

        # x の先頭行: processed(=mean=1.0の3要素) + feat1(=1.0)
        assert torch.allclose(x[0], torch.tensor([1.0, 1.0, 1.0, 1.0]))
        # y の先頭行: label1(=11.0)
        assert torch.allclose(y[0], torch.tensor([11.0]))

        # キャッシュが有効なら同一オブジェクトが返る
        x2, y2, back2 = batch.use()
        assert x is x2 and y is y2 and back is back2

    def test_use_without_cache_returns_new_tuple(self):
        L, D = 3, 2
        proteins = [
            _make_protein(idx=3, L=L, D=D),
            _make_protein(idx=4, L=L, D=D),
        ]
        plist = ProteinList(proteins)

        config = DataloaderConfig(
            protein_list=plist,
            input_props=["feat1"],
            output_props=["label1"],
            batch_size=16,
            cacheable=False,
            process_list=DataProcessList([Initializer(), Aggregator("mean")]),
        )

        batch = DataBatch(config=config)
        out1 = batch.use()
        out2 = batch.use()

        assert out1 is not out2  # キャッシュなしなので都度新規
