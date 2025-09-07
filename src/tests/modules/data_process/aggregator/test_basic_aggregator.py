#!/usr/bin/env python3
"""
Aggregator の基本機能テスト（平均）。
"""

import torch

from src.modules.data_process.aggregator import Aggregator
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class TestBasicBehavior:
    """基本動作のテスト"""

    def test_mean_single_protein(self):
        reps = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 4.0]])
        protein = Protein(key="k1", props={"seq": "AAA"}, representations=reps)
        protein.set_processed(reps)
        plist = ProteinList([protein])

        agg = Aggregator("mean")
        out = agg(plist)

        expected = torch.tensor([3.0, 4.0])
        assert torch.allclose(out[0].get_processed(), expected)

    def test_mean_multiple_proteins(self):
        reps1 = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
        reps2 = torch.tensor([[2.0, 0.0], [4.0, 4.0], [6.0, 2.0]])
        p1 = Protein(key="p1", props={"seq": "AAA"}, representations=reps1)
        p1.set_processed(reps1)
        p2 = Protein(key="p2", props={"seq": "BBB"}, representations=reps2)
        p2.set_processed(reps2)
        plist = ProteinList([p1, p2])

        agg = Aggregator("mean")
        out = agg(plist)

        assert torch.allclose(out[0].get_processed(), torch.tensor([2.0, 4.0]))
        assert torch.allclose(out[1].get_processed(), torch.tensor([4.0, 2.0]))
