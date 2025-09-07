#!/usr/bin/env python3
import torch

from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def test_data_process_list_applies_all_steps_in_order():
    reps = torch.randn(3, 2)
    p = Protein(key="k", props={"seq": "AAA"}, representations=reps, processed=None)
    plist = ProteinList([p])

    class AddOne(Initializer):
        # Initializer を継承して processed に reps を入れた後、さらに +1 するための簡易プロセス
        def _act(self, protein: Protein):  # type: ignore[override]
            out = super()._act(protein)  # set processed = representations
            x = out.get_processed() + 1.0
            return out.set_processed(x)

    dp = DataProcessList([Initializer(), AddOne()])
    out = dp(plist)
    assert torch.allclose(out[0].get_processed(), reps + 1.0)
