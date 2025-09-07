#!/usr/bin/env python3
import torch

from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def test_call_replaces_proteins_and_preserves_list():
    reps = torch.randn(2, 3)
    p = Protein(key="k", props={"seq": "AA"}, representations=reps, processed=None)
    plist = ProteinList([p])

    dp = DataProcessList([Initializer()])
    out = dp(plist)
    assert out is plist
    assert torch.allclose(out[0].get_processed(), reps)
