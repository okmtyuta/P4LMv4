#!/usr/bin/env python3
"""Initializer の動作テスト。"""

import torch

from src.modules.data_process.initializer import Initializer
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def test_initializer_copies_representations_to_processed():
    reps = torch.randn(4, 3)
    p = Protein(key="k", props={"seq": "AAAA"}, representations=reps.clone())
    plist = ProteinList([p])

    init = Initializer()
    out = init(plist)
    assert torch.allclose(out[0].get_processed(), reps)
