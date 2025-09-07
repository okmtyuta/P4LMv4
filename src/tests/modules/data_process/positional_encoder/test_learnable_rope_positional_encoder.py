#!/usr/bin/env python3
import torch

from src.modules.data_process.positional_encoder.learnable_rope_positional_encoder import (
    BidirectionalLearnableRoPEPositionalEncoder,
    LearnableRoPEPositionalEncoder,
    ReversedLearnableRoPEPositionalEncoder,
)
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _plist(L: int, D: int) -> ProteinList:
    torch.manual_seed(1)
    reps = torch.randn(L, D)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    return ProteinList([p])


def test_learnable_rope_shapes():
    plist = _plist(6, 4)
    rope = LearnableRoPEPositionalEncoder(dim=4, theta_base=10000)
    rrope = ReversedLearnableRoPEPositionalEncoder(dim=4, theta_base=10000)
    brope = BidirectionalLearnableRoPEPositionalEncoder(dim=4, theta_base=10000)

    a = rope(plist)[0].get_processed()
    assert a.shape == (6, 4)
    plist[0].set_processed(plist[0].get_representations())
    b = rrope(plist)[0].get_processed()
    assert b.shape == (6, 4)
    plist[0].set_processed(plist[0].get_representations())
    c = brope(plist)[0].get_processed()
    assert c.shape == (6, 8)
