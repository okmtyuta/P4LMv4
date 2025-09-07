#!/usr/bin/env python3
from __future__ import annotations

import torch

from src.modules.data_process.positional_encoder import (
    FloaterPositionalEncoder,
    LearnableAbsolutePositionalAdder,
    LearnableAbsolutePositionalScaler,
    LearnableFourierPositionalEncoder,
)
from src.modules.data_process.positional_encoder.floater_positional_encoder import SimpleGatedSineDynamics
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _plist(L: int, D: int) -> ProteinList:
    torch.manual_seed(0)
    reps = torch.randn(L, D)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    return ProteinList([p])


def test_learnable_absolute_add_and_scale_identity():
    plist = _plist(6, 4)
    add = LearnableAbsolutePositionalAdder(max_length=10)
    scale = LearnableAbsolutePositionalScaler(max_length=10)

    before = plist[0].get_processed().clone()
    out = add(plist)[0].get_processed()
    assert torch.allclose(before, out)

    # scale はゲートが 1 で初期化 → 恒等写像
    plist[0].set_processed(before)
    out2 = scale(plist)[0].get_processed()
    assert torch.allclose(before, out2)


def test_learnable_fourier_encoder_shape():
    plist = _plist(5, 3)
    enc = LearnableFourierPositionalEncoder(num_bases=4, min_period=2.0, max_period=16.0, projection_scale=0.1)
    out = enc(plist)[0].get_processed()
    assert out.shape == (5, 3)


def test_floater_positional_encoder_smoke():
    torch.manual_seed(0)
    L, D = 5, 4
    plist = _plist(L, D)
    dyn = SimpleGatedSineDynamics(dim=D, hidden=8, num_freqs=2, omega_min=0.5, omega_max=2.0, damping=0.1)
    floater = FloaterPositionalEncoder(
        dim=D, dynamics=dyn, delta_t=0.2, method="rk4", rtol=1e-3, atol=1e-5, use_adjoint=False
    )
    before = plist[0].get_processed().clone()
    out = floater(plist)[0].get_processed()
    assert out.shape == before.shape
    # 位置エンコードの影響で一般には変更されるはず
    assert not torch.allclose(before, out)
