#!/usr/bin/env python3
"""Multiplicative sinusoidal positional encoder の動作テスト"""

import torch

from src.modules.data_process.positional_encoder.multiplicative_sinusoidal_positional_encoder import (
    BidirectionalMultiplicativeSinusoidalPositionalEncoder,
    MultiplicativeSinusoidalPositionalEncoder,
    ReversedMultiplicativeSinusoidalPositionalEncoder,
)
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _expected_pos(a: float, b: float, gamma: float, L: int, D: int, reversed: bool) -> torch.Tensor:
    positions = torch.arange(L, 0, -1, dtype=torch.float32) if reversed else torch.arange(1, L + 1, dtype=torch.float32)
    p_col = positions[:, None]
    i = torch.arange(1, D + 1, dtype=torch.float32)
    even_mask = i.remainder(2) == 0
    den_even = a ** ((i - 1.0) / float(D))
    den_odd = a ** ((i - 2.0) / float(D))
    even_vals = torch.sin(p_col / den_even) ** b + gamma
    odd_vals = torch.sin(p_col / den_odd) ** b + gamma
    return torch.where(even_mask[None, :], even_vals, odd_vals)


class TestMultiplicativeSinusoidalPositionalEncoder:
    def test_normal_and_reversed_values(self):
        a, b, gamma = 2.0, 1.0, 0.0
        L, D = 4, 6
        reps = torch.ones((L, D), dtype=torch.float32)
        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps)
        p.set_processed(reps)
        plist = ProteinList([p])

        enc = MultiplicativeSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out = enc(plist)[0].get_processed()
        expected = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=False)
        assert torch.allclose(out, reps * expected, atol=1e-6)

        p_r = Protein(key="p1r", props={"seq": "AAAA"}, representations=reps)
        p_r.set_processed(reps)
        plist_r = ProteinList([p_r])
        enc_r = ReversedMultiplicativeSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out_r = enc_r(plist_r)[0].get_processed()
        expected_r = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=True)
        assert torch.allclose(out_r, reps * expected_r, atol=1e-6)

    def test_bidirectional_concat(self):
        a, b, gamma = 2.0, 1.0, 0.0
        L, D = 3, 4
        reps = torch.ones((L, D), dtype=torch.float32)
        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps)
        p.set_processed(reps)
        plist = ProteinList([p])
        enc_bi = BidirectionalMultiplicativeSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out = enc_bi(plist)[0].get_processed()
        assert out.shape == (L, 2 * D)
        expected_norm = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=False)
        expected_rev = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=True)
        assert torch.allclose(out[:, :D], reps * expected_norm, atol=1e-6)
        assert torch.allclose(out[:, D:], reps * expected_rev, atol=1e-6)
