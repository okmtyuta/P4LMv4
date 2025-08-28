#!/usr/bin/env python3
"""sinusoidal_positional_encoder の動作テスト"""

import torch

from src.modules.data_process.positional_encoder.sinusoidal_positional_encoder import (
    BidirectionalSinusoidalPositionalEncoder,
    ReversedSinusoidalPositionalEncoder,
    SinusoidalPositionalEncoder,
)
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _expected_pos(a: float, b: float, gamma: float, L: int, D: int, reversed: bool) -> torch.Tensor:
    # 位置ベクトル
    positions = torch.arange(L, 0, -1, dtype=torch.float32) if reversed else torch.arange(1, L + 1, dtype=torch.float32)
    p_col = positions[:, None]  # (L, 1)

    # 次元インデックス（1 始まり）
    i = torch.arange(1, D + 1, dtype=torch.float32)
    even_mask = i.remainder(2) == 0

    den_even = a ** ((i - 1.0) / float(D))
    den_odd = a ** ((i - 2.0) / float(D))

    even_vals = torch.sin(p_col / den_even) ** b + gamma
    odd_vals = torch.sin(p_col / den_odd) ** b + gamma
    return torch.where(even_mask[None, :], even_vals, odd_vals)


class TestSinusoidalPositionalEncoder:
    def test_normal_and_reversed_values(self):
        a, b, gamma = 2.0, 1.0, 0.0
        L, D = 4, 6
        reps = torch.ones((L, D), dtype=torch.float32)

        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps)
        p.set_processed(reps)
        plist = ProteinList([p])

        # normal
        enc = SinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out_list = enc(plist)
        out = out_list[0].get_processed()
        expected = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=False)
        assert torch.allclose(out, expected, atol=1e-6)

        # reversed（元の表現に対して検証するため、新しい ProteinList を用意）
        p_rev = Protein(key="p1r", props={"seq": "AAAA"}, representations=reps)
        p_rev.set_processed(reps)
        plist_rev = ProteinList([p_rev])
        enc_rev = ReversedSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out_list_rev = enc_rev(plist_rev)
        out_rev = out_list_rev[0].get_processed()
        expected_rev = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=True)
        assert torch.allclose(out_rev, expected_rev, atol=1e-6)

    def test_bidirectional_shape_and_parts(self):
        a, b, gamma = 2.0, 1.0, 0.0
        L, D = 3, 4
        reps = torch.ones((L, D), dtype=torch.float32)

        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps)
        p.set_processed(reps)
        plist = ProteinList([p])

        enc_bi = BidirectionalSinusoidalPositionalEncoder(a=a, b=b, gamma=gamma)
        out_list = enc_bi(plist)
        out = out_list[0].get_processed()

        assert out.shape == (L, 2 * D)

        expected_norm = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=False)
        expected_rev = _expected_pos(a=a, b=b, gamma=gamma, L=L, D=D, reversed=True)

        # 左半分が通常、右半分が逆順
        assert torch.allclose(out[:, :D], expected_norm, atol=1e-6)
        assert torch.allclose(out[:, D:], expected_rev, atol=1e-6)
