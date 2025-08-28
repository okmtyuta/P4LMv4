#!/usr/bin/env python3
"""RoPE positional encoder の動作テスト"""

import torch

from src.modules.data_process.positional_encoder.rope_positional_encoder import (
    BidirectionalRoPEPositionalEncoder,
    ReversedRoPEPositionalEncoder,
    RoPEPositionalEncoder,
)
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _expected_rope(reps: torch.Tensor, theta_base: float, reversed: bool) -> torch.Tensor:
    L, D = reps.shape
    half = D // 2
    if half == 0:
        return reps.clone()

    positions = torch.arange(L, 0, -1, dtype=torch.float32) if reversed else torch.arange(1, L + 1, dtype=torch.float32)
    p_col = positions[:, None]

    m = torch.arange(0, half, dtype=torch.float32)
    inv_freq = theta_base ** (-2.0 * m / float(D))
    angles = p_col * inv_freq[None, :]
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    out = reps.clone()
    x_even = reps[:, 0 : 2 * half : 2]
    x_odd = reps[:, 1 : 2 * half : 2]
    out[:, 0 : 2 * half : 2] = x_even * cos - x_odd * sin
    out[:, 1 : 2 * half : 2] = x_even * sin + x_odd * cos
    return out


class TestRoPE:
    def test_rope_normal_and_reversed(self):
        theta = 10000.0
        L, D = 5, 6
        reps = torch.randn((L, D), dtype=torch.float32)

        p = Protein(key="p1", props={"seq": "AAAAAA"}, representations=reps.clone())
        p.set_processed(reps.clone())
        plist = ProteinList([p])

        # normal
        enc = RoPEPositionalEncoder(theta_base=theta)
        out_list = enc(plist)
        got = out_list[0].get_processed()
        want = _expected_rope(reps, theta_base=theta, reversed=False)
        assert torch.allclose(got, want, atol=1e-6)

        # reversed（元の表現に対して検証するため、新しい ProteinList を用意）
        p_r = Protein(key="p1r", props={"seq": "AAAAAA"}, representations=reps.clone())
        p_r.set_processed(reps.clone())
        plist_r = ProteinList([p_r])
        enc_r = ReversedRoPEPositionalEncoder(theta_base=theta)
        out_list_r = enc_r(plist_r)
        got_r = out_list_r[0].get_processed()
        want_r = _expected_rope(reps, theta_base=theta, reversed=True)
        assert torch.allclose(got_r, want_r, atol=1e-6)

    def test_rope_odd_dimension_last_channel_kept(self):
        theta = 10000.0
        L, D = 4, 5  # 奇数次元
        reps = torch.randn((L, D), dtype=torch.float32)

        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps.clone())
        p.set_processed(reps.clone())
        plist = ProteinList([p])

        enc = RoPEPositionalEncoder(theta_base=theta)
        out_list = enc(plist)
        out = out_list[0].get_processed()

        # 最終チャネルは無変換
        assert torch.allclose(out[:, -1], reps[:, -1], atol=1e-6)

    def test_bidirectional_concat(self):
        theta = 10000.0
        L, D = 3, 4
        reps = torch.randn((L, D), dtype=torch.float32)

        p = Protein(key="p1", props={"seq": "AAAA"}, representations=reps.clone())
        p.set_processed(reps.clone())
        plist = ProteinList([p])

        enc_bi = BidirectionalRoPEPositionalEncoder(theta_base=theta)
        out_list = enc_bi(plist)
        out = out_list[0].get_processed()

        assert out.shape == (L, 2 * D)

        want_left = _expected_rope(reps, theta_base=theta, reversed=False)
        want_right = _expected_rope(reps, theta_base=theta, reversed=True)
        assert torch.allclose(out[:, :D], want_left, atol=1e-6)
        assert torch.allclose(out[:, D:], want_right, atol=1e-6)
