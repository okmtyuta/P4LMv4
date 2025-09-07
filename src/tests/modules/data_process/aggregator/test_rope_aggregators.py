#!/usr/bin/env python3
"""RoPE 整合型 Aggregator のテスト。"""

import torch

from src.modules.data_process.aggregator.rope_aligned_mean_aggregator import RoPEAlignedMeanAggregator
from src.modules.data_process.aggregator.rope_attention_pooling_aggregator import (
    RoPEAttentionPoolingAggregator,
)
from src.modules.data_process.positional_encoder.rope_positional_encoder import (
    ReversedRoPEPositionalEncoder,
    RoPEPositionalEncoder,
)
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _plist(L: int, D: int) -> ProteinList:
    torch.manual_seed(0)
    reps = torch.randn(L, D)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    return ProteinList([p])


def test_rope_aligned_mean_recovers_original_mean():
    theta = 10000.0
    L, D = 6, 4
    plist = _plist(L, D)
    # 元の平均
    original = plist[0].get_processed().mean(0)

    # RoPE を適用（通常）
    enc = RoPEPositionalEncoder(theta_base=theta)
    plist_enc = enc(plist)

    # 位相整合平均で元の平均に戻るはず
    agg = RoPEAlignedMeanAggregator(theta_base=theta, reversed=False)
    out = agg(plist_enc)[0].get_processed()
    assert torch.allclose(out, original, atol=1e-5)


def test_rope_aligned_mean_recovers_original_mean_reversed():
    theta = 10000.0
    L, D = 5, 6
    plist = _plist(L, D)
    original = plist[0].get_processed().mean(0)

    # RoPE を逆順で適用
    enc_r = ReversedRoPEPositionalEncoder(theta_base=theta)
    plist_enc = enc_r(plist)

    # 逆順設定で整合平均
    agg_r = RoPEAlignedMeanAggregator(theta_base=theta, reversed=True)
    out = agg_r(plist_enc)[0].get_processed()
    assert torch.allclose(out, original, atol=1e-5)


def test_rope_attention_pooling_shapes_and_zero_query_consistency():
    theta = 10000.0
    L, D, K = 7, 4, 3
    plist = _plist(L, D)

    # Q を 0 にしてスコアを 0 → 一様重み → 平均と一致
    agg = RoPEAttentionPoolingAggregator(dim=D, num_queries=K, theta_base=theta, reversed=False, temperature=1.0)
    with torch.no_grad():
        agg._Q.zero_()

    # 出力形状
    before = plist[0].get_processed().clone()
    out = agg(plist)[0].get_processed()
    assert out.shape == (K * D,)

    # 一様重みなら列平均の繰り返し
    mean_vec = torch.mean(before, dim=0)
    expected = torch.cat([mean_vec for _ in range(K)], dim=0)
    assert torch.allclose(out, expected, atol=1e-5)
