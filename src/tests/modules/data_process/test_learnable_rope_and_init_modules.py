#!/usr/bin/env python3
import torch

from src.modules.data_process.aggregator.attention_pooling_aggregator import AttentionPoolingAggregator
from src.modules.data_process.aggregator.ends_segment_mean_aggregator import EndsSegmentMeanAggregator
from src.modules.data_process.aggregator.logsumexp_aggregator import LogSumExpAggregator
from src.modules.data_process.aggregator.weighted_mean_aggregator import WeightedMeanAggregator
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


def test_weighted_mean_aggregator_constant_invariance():
    L, D = 5, 3
    x = torch.randn(D)
    reps = x.repeat(L, 1)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    plist = ProteinList([p])
    agg = WeightedMeanAggregator(dim=D, max_length=8)
    out = agg(plist)[0].get_processed()
    assert torch.allclose(out, x, atol=1e-5)


def test_ends_segment_mean_shapes_and_values():
    L, D = 5, 2
    reps = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    plist = ProteinList([p])
    esm = EndsSegmentMeanAggregator(head_len=2, tail_len=2)
    out = esm(plist)[0].get_processed()
    assert out.shape == (3 * D,)
    # 手計算: head(0,1), center(2), tail(3,4)
    head = reps[:2].mean(0)
    center = reps[2:3].mean(0)
    tail = reps[3:].mean(0)
    expected = torch.cat([head, center, tail], dim=0)
    assert torch.allclose(out, expected)


def test_logsumexp_aggregator_length_invariance_on_constant():
    L, D = 7, 4
    x = torch.randn(D)
    reps = x.repeat(L, 1)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    plist = ProteinList([p])
    ls = LogSumExpAggregator(tau=3.0)
    out = ls(plist)[0].get_processed()
    assert torch.allclose(out, x, atol=1e-5)


def test_attention_pooling_aggregator_shapes_on_constant():
    L, D, K = 6, 4, 3
    x = torch.randn(D)
    reps = x.repeat(L, 1)
    p = Protein(key="k", props={"seq": "A" * L}, representations=reps, processed=reps.clone())
    plist = ProteinList([p])
    ap = AttentionPoolingAggregator(dim=D, num_queries=K, temperature=10.0)
    out = ap(plist)[0].get_processed()
    assert out.shape == (K * D,)
