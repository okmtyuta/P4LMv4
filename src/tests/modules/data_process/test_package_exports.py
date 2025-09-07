#!/usr/bin/env python3
from src.modules.data_process.aggregator import (
    Aggregator,
    AttentionPoolingAggregator,
    EndsSegmentMeanAggregator,
    LogSumExpAggregator,
    WeightedMeanAggregator,
)
from src.modules.data_process.positional_encoder import (
    BidirectionalLearnableRoPEPositionalEncoder,
    BidirectionalRoPEPositionalEncoder,
    BidirectionalSinusoidalPositionalEncoder,
    FloaterPositionalEncoder,
    LearnableAbsolutePositionalAdder,
    LearnableAbsolutePositionalScaler,
    LearnableFourierPositionalEncoder,
    LearnableRoPEPositionalEncoder,
    ReversedLearnableRoPEPositionalEncoder,
    ReversedRoPEPositionalEncoder,
    ReversedSinusoidalPositionalEncoder,
    RoPEPositionalEncoder,
    SinusoidalPositionalEncoder,
)


def test_imports_exist():
    # 何もしない。インポートできれば OK（__init__ のエクスポート確認）。
    assert Aggregator is not None
    assert WeightedMeanAggregator is not None
    assert EndsSegmentMeanAggregator is not None
    assert LogSumExpAggregator is not None
    assert AttentionPoolingAggregator is not None

    assert FloaterPositionalEncoder is not None
    assert LearnableAbsolutePositionalAdder is not None
    assert LearnableAbsolutePositionalScaler is not None
    assert LearnableFourierPositionalEncoder is not None
    assert LearnableRoPEPositionalEncoder is not None
    assert ReversedLearnableRoPEPositionalEncoder is not None
    assert BidirectionalLearnableRoPEPositionalEncoder is not None
    assert RoPEPositionalEncoder is not None
    assert ReversedRoPEPositionalEncoder is not None
    assert BidirectionalRoPEPositionalEncoder is not None
    assert SinusoidalPositionalEncoder is not None
    assert ReversedSinusoidalPositionalEncoder is not None
    assert BidirectionalSinusoidalPositionalEncoder is not None
