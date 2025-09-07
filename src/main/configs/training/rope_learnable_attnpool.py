#!/usr/bin/env python3
"""
Learnable RoPE + RoPE整合注意プーリングによる学習設定。

データ: outputs/plasma_lumos_1h/plasma_lumos_1h_data_esm2.h5
出力: outputs/experiments/plasma_lumos_1h/rope_learnable_attnpool/train_result.h5
"""

from __future__ import annotations

from schedulefree import RAdamScheduleFree

from src.main.training import TrainingRunnerConfig
from src.modules.data_process.aggregator import RoPEAttentionPoolingAggregator
from src.modules.data_process.aggregator.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.data_process.positional_encoder import LearnableRoPEPositionalEncoder
from src.modules.data_process.positional_encoder.multiplicative_sinusoidal_positional_encoder import (
    MultiplicativeSinusoidalPositionalEncoder,
)
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.helper.helper import Helper
from src.modules.model.basic import BasicModel
from src.modules.protein.protein_list import ProteinList

# 入出力パス
INPUT_H5 = Helper.ROOT / "outputs" / "plasma_lumos_1h" / "plasma_lumos_1h_data_esm2.h5"
OUTPUT_H5 = Helper.ROOT / "outputs" / "experiments" / "plasma_lumos_1h" / "rope_learnable_attnpool" / "train_result.h5"

# 代表的な ESM2 次元
REP_DIM = 1280

# ハイパーパラメータ
THETA_BASE = 10000.0
NUM_QUERIES = 4  # K（出力は K×D）
TEMPERATURE = 1.0
REVERSED = False

lrope = LearnableRoPEPositionalEncoder(dim=REP_DIM, theta_base=THETA_BASE)
ropa = RoPEAttentionPoolingAggregator(
    dim=REP_DIM, num_queries=NUM_QUERIES, theta_base=THETA_BASE, reversed=REVERSED, temperature=TEMPERATURE
)
# 前処理パイプライン: 初期化 → Learnable RoPE → RoPE整合注意プーリング
process_list = DataProcessList(
    iterable=[Initializer(), MultiplicativeSinusoidalPositionalEncoder(a=1000, b=1, gamma=1 / 2), Aggregator("mean")]
)

# Dataloader 構築
input_props: list[str] = []
output_props: list[str] = ["rt"]


protein_list = ProteinList.load_from_hdf5(str(INPUT_H5))

dataloader_config = DataloaderConfig(
    protein_list=protein_list,
    input_props=input_props,
    output_props=output_props,
    batch_size=64,
    cacheable=False,
    process_list=process_list,
)
dataloader = Dataloader(config=dataloader_config)

# モデル
model_input_dim = dataloader.output_dim(input_dim=REP_DIM)
model = BasicModel(input_dim=model_input_dim)

optimizer = RAdamScheduleFree(
    [
        {"params": model.parameters(), "lr": 1e-3},
        # {"params": lrope.parameters(), "lr": 5e-4},
        # {"params": ropa.parameters(), "lr": 5e-4},
    ]
)

rope_learnable_attnpool_config = TrainingRunnerConfig(
    input_hdf5_path=INPUT_H5,
    output_hdf5_path=OUTPUT_H5,
    dataset_name="plasma_lumos_1h_rope_learnable_attnpool",
    dataloader=dataloader,
    model=model,
    optimizer=optimizer,
    patience=100,
)
