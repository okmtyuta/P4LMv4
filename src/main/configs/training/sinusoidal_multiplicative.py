#!/usr/bin/env python3
"""
MultiplicativeSinusoidalPositionalEncoder を用いた学習設定。

目的:
- 乗法的サイン波位置エンコードと単純平均集約の組み合わせを評価し、
  加法的手法との性能差（収束速度・汎化誤差）を比較する。

比較対象:
- `src/main/configs/training/sinusoidal_additive.py`
"""

import torch
from schedulefree import RAdamScheduleFree

from src.main.training import TrainingRunnerConfig
from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.data_process.positional_encoder import MultiplicativeSinusoidalPositionalEncoder
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.helper.helper import Helper
from src.modules.model.basic import BasicModel
from src.modules.protein.protein_list import ProteinList

# 入力/出力 HDF5（plasma_lumos_1h を使用）
INPUT_H5 = Helper.ROOT / "outputs" / "plasma_lumos_1h" / "plasma_lumos_1h_data_esm2.h5"
OUTPUT_H5 = (
    Helper.ROOT / "outputs" / "experiments" / "plasma_lumos_1h" / "sinusoidal" / "multiplicative" / "train_result.h5"
)

# 前処理パイプライン: 初期化 → 乗法的PE → 平均集約
process_list = DataProcessList(
    iterable=[
        Initializer(),
        MultiplicativeSinusoidalPositionalEncoder(a=10000.0, b=1.0, gamma=0.0),
        Aggregator("mean"),
    ]
)

# Dataloader を外部で構築
input_props: list[str] = []
output_props: list[str] = ["rt"]

torch.manual_seed(20240907)
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

# モデルの入力次元は前処理で決定
model_input_dim = dataloader.output_dim(input_dim=1280)
model = BasicModel(input_dim=model_input_dim)

optimizer = RAdamScheduleFree(
    [
        {"params": model.parameters(), "lr": 1e-3},
    ]
)

multiplicative_config = TrainingRunnerConfig(
    input_hdf5_path=INPUT_H5,
    output_hdf5_path=OUTPUT_H5,
    dataset_name="plasma_lumos_1h_sinusoidal_multiplicative",
    dataloader=dataloader,
    model=model,
    optimizer=optimizer,
    patience=100,
)
