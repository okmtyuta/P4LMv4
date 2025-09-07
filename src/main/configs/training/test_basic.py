#!/usr/bin/env python3
"""
最小構成のトレーニング設定（extraction/test.py の出力を前提）。
"""

from schedulefree import RAdamScheduleFree  # type: ignore[import-untyped]

from src.main.training import TrainingRunnerConfig
from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.helper.helper import Helper
from src.modules.model.basic import BasicModel

# 入力は `src/main/configs/extraction/test.py` で生成される HDF5 を想定
INPUT_H5 = Helper.ROOT / "outputs" / "test" / "test_data_esm2.h5"
OUTPUT_H5 = Helper.ROOT / "outputs" / "test" / "train_result.h5"

process_list = DataProcessList(iterable=[Initializer(), Aggregator("mean")])

# --- モデルの入力次元を手動で算出 ---
representation_dim = 1280  # ESM2 の埋め込み次元
input_props: list[str] = []
output_props: list[str] = ["rt"]

processed_dim = process_list.output_dim(input_dim=representation_dim)
model_input_dim = processed_dim + len(input_props)

model = BasicModel(input_dim=model_input_dim)


# 学習可能な DataProcess のパラメータも最適化対象に含める（例: LearnableRoPE / Floater 等）
def collect_pipeline_params(pl: DataProcessList):
    params = []
    for p in pl:
        if hasattr(p, "parameters"):
            params.extend(list(p.parameters()))  # type: ignore[attr-defined]
    return params


optimizer = RAdamScheduleFree(
    [
        {"params": model.parameters(), "lr": 1e-3},
        {"params": collect_pipeline_params(process_list), "lr": 5e-4},
    ]
)

test_training_config = TrainingRunnerConfig(
    input_hdf5_path=INPUT_H5,
    output_hdf5_path=OUTPUT_H5,
    dataset_name="test",
    input_props=input_props,
    output_props=output_props,
    batch_size=32,
    cacheable=False,
    process_list=process_list,
    model=model,
    optimizer=optimizer,
    patience=3,
    shuffle_seed=123456,
)
