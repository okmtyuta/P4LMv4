#!/usr/bin/env python3
"""
最小構成のトレーニング設定（extraction/test.py の出力を前提）。
"""

from schedulefree import RAdamScheduleFree

from src.main.training import TrainingRunnerConfig
from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.helper.helper import Helper
from src.modules.model.basic import BasicModel
from src.modules.protein.protein_list import ProteinList

# 入力は `src/main/configs/extraction/test.py` で生成される HDF5 を想定
INPUT_H5 = Helper.ROOT / "outputs" / "test" / "test_data_esm2.h5"
OUTPUT_H5 = Helper.ROOT / "outputs" / "test" / "train_result.h5"

process_list = DataProcessList(iterable=[Initializer(), Aggregator("mean")])

# --- Dataloader を外部で構築し、output_dim を利用 ---
input_props: list[str] = []
output_props: list[str] = ["rt"]

protein_list = ProteinList.load_from_hdf5(str(INPUT_H5))
dataloader_config = DataloaderConfig(
    protein_list=protein_list,
    input_props=input_props,
    output_props=output_props,
    batch_size=32,
    cacheable=False,
    process_list=process_list,
)
dataloader = Dataloader(config=dataloader_config)

model_input_dim = dataloader.output_dim(input_dim=1280)
model = BasicModel(input_dim=model_input_dim)

# モデルのみを最適化対象にする。
optimizer = RAdamScheduleFree(
    [
        {"params": model.parameters(), "lr": 1e-3},
    ]
)

test_config = TrainingRunnerConfig(
    input_hdf5_path=INPUT_H5,
    output_hdf5_path=OUTPUT_H5,
    dataset_name="test",
    dataloader=dataloader,
    model=model,
    optimizer=optimizer,
)
