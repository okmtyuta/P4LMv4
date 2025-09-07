#!/usr/bin/env python3
"""
src/main/training.py の統合テスト。

小規模な合成 ProteinList を HDF5 に保存し、TrainingRunner で学習～保存まで
通しで動作することを確認する。
"""

from __future__ import annotations

from pathlib import Path

import torch
from schedulefree import RAdamScheduleFree

from src.main.training import TrainingRunner, TrainingRunnerConfig
from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.model.basic import BasicModel
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


def _make_synthetic_h5(path: Path, n: int, rep_dim: int) -> None:
    """長さ可変の合成 ProteinList を作成して保存する。"""
    torch.manual_seed(0)
    proteins: list[Protein] = []
    for i in range(n):
        L = 5 + (i % 4)  # 5..8
        seq = "A" * L
        reps = torch.randn(L, rep_dim)
        props = {"seq": seq, "rt": float(i), "length": L}
        proteins.append(Protein(key=str(i), props=props, representations=reps))

    ProteinList(proteins).save_as_hdf5(str(path))


def test_training_runner_smoke(tmp_path, monkeypatch):
    # Slack 通知をテスト中は無効化する
    class _DummySlackService:
        def __init__(self) -> None:
            pass

        def send(self, text: str) -> bool:
            return True

    monkeypatch.setattr("src.main.training.SlackService", _DummySlackService)

    # 1) 合成データを保存
    inp = tmp_path / "input.h5"
    out = tmp_path / "train_result.h5"
    _make_synthetic_h5(inp, n=20, rep_dim=16)

    # 2) パイプライン・モデル・最適化
    process_list = DataProcessList(iterable=[Initializer(), Aggregator("mean")])
    input_props: list[str] = []
    output_props: list[str] = ["rt"]

    # Dataloader を外部で構築（HDF5 からロード）
    plist = ProteinList.load_from_hdf5(str(inp))
    dl_conf = DataloaderConfig(
        protein_list=plist,
        input_props=input_props,
        output_props=output_props,
        batch_size=8,
        cacheable=False,
        process_list=process_list,
    )
    dataloader = Dataloader(config=dl_conf)

    # dataloader.output_dim を用いてモデル入力次元を決定
    model_input_dim = dataloader.output_dim(input_dim=16)

    model = BasicModel(input_dim=model_input_dim)
    optimizer = RAdamScheduleFree([{"params": model.parameters(), "lr": 1e-3}])

    # 3) 実行
    cfg = TrainingRunnerConfig(
        input_hdf5_path=str(inp),
        output_hdf5_path=str(out),
        dataset_name="test",
        dataloader=dataloader,
        model=model,
        optimizer=optimizer,
        patience=1,
        shuffle_seed=123,
    )

    rec = TrainingRunner(cfg).run()

    # 4) 保存物と結果の妥当性
    assert out.exists()
    assert rec.belle_epoch_summary is not None
    assert len(rec.history) >= 1
