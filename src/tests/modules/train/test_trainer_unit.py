#!/usr/bin/env python3
from __future__ import annotations

import torch
from schedulefree import RAdamScheduleFree

from src.modules.data_process.aggregator import Aggregator
from src.modules.data_process.data_process_list import DataProcessList
from src.modules.data_process.initializer import Initializer
from src.modules.dataloader.dataloader import Dataloader, DataloaderConfig
from src.modules.model.basic import BasicModel
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.train.trainer import Trainer


def _make_list(n: int, L: int, D: int) -> ProteinList:
    torch.manual_seed(0)
    ps = []
    for i in range(n):
        reps = torch.randn(L, D)
        p = Protein(key=str(i), props={"seq": "A" * L, "rt": float(i)}, representations=reps)
        ps.append(p)
    return ProteinList(ps)


def test_trainer_smoke():
    # 18 サンプルにして検証/評価に 2 サンプル以上を確保
    protein_list = _make_list(n=18, L=6, D=8)
    process_list = DataProcessList([Initializer(), Aggregator("mean")])
    cfg = DataloaderConfig(
        protein_list=protein_list,
        input_props=[],
        output_props=["rt"],
        batch_size=8,
        cacheable=False,
        process_list=process_list,
    )
    dl = Dataloader(cfg)

    model = BasicModel(input_dim=dl.output_dim(input_dim=8))
    opt = RAdamScheduleFree([{"params": model.parameters(), "lr": 1e-3}])

    trainer = Trainer(model=model, dataloader=dl, optimizer=opt)
    # 早期停止を迅速化
    trainer._recorder._patience = 1
    trainer.train()

    assert len(trainer._recorder.history) >= 1
