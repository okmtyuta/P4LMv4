#!/usr/bin/env python3
from __future__ import annotations

import h5py
import torch

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.train.recorder import TrainRecorder
from src.modules.train.train_result import EpochPhaseResult, EpochSummary


def _make_phase(epoch: int, n: int) -> EpochPhaseResult:
    # 完全一致 -> pearson=1
    output = torch.arange(n, dtype=torch.float32)[:, None]
    label = output.clone()
    plist = ProteinList([Protein(key=str(i), props={"seq": "A", "rt": float(i)}) for i in range(n)])
    phase = EpochPhaseResult(
        epoch=epoch,
        output=output,
        label=label,
        input_props=[],
        output_props=["rt"],
        protein_list=plist,
    )
    return phase.compute_criteria()


def test_train_result_and_recorder(tmp_path):
    # 1) 2エポック分の要約を作成
    e1 = EpochSummary(epoch=1, train=_make_phase(1, 5), validate=_make_phase(1, 5), evaluate=_make_phase(1, 5))
    e2 = EpochSummary(epoch=2, train=_make_phase(2, 5), validate=_make_phase(2, 5), evaluate=_make_phase(2, 5))

    # 2) Recorder に登録し、belle を確認
    rec = TrainRecorder(patience=1)
    rec.append_epoch_results(e1)
    assert rec.belle_epoch_summary is not None
    assert rec.belle_epoch_summary.epoch == 1

    rec.next_epoch()
    assert rec.current_epoch == 2
    assert rec.to_continue() is False  # patience=1 なので停止

    # 3) 追加後も履歴が正しく保持される
    rec.append_epoch_results(e2)
    assert len(rec.history) == 2

    # 4) evaluate の予測書き戻しが save() 内で行われる（finalize 経由）
    out = tmp_path / "train_result.h5"
    rec.save(str(out))
    assert out.exists()

    # 5) HDF5 の基本構造チェック
    with h5py.File(out, "r") as f:
        assert "result" in f
        root = f["result"]
        assert "history" in root
        # belle は存在しても/しなくても良いが、属性は保存される
        assert "current_epoch" in root.attrs
        assert "patience" in root.attrs
