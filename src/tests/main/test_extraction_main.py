#!/usr/bin/env python3
"""
src/main/extraction.py の統合テスト。

実際の ESM モデルを使うと重いので、Runner 内で生成する言語モデルを
テストスタブに差し替えて実行可否と保存物の妥当性を検証する。
"""

from __future__ import annotations

import torch

from src.main.extraction import ExtractionRunner, ExtractionRunnerConfig
from src.modules.extract.language._language import _Language
from src.modules.helper.helper import Helper
from src.modules.protein.protein_list import ProteinList


class _FakeLanguage(_Language):
    """長さ×8 の決定的テンソルを各 Protein に与える軽量スタブ。"""

    name = "fake"

    def __call__(self, protein_list: ProteinList) -> ProteinList:  # type: ignore[override]
        for i, p in enumerate(protein_list):
            L = len(p.seq)
            # 形状 (L, 8)・決定的
            reps = torch.arange(L * 8, dtype=torch.float32).reshape(L, 8) + float(i)
            p.set_representations(representations=reps)
        return protein_list


def test_extraction_runner_smoke(monkeypatch, tmp_path):
    # 言語モデル生成をスタブ化
    monkeypatch.setattr(
        ExtractionRunner,
        "_create_language_model",
        staticmethod(lambda name: _FakeLanguage()),  # type: ignore[arg-type]
    )

    csv_path = Helper.ROOT / "data" / "test" / "data.csv"
    out_path = tmp_path / "extracted.h5"

    cfg = ExtractionRunnerConfig(
        csv_path=str(csv_path),
        output_path=str(out_path),
        dataset_name="test",
        language_model="esm2",
        batch_size=16,
        parallel=False,
        max_workers=None,
    )

    runner = ExtractionRunner(cfg)
    result = runner.run()

    # 保存物の存在と内容を検証
    assert out_path.exists()
    loaded = ProteinList.load_from_hdf5(str(out_path))
    assert len(loaded) == len(result)

    # 先頭2件で形状チェック（L×8）
    for p in loaded[:2]:
        reps = p.get_representations()
        assert reps.ndim == 2
        assert reps.shape[1] == 8
