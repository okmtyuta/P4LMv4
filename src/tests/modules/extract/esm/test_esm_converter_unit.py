#!/usr/bin/env python3
"""
ESMConverter/_ESMLanguage の軽量スタブテスト。
実際の fair-esm を起動せず、`esm.pretrained.*` をモンキーパッチして形状を検証する。
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from src.modules.extract.language.esm._esm import _ESMLanguage
from src.modules.extract.language.esm.esm_converter import ESMConverter
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList


class _StubAlphabet:
    padding_idx = 0

    @staticmethod
    def get_batch_converter() -> Callable[[list[tuple[str, str]]], tuple[list[str], list[str], torch.Tensor]]:
        def _conv(pairs: list[tuple[str, str]]):
            seqs = [s for _, s in pairs]
            lens = [len(s) + 2 for s in seqs]  # +2 for [CLS],[SEP]
            max_len = max(lens)
            toks = torch.zeros(len(seqs), max_len, dtype=torch.long)
            for i, L in enumerate(lens):
                toks[i, :L] = 1  # non-padding
            labels = [s for s in seqs]
            strs = list(seqs)
            return labels, strs, toks

        return _conv


class _StubModel:
    def eval(self):
        return self

    def __call__(self, tokens: torch.Tensor, repr_layers: list[int], return_contacts: bool) -> dict[str, Any]:
        B, T = tokens.shape
        D = 8
        # ダミー表現（非ゼロ領域に依存する値）
        reps = torch.randn(B, T, D)
        return {"representations": {repr_layers[0]: reps}}


class _StubESM:
    class pretrained:  # type: ignore[override]
        @staticmethod
        def esm2_t33_650M_UR50D():
            return _StubModel(), _StubAlphabet()

        @staticmethod
        def esm1b_t33_650M_UR50S():
            return _StubModel(), _StubAlphabet()


def test_esm_converter_call(monkeypatch):
    # esm モジュールをスタブ化
    import src.modules.extract.language.esm.esm_converter as conv_mod

    monkeypatch.setattr(conv_mod, "esm", _StubESM())

    conv = ESMConverter("esm2")
    seqs = ["AAAAA", "BBBB"]
    outs = conv(seqs)

    assert len(outs) == 2
    assert outs[0].shape == (5, 8)
    assert outs[1].shape == (4, 8)


def test__esm_language_with_stubbed_converter(monkeypatch):
    # Converter を置き換え、_ESMLanguage が representations を設定することを確認
    class _FakeConv:
        def __init__(self, model_name: str) -> None:
            pass

        def __call__(self, seqs: list[str]) -> list[torch.Tensor]:
            return [torch.ones(len(s), 8) for s in seqs]

    import src.modules.extract.language.esm._esm as esm_lang

    monkeypatch.setattr(esm_lang, "ESMConverter", _FakeConv)

    lang = _ESMLanguage("esm2")
    plist = ProteinList([Protein(key="k", props={"seq": "ABCDE"})])
    out = lang(plist)
    assert out[0].representations is not None
    assert out[0].representations.shape == (5, 8)
