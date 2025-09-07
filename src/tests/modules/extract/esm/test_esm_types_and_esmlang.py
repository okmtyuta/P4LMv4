#!/usr/bin/env python3
from src.modules.extract.language.esm._esm import _ESMLanguage
from src.modules.extract.language.esm.esm_types import ESMModelName


def test_esm_types_importable():
    # 型エイリアスが存在することの確認（実行時チェック）
    assert isinstance("esm2", str)
    v: ESMModelName = "esm2"
    assert v in ("esm2", "esm1b")


def test_esmlang_instantiation(monkeypatch):
    # _ESMLanguage が Converter を用いることは別テストで確認済み。ここでは単純生成の確認。
    class _DummyConv:
        def __init__(self, model_name: str) -> None:
            pass

        def __call__(self, seqs):
            return []

    import src.modules.extract.language.esm._esm as esm_mod

    monkeypatch.setattr(esm_mod, "ESMConverter", _DummyConv)
    lang = _ESMLanguage("esm1b")
    assert lang is not None
