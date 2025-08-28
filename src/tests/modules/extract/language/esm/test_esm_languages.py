#!/usr/bin/env python3
"""
ESM1bLanguage と ESM2Language の機能テスト
"""

import pytest
import torch

from src.modules.extract.language.esm.esm1b import ESM1bLanguage
from src.modules.extract.language.esm.esm2 import ESM2Language
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinProps


class TestESM1bLanguage:
    """ESM1bLanguageのテスト"""

    @pytest.fixture
    def esm1b_model(self):
        """ESM1bLanguageのフィクスチャ"""
        return ESM1bLanguage()

    @pytest.fixture
    def sample_protein_list(self):
        """テスト用ProteinListのフィクスチャ"""
        proteins = []
        test_sequences = [
            "MKLLVLSLCF",  # 10 amino acids
            "ASTSSMTSVP",  # 10 amino acids
            "KPLQVWERTY",  # 10 amino acids
        ]

        for i, seq in enumerate(test_sequences):
            props: ProteinProps = {"seq": seq, "rt": float(i * 100), "length": len(seq)}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        return ProteinList(proteins)

    def test_initialization(self, esm1b_model):
        """初期化テスト"""
        assert esm1b_model.name == "esm1b"
        assert hasattr(esm1b_model, "_converter")
        assert esm1b_model._converter._model_name == "esm1b"

    def test_call_method(self, esm1b_model, sample_protein_list):
        """__call__メソッドのテスト"""
        # 初期状態でrepresentationsがNoneであることを確認
        for protein in sample_protein_list:
            assert protein.representations is None

        # ESM1bLanguageを適用
        result = esm1b_model(sample_protein_list)

        # 同じオブジェクトが返されることを確認
        assert result is sample_protein_list

        # 各Proteinにrepresentationsが設定されていることを確認
        for protein in sample_protein_list:
            assert protein.representations is not None
            assert isinstance(protein.representations, torch.Tensor)
            # ESM1bの特徴量次元は1280
            assert protein.representations.shape[1] == 1280
            # 配列長 - 2 (CLS, SEP除去) の長さになる
            seq_len = len(protein.seq)
            assert protein.representations.shape[0] == seq_len

    def test_representations_shape_consistency(self, esm1b_model, sample_protein_list):
        """異なる配列長でのrepresentationsの形状テスト"""
        esm1b_model(sample_protein_list)

        for protein in sample_protein_list:
            seq_len = len(protein.seq)
            representations = protein.representations

            # 形状の確認
            assert representations.shape == (seq_len, 1280)
            assert representations.dtype == torch.float32


class TestESM2Language:
    """ESM2Languageのテスト"""

    @pytest.fixture
    def esm2_model(self):
        """ESM2Languageのフィクスチャ"""
        return ESM2Language()

    @pytest.fixture
    def sample_protein_list(self):
        """テスト用ProteinListのフィクスチャ"""
        proteins = []
        test_sequences = ["MKLLVLSLCF", "ASTSSMTSVP", "KPLQVWERTY"]

        for i, seq in enumerate(test_sequences):
            props: ProteinProps = {"seq": seq, "rt": float(i * 100), "length": len(seq)}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        return ProteinList(proteins)

    def test_initialization(self, esm2_model):
        """初期化テスト"""
        assert esm2_model.name == "esm2"
        assert hasattr(esm2_model, "_converter")
        assert esm2_model._converter._model_name == "esm2"

    def test_call_method(self, esm2_model, sample_protein_list):
        """__call__メソッドのテスト"""
        # 初期状態でrepresentationsがNoneであることを確認
        for protein in sample_protein_list:
            assert protein.representations is None

        # ESM2Languageを適用
        result = esm2_model(sample_protein_list)

        # 同じオブジェクトが返されることを確認
        assert result is sample_protein_list

        # 各Proteinにrepresentationsが設定されていることを確認
        for protein in sample_protein_list:
            assert protein.representations is not None
            assert isinstance(protein.representations, torch.Tensor)
            # ESM2の特徴量次元は1280
            assert protein.representations.shape[1] == 1280
            # 配列長の長さになる（CLS, SEP除去済み）
            seq_len = len(protein.seq)
            assert protein.representations.shape[0] == seq_len

    def test_representations_shape_consistency(self, esm2_model, sample_protein_list):
        """異なる配列長でのrepresentationsの形状テスト"""
        esm2_model(sample_protein_list)

        for protein in sample_protein_list:
            seq_len = len(protein.seq)
            representations = protein.representations

            # 形状の確認
            assert representations.shape == (seq_len, 1280)
            assert representations.dtype == torch.float32


class TestESMLanguageComparison:
    """ESM1bLanguageとESM2Languageの比較テスト"""

    @pytest.fixture
    def sample_protein_list(self):
        """テスト用ProteinListのフィクスチャ"""
        proteins = []
        test_sequences = ["MKLLVLSLCF"]  # 単一配列でテスト

        for i, seq in enumerate(test_sequences):
            props: ProteinProps = {"seq": seq, "rt": float(i * 100), "length": len(seq)}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        return ProteinList(proteins)

    def test_both_models_produce_same_shape(self, sample_protein_list):
        """両モデルが同じ形状のrepresentationsを生成することを確認"""
        esm1b = ESM1bLanguage()
        esm2 = ESM2Language()

        # 別々のProteinListを作成（同じデータ）
        protein_list_1b = ProteinList(
            [
                Protein(key=p.key, props=dict(p.props), representations=None, processed=None, predicted={})
                for p in sample_protein_list
            ]
        )

        protein_list_2 = ProteinList(
            [
                Protein(key=p.key, props=dict(p.props), representations=None, processed=None, predicted={})
                for p in sample_protein_list
            ]
        )

        # 各モデルを適用
        esm1b(protein_list_1b)
        esm2(protein_list_2)

        # 形状が同じことを確認
        for p1b, p2 in zip(protein_list_1b, protein_list_2):
            assert p1b.representations.shape == p2.representations.shape
            assert p1b.representations.dtype == p2.representations.dtype

    def test_models_produce_different_values(self, sample_protein_list):
        """両モデルが異なる値のrepresentationsを生成することを確認"""
        esm1b = ESM1bLanguage()
        esm2 = ESM2Language()

        # 別々のProteinListを作成
        protein_list_1b = ProteinList(
            [
                Protein(key=p.key, props=dict(p.props), representations=None, processed=None, predicted={})
                for p in sample_protein_list
            ]
        )

        protein_list_2 = ProteinList(
            [
                Protein(key=p.key, props=dict(p.props), representations=None, processed=None, predicted={})
                for p in sample_protein_list
            ]
        )

        # 各モデルを適用
        esm1b(protein_list_1b)
        esm2(protein_list_2)

        # 値が異なることを確認
        for p1b, p2 in zip(protein_list_1b, protein_list_2):
            assert not torch.equal(p1b.representations, p2.representations)


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_protein_list(self):
        """空のProteinListの処理テスト"""
        empty_list = ProteinList([])
        esm2 = ESM2Language()

        result = esm2(empty_list)
        assert result is empty_list
        assert len(result) == 0

    def test_single_amino_acid_sequence(self):
        """単一アミノ酸配列のテスト"""
        props: ProteinProps = {"seq": "M", "length": 1}
        protein = Protein(key="single", props=props)
        protein_list = ProteinList([protein])

        esm2 = ESM2Language()
        result = esm2(protein_list)

        assert result[0].representations.shape == (1, 1280)

    def test_long_sequence(self):
        """長い配列のテスト"""
        # 50アミノ酸の配列（有効なアミノ酸のみ使用）
        long_seq = "MKLLVLSLCFAQDVASNPLLKELASRTSMTSVPKPLQVWERTYGLMNGH"
        props: ProteinProps = {"seq": long_seq, "length": len(long_seq)}
        protein = Protein(key="long", props=props)
        protein_list = ProteinList([protein])

        esm1b = ESM1bLanguage()
        result = esm1b(protein_list)

        seq_len = len(long_seq)
        assert result[0].representations.shape == (seq_len, 1280)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
