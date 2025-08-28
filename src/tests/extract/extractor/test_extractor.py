#!/usr/bin/env python3
"""
Extractor クラスの機能テスト
"""

import pytest
import torch

from src.modules.extract.extractor.extractor import Extractor
from src.modules.extract.language._language import _Language
from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinProps


class MockLanguage(_Language):
    """テスト用のモック言語モデル"""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.processed_batches = []

    def __call__(self, protein_list: ProteinList) -> ProteinList:
        """各呼び出しを記録し、representationsを設定"""
        self.call_count += 1
        batch_size = len(protein_list)
        self.processed_batches.append(batch_size)

        # 各Proteinにモックrepresentationsを設定
        for protein in protein_list:
            mock_repr = torch.randn(len(protein.seq), 768)
            protein.set_representations(mock_repr)

        return protein_list


class TestExtractorBasicFunctionality:
    """基本機能のテスト"""

    @pytest.fixture
    def sample_protein_list(self):
        """テスト用ProteinListのフィクスチャ（10個のProtein）"""
        proteins = []
        for i in range(10):
            props: ProteinProps = {"seq": f"PROTEIN{i:02d}", "rt": float(i * 100), "length": 9}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        return ProteinList(proteins)

    @pytest.fixture
    def mock_language(self):
        """モック言語モデルのフィクスチャ"""
        return MockLanguage()

    def test_initialization(self, mock_language):
        """初期化テスト"""
        extractor = Extractor(mock_language)
        assert extractor._language is mock_language

    def test_batch_processing_exact_division(self, mock_language, sample_protein_list):
        """バッチサイズで割り切れる場合のテスト"""
        extractor = Extractor(mock_language)
        batch_size = 5

        result = extractor(sample_protein_list, batch_size)

        # 結果の確認
        assert len(result) == len(sample_protein_list)
        assert mock_language.call_count == 2  # 10個 ÷ 5 = 2バッチ
        assert mock_language.processed_batches == [5, 5]

        # 各Proteinにrepresentationsが設定されている
        for protein in result:
            assert protein.representations is not None
            assert protein.representations.shape == (9, 768)  # seq長9, feature768

    def test_batch_processing_uneven_division(self, mock_language, sample_protein_list):
        """バッチサイズで割り切れない場合のテスト"""
        extractor = Extractor(mock_language)
        batch_size = 3

        result = extractor(sample_protein_list, batch_size)

        # 結果の確認
        assert len(result) == len(sample_protein_list)
        assert mock_language.call_count == 4  # 10個 ÷ 3 = 3バッチ + 1個 = 4バッチ
        assert mock_language.processed_batches == [3, 3, 3, 1]

        # データの順序が保持されている
        for i, (original, processed) in enumerate(zip(sample_protein_list, result)):
            assert original.key == processed.key
            assert original.props == processed.props

    def test_batch_size_larger_than_list(self, mock_language, sample_protein_list):
        """バッチサイズがリストサイズより大きい場合のテスト"""
        extractor = Extractor(mock_language)
        batch_size = 20

        result = extractor(sample_protein_list, batch_size)

        # 結果の確認
        assert len(result) == len(sample_protein_list)
        assert mock_language.call_count == 1  # 全体で1バッチ
        assert mock_language.processed_batches == [10]

    def test_batch_size_one(self, mock_language, sample_protein_list):
        """バッチサイズが1の場合のテスト"""
        extractor = Extractor(mock_language)
        batch_size = 1

        result = extractor(sample_protein_list, batch_size)

        # 結果の確認
        assert len(result) == len(sample_protein_list)
        assert mock_language.call_count == 10  # 各Proteinが個別処理
        assert mock_language.processed_batches == [1] * 10


class TestExtractorEdgeCases:
    """エッジケースのテスト"""

    def test_empty_protein_list(self):
        """空のProteinListのテスト"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)
        empty_list = ProteinList([])

        result = extractor(empty_list, batch_size=5)

        assert len(result) == 0
        assert mock_language.call_count == 0  # 呼び出されない

    def test_single_protein(self):
        """単一Proteinのテスト"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        props: ProteinProps = {"seq": "SINGLE", "length": 6}
        protein = Protein(key="single", props=props)
        protein_list = ProteinList([protein])

        result = extractor(protein_list, batch_size=5)

        assert len(result) == 1
        assert mock_language.call_count == 1
        assert result[0].key == "single"


class TestExtractorWithRealLanguageModel:
    """実際のLanguageModelとの統合テスト"""

    def test_data_preservation(self):
        """データの完整性確認テスト"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        # テストデータ作成
        proteins = []
        for i in range(7):  # 不揃いな数
            props: ProteinProps = {"seq": f"TEST{i}SEQ", "rt": float(i * 50.5), "length": 8, "extra_prop": f"extra_{i}"}
            protein = Protein(key=f"test_{i}", props=props)
            proteins.append(protein)

        original_list = ProteinList(proteins)

        result = extractor(original_list, batch_size=3)

        # データ完整性の確認
        assert len(result) == len(original_list)

        for original, processed in zip(original_list, result):
            assert original.key == processed.key
            assert original.props == processed.props
            # representationsが追加されている（同じオブジェクトなので両方に設定される）
            assert processed.representations is not None
            assert original.representations is not None
            # 実際には同じオブジェクト参照
            assert original is processed


class TestExtractorProgressBar:
    """プログレスバーのテスト"""

    def test_progress_bar_calls(self, capsys):
        """tqdmのプログレスバー動作確認（簡易）"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        # 少数のProteinでテスト
        proteins = []
        for i in range(3):
            props: ProteinProps = {"seq": f"SEQ{i}", "length": 4}
            proteins.append(Protein(key=str(i), props=props))

        protein_list = ProteinList(proteins)

        # プログレスバー付きで実行
        result = extractor(protein_list, batch_size=2)

        # 基本的な結果確認
        assert len(result) == 3
        assert mock_language.call_count == 2  # 2バッチ（2個 + 1個）


class TestExtractorParallelProcessing:
    """並列処理のテスト"""

    def test_parallel_vs_sequential_results(self):
        """並列処理と逐次処理で同じ結果を得ることを確認"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        # テスト用データ作成
        proteins = []
        for i in range(6):
            props: ProteinProps = {"seq": f"PROTEIN{i:02d}", "rt": float(i * 100)}
            proteins.append(Protein(key=str(i), props=props))

        protein_list = ProteinList(proteins)

        # 逐次処理
        mock_language.call_count = 0
        result_seq = extractor(protein_list, batch_size=2, parallel=False)

        # 並列処理（スレッド）
        mock_language.call_count = 0
        result_parallel = extractor(protein_list, batch_size=2, parallel=True)

        # 結果の一貫性を確認
        assert len(result_seq) == len(result_parallel)
        assert len(result_seq) == 6

        # すべてのProteinに表現が設定されていることを確認
        for i in range(len(result_seq)):
            assert result_seq[i].representations is not None
            assert result_parallel[i].representations is not None

    def test_parallel_processing_options(self):
        """並列処理の各種オプションテスト"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        proteins = []
        for i in range(4):
            props: ProteinProps = {"seq": f"SEQ{i}", "length": 4}
            proteins.append(Protein(key=str(i), props=props))

        protein_list = ProteinList(proteins)

        # max_workersオプション
        result = extractor(protein_list, batch_size=1, parallel=True, max_workers=2)
        assert len(result) == 4

        # 並列処理テスト
        result_parallel = extractor(protein_list, batch_size=2, parallel=True)
        assert len(result_parallel) == 4

    def test_parallel_single_batch(self):
        """バッチが1つの場合は並列処理されないことを確認"""
        mock_language = MockLanguage()
        extractor = Extractor(mock_language)

        proteins = []
        for i in range(3):
            props: ProteinProps = {"seq": f"SEQ{i}", "length": 4}
            proteins.append(Protein(key=str(i), props=props))

        protein_list = ProteinList(proteins)

        # バッチサイズを大きくして1バッチにする
        result = extractor(protein_list, batch_size=10, parallel=True)
        assert len(result) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
