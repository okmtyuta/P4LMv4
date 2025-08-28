#!/usr/bin/env python3
"""
ProteinList クラスの機能軸テスト
"""

import os
import tempfile

import h5py
import pytest
import torch

from src.modules.protein.protein import Protein
from src.modules.protein.protein_list import ProteinList
from src.modules.protein.protein_types import ProteinProps


class TestBasicFunctionality:
    """基本機能のテスト"""

    def test_initialization(self):
        """初期化テスト"""
        # 空のリスト
        empty_list = ProteinList([])
        assert len(empty_list) == 0
        assert empty_list.is_empty

        # Proteinオブジェクトのリスト
        proteins = []
        for i in range(3):
            props: ProteinProps = {"seq": f"PROTEIN{i}", "index": i}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        protein_list = ProteinList(proteins)
        assert len(protein_list) == 3
        assert not protein_list.is_empty

    def test_sequence_container_functionality(self):
        """SequenceContainer機能のテスト"""
        # テストデータ作成
        proteins = []
        for i in range(5):
            props: ProteinProps = {"seq": f"SEQ{i}", "value": i * 10}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        protein_list = ProteinList(proteins)

        # インデックスアクセス
        assert protein_list[0].key == "0"
        assert protein_list[-1].key == "4"

        # スライスアクセス
        subset = protein_list[1:3]
        assert len(subset) == 2
        assert subset[0].key == "1"
        assert subset[1].key == "2"

        # イテレーション
        keys = [protein.key for protein in protein_list]
        assert keys == ["0", "1", "2", "3", "4"]

        # 包含チェック
        assert proteins[0] in protein_list

    def test_addition(self):
        """リスト結合のテスト"""
        # 最初のリスト
        proteins1 = []
        for i in range(2):
            props: ProteinProps = {"seq": f"A{i}"}
            proteins1.append(Protein(key=f"a{i}", props=props))

        # 2番目のリスト
        proteins2 = []
        for i in range(2):
            props: ProteinProps = {"seq": f"B{i}"}
            proteins2.append(Protein(key=f"b{i}", props=props))

        list1 = ProteinList(proteins1)
        list2 = ProteinList(proteins2)

        # 結合
        combined = list1 + list2
        assert len(combined) == 4
        assert combined[0].key == "a0"
        assert combined[1].key == "a1"
        assert combined[2].key == "b0"
        assert combined[3].key == "b1"


class TestCsvFunctionality:
    """CSV機能のテスト"""

    def test_from_csv(self):
        """CSVからの読み込みテスト"""
        # テスト用CSVデータを作成
        csv_data = """index,seq,rt,length
0,ABCDEFGHIJ,1234.56,10
1,KLMNOPQRST,2345.67,10
2,UVWXYZABCD,3456.78,10"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_csv_path = f.name

        try:
            # CSVから読み込み
            protein_list = ProteinList.from_csv(temp_csv_path)

            assert len(protein_list) == 3

            # 最初のProteinの確認
            protein0 = protein_list[0]
            assert protein0.key == "0"
            assert protein0.props["seq"] == "ABCDEFGHIJ"
            assert protein0.props["rt"] == 1234.56
            assert protein0.props["length"] == 10
            # indexはkeyに使用されるためpropsには含まれない

            # 2番目のProteinの確認
            protein1 = protein_list[1]
            assert protein1.key == "1"
            assert protein1.props["seq"] == "KLMNOPQRST"
            assert protein1.props["rt"] == 2345.67

            # 3番目のProteinの確認
            protein2 = protein_list[2]
            assert protein2.key == "2"
            assert protein2.props["seq"] == "UVWXYZABCD"
            assert protein2.props["rt"] == 3456.78

        finally:
            os.unlink(temp_csv_path)

    def test_csv_props_handling(self):
        """CSVプロパティの処理確認"""
        csv_data = """index,seq,rt,length,species,function
0,MKLLVL,100.0,6,human,transport
1,ASTSMT,200.0,6,mouse,enzyme"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_data)
            temp_csv_path = f.name

        try:
            protein_list = ProteinList.from_csv(temp_csv_path)

            # 追加カラムもpropsに含まれる
            protein0 = protein_list[0]
            assert protein0.props["species"] == "human"
            assert protein0.props["function"] == "transport"

            protein1 = protein_list[1]
            assert protein1.props["species"] == "mouse"
            assert protein1.props["function"] == "enzyme"

        finally:
            os.unlink(temp_csv_path)


class TestHdf5Functionality:
    """HDF5機能のテスト"""

    def test_hdf5_roundtrip_basic(self):
        """基本的なHDF5往復テスト"""
        # テストデータ作成
        proteins = []
        for i in range(3):
            props: ProteinProps = {"seq": f"PROTEIN{i}", "rt": float(i * 100), "length": 8, "index": i}
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        original_list = ProteinList(proteins)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5_path = f.name

        try:
            # 保存
            original_list.save_as_hdf5(temp_h5_path)

            # 読み込み
            loaded_list = ProteinList.load_from_hdf5(temp_h5_path)

            # 確認
            assert len(loaded_list) == len(original_list)

            for i, (original, loaded) in enumerate(zip(original_list, loaded_list)):
                assert loaded.key == original.key
                assert loaded.props == original.props

        finally:
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)

    def test_hdf5_with_tensors(self):
        """テンソル付きHDF5テスト"""
        # テンソル付きProteinを作成
        proteins = []
        for i in range(2):
            props: ProteinProps = {"seq": f"SEQ{i}", "index": i}
            protein = Protein(key=str(i), props=props)

            # テンソルデータを追加
            representations = torch.randn(10, 768)
            processed = torch.randn(768)
            predicted: ProteinProps = {"score": float(i * 0.5)}

            protein.set_representations(representations)
            protein.set_processed(processed)
            protein.set_predicted(predicted)

            proteins.append(protein)

        original_list = ProteinList(proteins)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5_path = f.name

        try:
            # 保存・読み込み
            original_list.save_as_hdf5(temp_h5_path)
            loaded_list = ProteinList.load_from_hdf5(temp_h5_path)

            # テンソルデータの確認
            for i, (original, loaded) in enumerate(zip(original_list, loaded_list)):
                assert loaded.key == original.key
                assert loaded.props == original.props
                assert torch.equal(loaded.representations, original.representations)
                assert torch.equal(loaded.processed, original.processed)
                assert loaded.predicted == original.predicted

        finally:
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)

    def test_hdf5_group_names(self):
        """HDF5グループ名のテスト（SerializableContainerListではインデックス番号が使用される）"""
        # 特定のキーを持つProteinを作成
        proteins = []
        keys = ["protein_a", "protein_b", "test_123"]

        for i, key in enumerate(keys):
            props: ProteinProps = {"seq": f"SEQ{i}"}
            protein = Protein(key=key, props=props)
            proteins.append(protein)

        protein_list = ProteinList(proteins)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5_path = f.name

        try:
            # 保存
            protein_list.save_as_hdf5(temp_h5_path)

            # HDF5ファイルの構造を直接確認
            with h5py.File(temp_h5_path, "r") as f:
                group_names = list(f.keys())
                # SerializableContainerListは要素をインデックス番号（"0", "1", "2"）で保存
                expected_indices = [str(i) for i in range(len(keys))]
                assert set(group_names) == set(expected_indices)

                # 各グループの内容確認
                for i, expected_key in enumerate(keys):
                    group = f[str(i)]
                    # Proteinのデータが保存されているか確認
                    assert "key" in group
                    assert "props" in group
                    # keyの値が正しく保存されているか確認
                    stored_key = group["key"][()].decode() if isinstance(group["key"][()], bytes) else group["key"][()]
                    assert stored_key == expected_key

            # 読み込みテスト
            loaded_list = ProteinList.load_from_hdf5(temp_h5_path)
            loaded_keys = [p.key for p in loaded_list]
            assert set(loaded_keys) == set(keys)

        finally:
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)


class TestComplexScenarios:
    """複雑なシナリオのテスト"""

    def test_csv_to_hdf5_workflow(self):
        """CSV→HDF5ワークフローテスト"""
        # CSVデータ作成
        csv_data = """index,seq,rt,length,species
0,MKLLVLSLCF,1000.0,10,human
1,ASTSSMTSVP,2000.0,10,mouse
2,KPLQVWERTY,3000.0,10,rat"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_f.write(csv_data)
            temp_csv_path = csv_f.name

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as h5_f:
            temp_h5_path = h5_f.name

        try:
            # CSVから読み込み
            protein_list = ProteinList.from_csv(temp_csv_path)

            # ML処理をシミュレート
            for protein in protein_list:
                # representationsを設定
                seq_len = len(protein.seq)
                representations = torch.randn(seq_len + 2, 1280)
                protein.set_representations(representations)

                # processedを設定
                processed = torch.mean(representations, dim=0)
                protein.set_processed(processed)

                # predictedを設定
                predicted: ProteinProps = {"function_class": "membrane_protein", "confidence": 0.85}
                protein.set_predicted(predicted)

            # HDF5に保存
            protein_list.save_as_hdf5(temp_h5_path)

            # 読み込んで確認
            loaded_list = ProteinList.load_from_hdf5(temp_h5_path)

            assert len(loaded_list) == 3
            for original, loaded in zip(protein_list, loaded_list):
                assert loaded.key == original.key
                assert loaded.props == original.props
                assert torch.equal(loaded.representations, original.representations)
                assert torch.equal(loaded.processed, original.processed)
                assert loaded.predicted == original.predicted

        finally:
            if os.path.exists(temp_csv_path):
                os.unlink(temp_csv_path)
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)

    def test_large_protein_list(self):
        """大きなProteinListのテスト"""
        # 100個のProteinを作成
        proteins = []
        for i in range(100):
            props: ProteinProps = {
                "seq": f"PROTEIN{i:03d}SEQUENCE",
                "rt": float(i * 10.5),
                "length": 15,
                "index": i,
                "batch": i // 10,  # バッチ情報
            }
            protein = Protein(key=str(i), props=props)
            proteins.append(protein)

        large_list = ProteinList(proteins)

        # 基本確認
        assert len(large_list) == 100
        assert large_list[0].key == "0"
        assert large_list[-1].key == "99"

        # スライシング
        first_batch = large_list[:10]
        assert len(first_batch) == 10
        for i, protein in enumerate(first_batch):
            assert protein.key == str(i)
            assert protein.props["batch"] == 0

        # 分割テスト（SequenceContainerの機能）
        split_lists = large_list.split_equal(10)
        assert len(split_lists) == 10
        for split_list in split_lists:
            assert len(split_list) == 10


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_list_operations(self):
        """空リストの操作テスト"""
        empty_list = ProteinList([])

        # 空リストの保存・読み込み
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5_path = f.name

        try:
            empty_list.save_as_hdf5(temp_h5_path)
            loaded_empty = ProteinList.load_from_hdf5(temp_h5_path)

            assert len(loaded_empty) == 0
            assert loaded_empty.is_empty

        finally:
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)

    def test_special_characters_in_keys(self):
        """特殊文字を含むキーのテスト"""
        # 特殊文字を含むキー（HDF5で有効な範囲）
        special_keys = ["protein_1", "test-protein", "p123", "PROTEIN_A"]

        proteins = []
        for i, key in enumerate(special_keys):
            props: ProteinProps = {"seq": f"SEQ{i}"}
            protein = Protein(key=key, props=props)
            proteins.append(protein)

        protein_list = ProteinList(proteins)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_h5_path = f.name

        try:
            protein_list.save_as_hdf5(temp_h5_path)
            loaded_list = ProteinList.load_from_hdf5(temp_h5_path)

            loaded_keys = [p.key for p in loaded_list]
            assert set(loaded_keys) == set(special_keys)

        finally:
            if os.path.exists(temp_h5_path):
                os.unlink(temp_h5_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
