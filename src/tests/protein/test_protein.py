#!/usr/bin/env python3
"""
Protein クラスの機能軸テスト
"""

import pytest
import torch

from src.modules.protein.protein import Protein
from src.modules.protein.protein_types import ProteinProps


class TestBasicProperties:
    """基本プロパティのテスト"""

    def test_basic_properties(self):
        """基本プロパティの取得テスト"""
        props: ProteinProps = {"seq": "ASTSSMTSVPKPLK", "rt": 2510.7222, "length": 14}
        protein = Protein(key="test_protein", props=props)

        assert protein.key == "test_protein"
        assert protein.props == props
        assert protein.predicted == {}
        assert protein.representations is None
        assert protein.processed is None

    def test_instance_properties(self):
        """インスタンスプロパティテスト"""
        props: ProteinProps = {"seq": "ABCDEF"}
        protein = Protein(key="instance_test", props=props)

        # インスタンス属性として直接アクセス可能
        assert protein.key == "instance_test"
        assert protein.props == props
        assert protein.representations is None
        assert protein.processed is None
        assert protein.predicted == {}


class TestPropsHandling:
    """プロパティ操作のテスト"""

    def test_read_props_success(self):
        """プロパティ読み込み成功テスト"""
        props: ProteinProps = {"seq": "ABCDEF", "rt": 123.45, "length": 6, "extra": "value"}
        protein = Protein(key="props_test", props=props)

        assert protein.read_props("rt") == 123.45
        assert protein.read_props("length") == 6
        assert protein.read_props("extra") == "value"

    def test_read_props_missing_key(self):
        """存在しないプロパティ読み込みエラーテスト"""
        props: ProteinProps = {"seq": "ABCDEF", "rt": 123.45}
        protein = Protein(key="error_test", props=props)

        with pytest.raises(RuntimeError) as exc_info:
            protein.read_props("missing_key")
        assert "missing_key" in str(exc_info.value)
        assert "not readable" in str(exc_info.value)

    def test_read_props_none_value(self):
        """Noneプロパティ読み込みエラーテスト"""
        props: ProteinProps = {"seq": "ABCDEF", "rt": None, "valid": 123}
        protein = Protein(key="none_test", props=props)

        with pytest.raises(RuntimeError) as exc_info:
            protein.read_props("rt")
        assert "rt" in str(exc_info.value)

        # 有効な値は正常に読める
        assert protein.read_props("valid") == 123

    def test_set_props(self):
        """プロパティ設定テスト"""
        props: ProteinProps = {"seq": "ABCDEF", "old": "value"}
        protein = Protein(key="set_test", props=props)

        new_props: ProteinProps = {"seq": "GHIJKL", "new": "data", "number": 42}
        result = protein.set_props(new_props)

        # メソッドチェイン確認
        assert result is protein

        # プロパティが置き換わった
        assert protein.props == new_props
        assert protein.seq == "GHIJKL"
        assert "old" not in protein.props


class TestRepresentations:
    """representations操作のテスト"""

    def test_representations_available(self):
        """representations取得成功テスト"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        props: ProteinProps = {"seq": "ABC"}
        protein = Protein(key="rep_test", props=props, representations=tensor)

        result = protein.get_representations()
        assert torch.equal(result, tensor)

    def test_representations_unavailable(self):
        """representations取得失敗テスト"""
        props: ProteinProps = {"seq": "ABC"}
        protein = Protein(key="no_rep_test", props=props)

        with pytest.raises(RuntimeError) as exc_info:
            protein.get_representations()
        assert "representations unavailable" in str(exc_info.value)

    def test_set_representations(self):
        """representations設定テスト"""
        props: ProteinProps = {"seq": "XYZ"}
        protein = Protein(key="set_rep_test", props=props)
        tensor = torch.tensor([4.0, 5.0, 6.0])

        result = protein.set_representations(tensor)

        # メソッドチェイン確認
        assert result is protein

        # representationsが設定された
        assert torch.equal(protein.get_representations(), tensor)
        assert torch.equal(protein.representations, tensor)


class TestProcessed:
    """processed操作のテスト"""

    def test_processed_available(self):
        """processed取得成功テスト"""
        tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        props: ProteinProps = {"seq": "DEFG"}
        protein = Protein(key="proc_test", props=props, processed=tensor)

        result = protein.get_processed()
        assert torch.equal(result, tensor)

    def test_processed_unavailable(self):
        """processed取得失敗テスト"""
        props: ProteinProps = {"seq": "DEFG"}
        protein = Protein(key="no_proc_test", props=props)

        with pytest.raises(RuntimeError) as exc_info:
            protein.get_processed()
        assert "processed unavailable" in str(exc_info.value)

    def test_set_processed(self):
        """processed設定テスト"""
        props: ProteinProps = {"seq": "MNOP"}
        protein = Protein(key="set_proc_test", props=props)
        tensor = torch.tensor([[7.0, 8.0], [9.0, 10.0]])

        result = protein.set_processed(tensor)

        # メソッドチェイン確認
        assert result is protein

        # processedが設定された
        assert torch.equal(protein.get_processed(), tensor)
        assert torch.equal(protein.processed, tensor)


class TestPredicted:
    """predicted操作のテスト"""

    def test_default_predicted(self):
        """predicted初期値テスト"""
        props: ProteinProps = {"seq": "TEST"}
        protein = Protein(key="pred_default_test", props=props)

        assert protein.predicted == {}

    def test_set_predicted(self):
        """predicted設定テスト"""
        props: ProteinProps = {"seq": "PRED"}
        protein = Protein(key="pred_test", props=props)
        predicted: ProteinProps = {"rt_pred": 1234.5, "class": "positive"}

        result = protein.set_predicted(predicted)

        # メソッドチェイン確認
        assert result is protein

        # predictedが設定された
        assert protein.predicted == predicted
        assert protein.predicted["rt_pred"] == 1234.5

    def test_predicted_initial_value(self):
        """predicted初期値を指定した場合のテスト"""
        props: ProteinProps = {"seq": "INITIAL"}
        initial_predicted: ProteinProps = {"score": 0.95}
        protein = Protein(key="initial_pred_test", props=props, predicted=initial_predicted)

        assert protein.predicted == initial_predicted
        assert protein.predicted["score"] == 0.95


class TestMethodChaining:
    """メソッドチェインのテスト"""

    def test_method_chaining(self):
        """複数メソッドのチェインテスト"""
        props: ProteinProps = {"seq": "CHAIN"}
        protein = Protein(key="chain_test", props=props)

        tensor1 = torch.tensor([1.0, 2.0])
        tensor2 = torch.tensor([3.0, 4.0])
        new_props: ProteinProps = {"seq": "UPDATED", "value": 42}
        predicted: ProteinProps = {"result": "success"}

        # メソッドチェイン実行
        result = (
            protein.set_representations(tensor1).set_processed(tensor2).set_props(new_props).set_predicted(predicted)
        )

        # 全て同じインスタンス
        assert result is protein

        # 全ての値が設定されている
        assert torch.equal(protein.representations, tensor1)
        assert torch.equal(protein.processed, tensor2)
        assert protein.props == new_props
        assert protein.predicted == predicted


class TestSerializableStorage:
    """SerializableStorage継承のテスト"""

    def test_to_dict_basic(self):
        """基本的な辞書変換テスト"""
        props: ProteinProps = {"seq": "SERIALIZE", "rt": 1000.0}
        protein = Protein(key="serialize_test", props=props)

        result = protein.to_dict()

        assert isinstance(result, dict)
        assert result["key"] == "serialize_test"
        assert result["props"] == props
        assert result["representations"] is None
        assert result["processed"] is None
        assert result["predicted"] == {}

    def test_to_dict_with_tensors(self):
        """テンソル付き辞書変換テスト"""
        props: ProteinProps = {"seq": "TENSOR_SERIALIZE"}
        tensor1 = torch.tensor([1.0, 2.0, 3.0])
        tensor2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        predicted: ProteinProps = {"confidence": 0.95}

        protein = Protein(
            key="tensor_serialize_test", props=props, representations=tensor1, processed=tensor2, predicted=predicted
        )

        result = protein.to_dict()

        assert result["key"] == "tensor_serialize_test"
        assert result["props"] == props
        assert torch.equal(result["representations"], tensor1)
        assert torch.equal(result["processed"], tensor2)
        assert result["predicted"] == predicted

    def test_from_dict_basic(self):
        """基本的な辞書からの復元テスト"""
        props: ProteinProps = {"seq": "DESERIALIZE", "length": 11}
        original = Protein(key="deserialize_test", props=props)

        data = original.to_dict()
        restored = Protein.from_dict(data)

        assert restored.key == original.key
        assert restored.props == original.props
        assert restored.representations == original.representations
        assert restored.processed == original.processed
        assert restored.predicted == original.predicted

    def test_roundtrip_serialization(self):
        """ラウンドトリップシリアライゼーションテスト"""
        props: ProteinProps = {"seq": "ROUNDTRIP", "rt": 2500.0, "length": 9}
        tensor1 = torch.tensor([10.0, 20.0, 30.0])
        tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        predicted: ProteinProps = {"rt_pred": 2480.0, "confidence": 0.88}

        original = Protein(
            key="roundtrip_test", props=props, representations=tensor1, processed=tensor2, predicted=predicted
        )

        # 辞書化 → 復元
        data = original.to_dict()
        restored = Protein.from_dict(data)

        # 全ての属性が正確に復元される
        assert restored.key == original.key
        assert restored.props == original.props
        assert torch.equal(restored.representations, original.representations)
        assert torch.equal(restored.processed, original.processed)
        assert restored.predicted == original.predicted
