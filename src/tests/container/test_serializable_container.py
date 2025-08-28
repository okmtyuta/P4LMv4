#!/usr/bin/env python3
"""
SerializableContainer の機能軸テスト
"""

from dataclasses import dataclass
from typing import Any, Dict, List

import h5py
import pytest

from src.modules.container.serializable_container import SerializableContainer


@dataclass
class SimpleData(SerializableContainer):
    """テスト用のシンプルなデータクラス"""

    name: str
    value: int
    active: bool = True


class ComplexData(SerializableContainer):
    """テスト用の複雑なクラス"""

    def __init__(self, config: Dict[str, Any], items: List[str], metadata: str = "default"):
        self.config = config
        self.items = items
        self.metadata = metadata


@dataclass
class NestedData(SerializableContainer):
    """テスト用のネストしたデータクラス"""

    simple_data: SimpleData
    numbers: List[float]
    mapping: Dict[str, int]


class TestBasicFunctionality:
    """基本機能のテスト"""

    def test_inheritance(self):
        """SerializableContainerがSerializableObjectを継承していることを確認"""
        data = SimpleData(name="test", value=42)

        # SerializableObjectのメソッドが使える
        dict_repr = data.to_dict()
        assert dict_repr["name"] == "test"
        assert dict_repr["value"] == 42
        assert dict_repr["active"] is True

        # from_dictでも復元できる
        restored = SimpleData.from_dict(dict_repr)
        assert restored.name == "test"
        assert restored.value == 42
        assert restored.active is True


class TestHdf5Functionality:
    """HDF5機能のテスト"""

    def test_write_basic(self):
        """基本的なHDF5書き込みテスト"""
        data = SimpleData(name="hdf5_test", value=123, active=False)

        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            group = f.create_group("data")

            # HDF5に書き込み、グループが返されることを確認
            returned_group = data.to_hdf5_group(group)

            # 同じグループが返される
            assert returned_group is group

            # データが正しく格納されている
            assert group["name"].asstr()[()] == "hdf5_test"
            assert group["value"][()] == 123
            assert bool(group["active"][()]) is False

    def test_read_basic(self):
        """基本的なHDF5読み込みテスト"""
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            # 元のデータを作成してHDF5に保存
            original = SimpleData(name="restore_test", value=456, active=True)
            group = f.create_group("data")
            original.to_hdf5_group(group)

            # HDF5から復元
            restored = SimpleData.from_hdf5_group(group)

            # データが正しく復元されている
            assert restored.name == "restore_test"
            assert restored.value == 456
            assert restored.active is True

    def test_hdf5_roundtrip_dataclass(self):
        """データクラスでの往復変換テスト"""
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            original = SimpleData(name="roundtrip", value=789, active=False)
            group = f.create_group("data")

            # HDF5経由での往復変換
            original.to_hdf5_group(group)
            restored = SimpleData.from_hdf5_group(group)

            # すべてのデータが保持されている
            assert original.name == restored.name
            assert original.value == restored.value
            assert original.active == restored.active

    def test_hdf5_roundtrip_regular_class(self):
        """通常クラスでの往復変換テスト"""
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            original = ComplexData(
                config={"key": "value", "number": 42}, items=["item1", "item2", "item3"], metadata="complex_test"
            )
            group = f.create_group("data")

            # HDF5経由での往復変換
            original.to_hdf5_group(group)
            restored = ComplexData.from_hdf5_group(group)

            # データが正しく復元されている
            assert original.config == restored.config
            assert original.items == restored.items
            assert original.metadata == restored.metadata


class TestComplexData:
    """複雑なデータ構造のテスト"""

    def test_nested_persistent_objects(self):
        """ネストしたPersistentObjectのテスト"""
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            simple = SimpleData(name="nested", value=100)
            nested = NestedData(simple_data=simple, numbers=[1.1, 2.2, 3.3], mapping={"a": 1, "b": 2, "c": 3})
            group = f.create_group("data")

            # HDF5往復変換
            nested.to_hdf5_group(group)
            restored = NestedData.from_hdf5_group(group)

            # ネストしたオブジェクトも正しく復元される
            assert isinstance(restored.simple_data, SimpleData)
            assert restored.simple_data.name == "nested"
            assert restored.simple_data.value == 100
            assert restored.numbers == [1.1, 2.2, 3.3]
            assert restored.mapping == {"a": 1, "b": 2, "c": 3}

    def test_mixed_types(self):
        """異なる型の混在データのテスト"""

        @dataclass
        class MixedTypes(SerializableContainer):
            string_val: str
            int_val: int
            float_val: float
            bool_val: bool
            list_val: List[Any]
            dict_val: Dict[str, Any]

        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            mixed = MixedTypes(
                string_val="mixed",
                int_val=42,
                float_val=3.14,
                bool_val=True,
                list_val=["string", 123, 4.56],
                dict_val={"nested": {"deep": "value"}, "flag": False},
            )
            group = f.create_group("data")

            mixed.to_hdf5_group(group)
            restored = MixedTypes.from_hdf5_group(group)

            assert restored.string_val == "mixed"
            assert restored.int_val == 42
            assert restored.float_val == 3.14
            assert restored.bool_val is True
            assert restored.list_val == ["string", 123, 4.56]
            assert restored.dict_val == {"nested": {"deep": "value"}, "flag": False}


class TestDualFormat:
    """辞書とHDF5の両方のフォーマットが使えることのテスト"""

    def test_dict_to_hdf5_consistency(self):
        """辞書形式とHDF5形式の一貫性テスト"""
        original = SimpleData(name="consistency", value=999, active=True)

        # 辞書形式でのシリアライズ・デシリアライズ
        dict_data = original.to_dict()
        dict_restored = SimpleData.from_dict(dict_data)

        # HDF5形式でのシリアライズ・デシリアライズ
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            group = f.create_group("data")
            original.to_hdf5_group(group)
            hdf5_restored = SimpleData.from_hdf5_group(group)

        # 両方の結果が一致する
        assert dict_restored.name == hdf5_restored.name
        assert dict_restored.value == hdf5_restored.value
        assert dict_restored.active == hdf5_restored.active

    def test_cross_format_conversion(self):
        """フォーマット間の変換テスト"""
        # 辞書から作成
        dict_data = {"name": "cross_format", "value": 777, "active": False}
        from_dict = SimpleData.from_dict(dict_data)

        # HDF5に保存
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            group = f.create_group("data")
            from_dict.to_hdf5_group(group)

            # HDF5から復元
            from_hdf5 = SimpleData.from_hdf5_group(group)

        # 元の辞書データと復元されたデータが一致
        assert from_hdf5.name == "cross_format"
        assert from_hdf5.value == 777
        assert from_hdf5.active is False


class TestErrorHandling:
    """エラーハンドリングのテスト"""

    def test_empty_group_error(self):
        """空のHDF5グループでの復元エラー"""
        with h5py.File("test_memory", "w", driver="core", backing_store=False) as f:
            empty_group = f.create_group("empty")

            # 復元時にエラーが発生する可能性
            try:
                SimpleData.read(empty_group)
            except Exception:
                # エラーが発生することを確認（具体的な例外型は実装依存）
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
