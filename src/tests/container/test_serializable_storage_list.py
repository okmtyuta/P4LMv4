#!/usr/bin/env python3
"""
SerializableStorageList の機能テスト
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import pytest

from src.modules.container.hdf5_io import HDF5IO
from src.modules.container.serializable_storage import SerializableStorage
from src.modules.container.serializable_storage_list import SerializableStorageList


@dataclass
class SimpleTestData(SerializableStorage):
    """テスト用の単純なデータクラス"""

    name: str
    value: int


@dataclass
class ComplexTestData(SerializableStorage):
    """テスト用の複雑なデータクラス"""

    title: str
    numbers: list[int]
    metadata: dict[str, str]


class TestBasicFunctionality:
    """基本機能テスト"""

    def test_inheritance(self):
        """継承関係の確認"""
        data1 = SimpleTestData("test1", 42)
        data2 = SimpleTestData("test2", 84)
        storage_list = SerializableStorageList([data1, data2])

        assert isinstance(storage_list, HDF5IO)
        assert len(storage_list) == 2
        assert storage_list[0].name == "test1"
        assert storage_list[1].value == 84

    def test_empty_list(self):
        """空リストのテスト"""
        empty_list = SerializableStorageList([])
        assert len(empty_list) == 0
        assert empty_list.is_empty


class TestHDF5Functionality:
    """HDF5機能テスト"""

    def test_to_hdf5_group_basic(self):
        """基本的なHDF5書き込み"""
        data1 = SimpleTestData("item1", 100)
        data2 = SimpleTestData("item2", 200)
        storage_list = SerializableStorageList([data1, data2])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                returned_group = storage_list.to_hdf5_group(f)
                assert returned_group is f

                # メタデータの確認
                assert f.attrs["__TYPE__"] == "SerializableStorageList"
                assert f.attrs["__LENGTH__"] == 2

                # 要素の存在確認
                assert "0" in f
                assert "1" in f

                # 要素のクラス情報確認
                assert "__CLASS__" in f["0"].attrs
                assert "__CLASS__" in f["1"].attrs
        finally:
            tmp_path.unlink()

    def test_from_hdf5_group_basic(self):
        """基本的なHDF5読み込み"""
        original_data = [SimpleTestData("original1", 300), SimpleTestData("original2", 400)]
        original_list = SerializableStorageList(original_data)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                original_list.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_list = SerializableStorageList.from_hdf5_group(f)

            # 検証
            assert len(restored_list) == 2
            assert restored_list[0].name == "original1"
            assert restored_list[0].value == 300
            assert restored_list[1].name == "original2"
            assert restored_list[1].value == 400
        finally:
            tmp_path.unlink()

    def test_roundtrip_complex_data(self):
        """複雑なデータのラウンドトリップテスト"""
        original_data = [
            ComplexTestData("first", [1, 2, 3], {"key1": "value1"}),
            ComplexTestData("second", [4, 5, 6], {"key2": "value2", "key3": "value3"}),
        ]
        original_list = SerializableStorageList(original_data)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                original_list.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_list = SerializableStorageList.from_hdf5_group(f)

            # 検証
            assert len(restored_list) == 2

            # 最初の要素
            assert restored_list[0].title == "first"
            assert restored_list[0].numbers == [1, 2, 3]
            assert restored_list[0].metadata == {"key1": "value1"}

            # 2番目の要素
            assert restored_list[1].title == "second"
            assert restored_list[1].numbers == [4, 5, 6]
            assert restored_list[1].metadata == {"key2": "value2", "key3": "value3"}
        finally:
            tmp_path.unlink()

    def test_mixed_types_list(self):
        """異なる型の混合リストテスト"""
        mixed_data = [SimpleTestData("simple", 100), ComplexTestData("complex", [7, 8, 9], {"mixed": "true"})]
        mixed_list = SerializableStorageList(mixed_data)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                mixed_list.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_list = SerializableStorageList.from_hdf5_group(f)

            # 検証
            assert len(restored_list) == 2
            assert isinstance(restored_list[0], SimpleTestData)
            assert isinstance(restored_list[1], ComplexTestData)

            assert restored_list[0].name == "simple"
            assert restored_list[0].value == 100

            assert restored_list[1].title == "complex"
            assert restored_list[1].numbers == [7, 8, 9]
        finally:
            tmp_path.unlink()


class TestFileMethods:
    """ファイルメソッドのテスト"""

    def test_save_and_load_from_hdf5(self):
        """save_as_hdf5とload_from_hdf5のテスト"""
        original_data = [SimpleTestData("file_test1", 500), SimpleTestData("file_test2", 600)]
        original_list = SerializableStorageList(original_data)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # ファイルに保存
            original_list.save_as_hdf5(str(tmp_path))

            # ファイルから読み込み
            loaded_list = SerializableStorageList.load_from_hdf5(str(tmp_path))

            # 検証
            assert len(loaded_list) == 2
            assert loaded_list[0].name == "file_test1"
            assert loaded_list[0].value == 500
            assert loaded_list[1].name == "file_test2"
            assert loaded_list[1].value == 600
        finally:
            tmp_path.unlink()


class TestEdgeCases:
    """エッジケースのテスト"""

    def test_empty_list_serialization(self):
        """空リストのシリアライゼーション"""
        empty_list = SerializableStorageList([])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                empty_list.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_list = SerializableStorageList.from_hdf5_group(f)

            # 検証
            assert len(restored_list) == 0
            assert restored_list.is_empty
        finally:
            tmp_path.unlink()

    def test_single_item_list(self):
        """単一要素リストのテスト"""
        single_item = SerializableStorageList([SimpleTestData("single", 777)])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                single_item.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_list = SerializableStorageList.from_hdf5_group(f)

            # 検証
            assert len(restored_list) == 1
            assert restored_list[0].name == "single"
            assert restored_list[0].value == 777
        finally:
            tmp_path.unlink()


class TestErrorHandling:
    """エラーハンドリングテスト"""

    def test_non_hdf5io_element_error(self):
        """HDF5IOを実装していない要素のエラー"""

        class NonHDF5IOClass:
            def __init__(self, value):
                self.value = value

        invalid_list = SerializableStorageList([NonHDF5IOClass("invalid")])

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                with pytest.raises(TypeError, match="does not implement HDF5IO interface"):
                    invalid_list.to_hdf5_group(f)
        finally:
            tmp_path.unlink()

    def test_wrong_group_type_error(self):
        """間違ったグループタイプのエラー"""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 間違ったタイプのデータを作成
            with h5py.File(tmp_path, "w") as f:
                f.attrs["__TYPE__"] = "WrongType"
                f.attrs["__LENGTH__"] = 0

            # 読み込み時にエラーが発生することを確認
            with h5py.File(tmp_path, "r") as f:
                with pytest.raises(TypeError, match="does not contain SerializableStorageList data"):
                    SerializableStorageList.from_hdf5_group(f)
        finally:
            tmp_path.unlink()
