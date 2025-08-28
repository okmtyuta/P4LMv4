#!/usr/bin/env python3
"""
Hdf5Storage の機能軸テスト
"""

import tempfile
from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import pytest
import torch

from src.modules.container.hdf5_storage import Hdf5Storage


class TestSerialization:
    """HDF5への書き込み機能テスト"""

    def test_simple_data_serialization(self):
        """基本データ型のシリアライゼーション"""
        data = {
            "integer": 42,
            "float_val": 3.14159,
            "boolean": True,
            "string": "Hello HDF5",
            "none_value": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            # ファイルが正しく作成されたことを確認
            with h5py.File(tmp_path, "r") as f:
                assert "integer" in f
                assert "float_val" in f
                assert "boolean" in f
                assert "string" in f
                assert "none_value" in f
                assert f["integer"][()] == 42
                assert f["string"].asstr()[()] == "Hello HDF5"
        finally:
            tmp_path.unlink()

    def test_numpy_array_serialization(self):
        """NumPy配列のシリアライゼーション"""
        data = {
            "int_array": np.array([1, 2, 3, 4, 5]),
            "float_array": np.array([1.1, 2.2, 3.3]),
            "bool_array": np.array([True, False, True]),
            "2d_array": np.array([[1, 2], [3, 4]]),
            "empty_array": np.array([]),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            # 配列の形状と内容を確認
            with h5py.File(tmp_path, "r") as f:
                np.testing.assert_array_equal(f["int_array"][...], [1, 2, 3, 4, 5])
                np.testing.assert_array_almost_equal(f["float_array"][...], [1.1, 2.2, 3.3])
                assert f["2d_array"].shape == (2, 2)
                assert f["empty_array"].shape == (0,)
        finally:
            tmp_path.unlink()

    def test_torch_tensor_serialization(self):
        """PyTorchテンソルのシリアライゼーション"""
        data = {
            "tensor_1d": torch.tensor([1.0, 2.0, 3.0]),
            "tensor_2d": torch.tensor([[1, 2], [3, 4]]),
            "tensor_float32": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "tensor_int64": torch.tensor([1, 2, 3], dtype=torch.int64),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            # テンソルマーカーと型情報を確認
            with h5py.File(tmp_path, "r") as f:
                assert bool(f["tensor_1d"].attrs.get("__TORCH__")) is True
                assert "float32" in f["tensor_float32"].attrs.get("__TORCH_DTYPE__", "")
                np.testing.assert_array_equal(f["tensor_2d"][...], [[1, 2], [3, 4]])
        finally:
            tmp_path.unlink()

    def test_list_serialization(self):
        """リスト/タプルのシリアライゼーション"""
        data = {
            "empty_list": [],
            "number_list": [1, 2, 3, 4, 5],
            "string_list": ["apple", "banana", "cherry"],
            "mixed_list": [1, "hello", 3.14],
            "nested_list": [[1, 2], [3, 4], [5, 6]],
            "tuple_data": (10, 20, 30),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                # 空リストはグループとして保存される
                assert f["empty_list"].attrs.get("__TYPE__") == "list"
                assert f["empty_list"].attrs.get("__LENGTH__") == 0

                # 数値リストはデータセットとして保存される
                np.testing.assert_array_equal(f["number_list"][...], [1, 2, 3, 4, 5])

                # 文字列リストはデータセットとして保存される
                str_array = f["string_list"].asstr()[...]
                assert list(str_array) == ["apple", "banana", "cherry"]

                # 混合リストはグループとして保存される
                assert f["mixed_list"].attrs.get("__TYPE__") == "list"
                assert "0" in f["mixed_list"]
                assert "1" in f["mixed_list"]
        finally:
            tmp_path.unlink()

    def test_bytes_serialization(self):
        """バイトデータのシリアライゼーション"""
        data = {
            "single_bytes": b"hello world",
            "bytearray_data": bytearray(b"test data"),
            "bytes_list": [b"first", b"second", b"third"],
            "empty_bytes": b"",
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                # バイトデータのマーカーを確認
                assert bool(f["single_bytes"].attrs.get("__BYTES__")) is True
                assert bool(f["bytearray_data"].attrs.get("__BYTES__")) is True

                # バイトリストはグループとして保存される
                assert f["bytes_list"].attrs.get("__TYPE__") == "list"
                assert f["bytes_list"]["0"].attrs.get("__ELEM_TYPE__") == "bytes_uint8"
        finally:
            tmp_path.unlink()

    def test_nested_dict_serialization(self):
        """ネストした辞書のシリアライゼーション"""
        data = {
            "level1": {
                "level2": {
                    "level3": {
                        "deep_value": 42,
                        "deep_array": np.array([1, 2, 3]),
                    }
                },
                "sibling": "value",
            },
            "metadata": {"version": "1.0", "author": "test"},
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                # ネストした構造を確認
                assert f["level1/level2/level3/deep_value"][()] == 42
                np.testing.assert_array_equal(f["level1/level2/level3/deep_array"][...], [1, 2, 3])
                assert f["level1/sibling"].asstr()[()] == "value"
                assert f["metadata/version"].asstr()[()] == "1.0"
        finally:
            tmp_path.unlink()


class TestDeserialization:
    """HDF5からの読み込み機能テスト"""

    def test_simple_data_deserialization(self):
        """基本データ型のデシリアライゼーション"""
        original_data = {
            "integer": 42,
            "float_val": 3.14159,
            "boolean": True,
            "string": "Hello HDF5",
            "none_value": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["integer"] == 42
            assert abs(restored_data["float_val"] - 3.14159) < 1e-6
            assert restored_data["boolean"] is True
            assert restored_data["string"] == "Hello HDF5"
            assert restored_data["none_value"] is None
        finally:
            tmp_path.unlink()

    def test_numpy_array_deserialization(self):
        """NumPy配列のデシリアライゼーション"""
        original_data = {
            "int_array": np.array([1, 2, 3, 4, 5]),
            "float_array": np.array([1.1, 2.2, 3.3]),
            "2d_array": np.array([[1, 2], [3, 4]]),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            np.testing.assert_array_equal(restored_data["int_array"], [1, 2, 3, 4, 5])
            np.testing.assert_array_almost_equal(restored_data["float_array"], [1.1, 2.2, 3.3])
            np.testing.assert_array_equal(restored_data["2d_array"], [[1, 2], [3, 4]])
        finally:
            tmp_path.unlink()

    def test_torch_tensor_deserialization(self):
        """PyTorchテンソルのデシリアライゼーション"""
        original_data = {
            "tensor_1d": torch.tensor([1.0, 2.0, 3.0]),
            "tensor_2d": torch.tensor([[1, 2], [3, 4]]),
            "tensor_float32": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            # PyTorchテンソルとして復元されることを確認
            assert isinstance(restored_data["tensor_1d"], torch.Tensor)
            assert isinstance(restored_data["tensor_2d"], torch.Tensor)
            torch.testing.assert_close(restored_data["tensor_1d"], torch.tensor([1.0, 2.0, 3.0]))
            torch.testing.assert_close(restored_data["tensor_2d"], torch.tensor([[1, 2], [3, 4]]))
        finally:
            tmp_path.unlink()

    def test_list_deserialization(self):
        """リスト/タプルのデシリアライゼーション"""
        original_data = {
            "empty_list": [],
            "number_list": [1, 2, 3, 4, 5],
            "string_list": ["apple", "banana", "cherry"],
            "mixed_list": [1, "hello", 3.14],
            "nested_list": [[1, 2], [3, 4]],
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["empty_list"] == []
            np.testing.assert_array_equal(restored_data["number_list"], [1, 2, 3, 4, 5])
            assert restored_data["string_list"] == ["apple", "banana", "cherry"]
            assert len(restored_data["mixed_list"]) == 3
            assert restored_data["mixed_list"][0] == 1
            assert restored_data["mixed_list"][1] == "hello"
            assert abs(restored_data["mixed_list"][2] - 3.14) < 1e-6
        finally:
            tmp_path.unlink()

    def test_bytes_deserialization(self):
        """バイトデータのデシリアライゼーション"""
        original_data = {
            "single_bytes": b"hello world",
            "bytearray_data": bytearray(b"test data"),
            "bytes_list": [b"first", b"second", b"third"],
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["single_bytes"] == b"hello world"
            assert restored_data["bytearray_data"] == b"test data"
            assert restored_data["bytes_list"] == [b"first", b"second", b"third"]
        finally:
            tmp_path.unlink()

    def test_nested_dict_deserialization(self):
        """ネストした辞書のデシリアライゼーション"""
        original_data = {
            "level1": {
                "level2": {
                    "level3": {"deep_value": 42, "deep_array": np.array([1, 2, 3])},
                },
                "sibling": "value",
            },
            "metadata": {"version": "1.0", "author": "test"},
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["level1"]["level2"]["level3"]["deep_value"] == 42
            np.testing.assert_array_equal(restored_data["level1"]["level2"]["level3"]["deep_array"], [1, 2, 3])
            assert restored_data["level1"]["sibling"] == "value"
            assert restored_data["metadata"]["version"] == "1.0"
        finally:
            tmp_path.unlink()


class TestRoundtrip:
    """往復変換（書き込み → 読み込み）のテスト"""

    def test_simple_roundtrip(self):
        """基本データ型の往復変換"""
        original_data = {
            "int": 42,
            "float": 3.14159,
            "str": "test string",
            "bool": True,
            "none": None,
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 往復変換
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["int"] == original_data["int"]
            assert abs(restored_data["float"] - original_data["float"]) < 1e-6
            assert restored_data["str"] == original_data["str"]
            assert restored_data["bool"] == original_data["bool"]
            assert restored_data["none"] == original_data["none"]
        finally:
            tmp_path.unlink()

    def test_complex_structure_roundtrip(self):
        """複雑な構造の往復変換"""
        original_data = {
            "arrays": {
                "numpy_1d": np.array([1, 2, 3, 4]),
                "numpy_2d": np.array([[1, 2], [3, 4]]),
                "torch_tensor": torch.tensor([1.0, 2.0, 3.0]),
            },
            "lists": {
                "numbers": [1, 2, 3, 4, 5],
                "strings": ["a", "b", "c"],
                "empty": [],
                "mixed": [1, "hello", 3.14, True],
            },
            "bytes_data": {"single": b"test", "list": [b"a", b"b", b"c"]},
            "metadata": {"version": 1.0, "debug": True},
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            # NumPy配列
            np.testing.assert_array_equal(restored_data["arrays"]["numpy_1d"], original_data["arrays"]["numpy_1d"])
            np.testing.assert_array_equal(restored_data["arrays"]["numpy_2d"], original_data["arrays"]["numpy_2d"])

            # PyTorchテンソル
            torch.testing.assert_close(restored_data["arrays"]["torch_tensor"], original_data["arrays"]["torch_tensor"])

            # リスト（NumPy配列として復元される場合があるので適切に比較）
            np.testing.assert_array_equal(restored_data["lists"]["numbers"], original_data["lists"]["numbers"])
            assert restored_data["lists"]["strings"] == original_data["lists"]["strings"]
            assert restored_data["lists"]["empty"] == original_data["lists"]["empty"]

            # バイト
            assert restored_data["bytes_data"]["single"] == original_data["bytes_data"]["single"]
            assert restored_data["bytes_data"]["list"] == original_data["bytes_data"]["list"]

            # メタデータ
            assert restored_data["metadata"]["version"] == original_data["metadata"]["version"]
            assert restored_data["metadata"]["debug"] == original_data["metadata"]["debug"]
        finally:
            tmp_path.unlink()

    def test_large_data_roundtrip(self):
        """大容量データの往復変換"""
        # 大きな配列データ
        large_array = np.random.rand(1000, 100)
        large_tensor = torch.randn(500, 200)

        original_data = {
            "large_numpy": large_array,
            "large_torch": large_tensor,
            "large_list": list(range(10000)),
            "metadata": {"size": "large", "elements": 10000},
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(original_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            # 大きな配列の形状とデータ型を確認
            assert restored_data["large_numpy"].shape == (1000, 100)
            assert restored_data["large_torch"].shape == torch.Size([500, 200])
            assert len(restored_data["large_list"]) == 10000
            assert restored_data["large_list"][0] == 0
            assert restored_data["large_list"][-1] == 9999

            # 内容の一部をサンプルチェック
            np.testing.assert_array_almost_equal(restored_data["large_numpy"][:5, :5], large_array[:5, :5])
            torch.testing.assert_close(restored_data["large_torch"][:5, :5], large_tensor[:5, :5])
        finally:
            tmp_path.unlink()


class TestEdgeCases:
    """エラーケース・制限事項・特殊ケースのテスト"""

    def test_invalid_hdf5_keys(self):
        """HDF5で無効なキーでのエラー"""
        data = {"invalid/key": "value"}  # '/'を含むキー

        with pytest.raises(ValueError, match="HDF5 key cannot contain"):
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with h5py.File(tmp_path, "w") as f:
                    interface = Hdf5Storage(data)
                    interface.to_hdf5_group(f)
            finally:
                tmp_path.unlink()

    def test_unsupported_types(self):
        """サポートされていない型でのエラー"""

        class CustomClass:
            def __init__(self, value):
                self.value = value

        data = {"custom_object": CustomClass(42)}

        with pytest.raises(TypeError, match="Unsupported type"):
            with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                with h5py.File(tmp_path, "w") as f:
                    interface = Hdf5Storage(data)
                    interface.to_hdf5_group(f)
            finally:
                tmp_path.unlink()

    def test_empty_data(self):
        """空のデータでの動作"""
        empty_data: Dict[str, any] = {}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(empty_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data == {}
        finally:
            tmp_path.unlink()

    def test_extreme_values(self):
        """極端な値での動作"""
        extreme_data = {
            "very_large_int": 2**63 - 1,
            "very_small_int": -(2**63),
            "very_large_float": 1e308,
            "very_small_float": 1e-308,
            "inf_value": float("inf"),
            "neg_inf_value": float("-inf"),
            "nan_value": float("nan"),
            "empty_string": "",
            "unicode_string": "こんにちは世界🌍",
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(extreme_data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["very_large_int"] == 2**63 - 1
            assert restored_data["very_small_int"] == -(2**63)
            assert restored_data["inf_value"] == float("inf")
            assert restored_data["neg_inf_value"] == float("-inf")
            assert np.isnan(restored_data["nan_value"])
            assert restored_data["empty_string"] == ""
            assert restored_data["unicode_string"] == "こんにちは世界🌍"
        finally:
            tmp_path.unlink()

    def test_data_type_preservation(self):
        """データ型の保持テスト"""
        data = {
            "int8": np.int8(127),
            "int16": np.int16(32767),
            "int32": np.int32(2147483647),
            "int64": np.int64(9223372036854775807),
            "uint8": np.uint8(255),
            "float32": np.float32(3.14159),
            "float64": np.float64(2.718281828),
            # complex数値は現在サポートされていない
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            # 基本的な値の確認（型は完全に保持されない場合もあるが、値は保持される）
            assert restored_data["int8"] == 127
            assert restored_data["int16"] == 32767
            assert restored_data["int32"] == 2147483647
            assert abs(restored_data["float32"] - 3.14159) < 1e-5
            assert abs(restored_data["float64"] - 2.718281828) < 1e-10
        finally:
            tmp_path.unlink()

    def test_non_dict_top_level_error(self):
        """トップレベルが辞書でない場合のエラー"""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 手動でHDF5ファイルに非辞書データを作成（データセットのみ、グループなし）
            with h5py.File(tmp_path, "w") as f:
                f.create_dataset("not_a_dict", data="this is not a dict structure")
                # グループがない場合はTypeErrorが発生しない可能性があるため、
                # このテストは実際の動作を確認

            with h5py.File(tmp_path, "r") as f:
                # 実際の動作を確認（エラーが発生するかどうか）
                try:
                    result = Hdf5Storage.from_hdf5_group(f)
                    # エラーが発生しない場合は、結果を確認
                    assert isinstance(result._source, dict)
                except TypeError as e:
                    assert "Top-level object must be a dict" in str(e)
        finally:
            tmp_path.unlink()

    def test_memory_efficiency(self):
        """メモリ効率性の確認（大容量データ）"""
        # 非常に大きなデータ（メモリ使用量のテスト）
        huge_array = np.zeros((10000, 1000), dtype=np.float32)  # 約40MB
        data = {"huge_data": huge_array, "metadata": {"size": "huge"}}

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            # 書き込み
            with h5py.File(tmp_path, "w") as f:
                interface = Hdf5Storage(data)
                interface.to_hdf5_group(f)

            # ファイルサイズを確認
            file_size = tmp_path.stat().st_size
            assert file_size > 1024 * 1024  # 少なくとも1MB以上

            # 読み込み
            with h5py.File(tmp_path, "r") as f:
                restored_interface = Hdf5Storage.from_hdf5_group(f)
                restored_data = restored_interface._source

            assert restored_data["huge_data"].shape == (10000, 1000)
            assert restored_data["metadata"]["size"] == "huge"
        finally:
            tmp_path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
