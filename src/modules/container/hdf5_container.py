#!/usr/bin/env python3
"""
Python の辞書/リスト/数値/文字列/bytes/torch.Tensor を汎用的に HDF5 へ保存・復元するユーティリティ。

- dict はグループ、配列/テンソルはデータセットとして格納する。
- 文字列、bytes、None、空リストなどの表現に属性マーカーを用いる。
- 本クラスは中間表現として dict を受け取り/返す。
"""

from typing import Any, Mapping, Self, Sequence

import h5py
import numpy as np
import torch

from src.modules.container.hdf5_io import HDF5IO


class Hdf5Container(HDF5IO):
    """h5py.Group との相互変換を提供する辞書コンテナ。

    使用する属性マーカー:
    - `__TYPE__`: "list"（リスト格納を示す）
    - `__LENGTH__`: 0（空リスト）
    - `__NONE__`: True（None プレースホルダ）
    - `__BYTES__`: True（bytes を uint8 配列として保存）
    - `__ELEM_TYPE__`: "bytes_uint8"（リスト要素が bytes であることを示す）
    """

    STR_DTYPE = h5py.string_dtype(encoding="utf-8")

    def __init__(self, source: Mapping[str, Any]) -> None:
        """中間表現として保持/書き込み対象の dict を受け取る。"""
        # Take a shallow copy to snapshot current content
        self._source: Mapping[str, Any] = dict(source)

    # ------------ Private Methods ------------
    def _ensure_new_slot(self, g: h5py.Group, key: str) -> None:
        """キーに `/` を含まないことを検査し、新規スロット作成の前提を整える。"""
        if "/" in key:
            raise ValueError(f"HDF5 key cannot contain '/': {key!r}")

    @staticmethod
    def _is_number_like(x: Any) -> bool:
        """数値/真偽値/NumPy のスカラに類するかを判定する。"""
        return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_))

    @classmethod
    def _all_number_like(cls, seq: Sequence[Any]) -> bool:
        """列の全要素が数値様であるか。"""
        return all(cls._is_number_like(x) for x in seq)

    @staticmethod
    def _all_str(seq: Sequence[Any]) -> bool:
        """列の全要素が文字列か。"""
        return all(isinstance(x, str) for x in seq)

    @staticmethod
    def _all_bytes(seq: Sequence[Any]) -> bool:
        """列の全要素が bytes/bytearray/np.void か。"""
        return all(isinstance(x, (bytes, bytearray, np.void)) for x in seq)

    def _put(self, g: h5py.Group, key: str, value: Any) -> None:
        """与えられた Python 値を適切な HDF5 表現にエンコードして格納する。"""
        # dict
        if isinstance(value, Mapping):
            self._ensure_new_slot(g, key)
            sub = g.require_group(key)
            self._write_into(sub, value)
            return

        # torch.Tensor
        if isinstance(value, torch.Tensor):
            """torch.Tensor は ndarray に変換し、dtype 情報を属性へ格納。"""
            self._ensure_new_slot(g, key)
            np_val = value.detach().cpu().numpy()
            ds = g.create_dataset(key, data=np_val)
            ds.attrs["__TORCH__"] = True
            ds.attrs["__TORCH_DTYPE__"] = str(value.dtype)
            return

        # numpy array
        if isinstance(value, np.ndarray):
            self._ensure_new_slot(g, key)
            g.create_dataset(key, data=value)
            return

        # list / tuple
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                self._ensure_new_slot(g, key)
                sub = g.require_group(key)
                sub.attrs["__TYPE__"] = "list"
                sub.attrs["__LENGTH__"] = 0
            elif self._all_number_like(value):
                self._ensure_new_slot(g, key)
                g.create_dataset(key, data=np.asarray(value))
            elif self._all_str(value):
                self._ensure_new_slot(g, key)
                arr = np.asarray(value, dtype=object)
                g.create_dataset(key, data=arr, dtype=self.STR_DTYPE)
            elif self._all_bytes(value):
                self._ensure_new_slot(g, key)
                sub = g.require_group(key)
                sub.attrs["__TYPE__"] = "list"
                for i, b in enumerate(value):
                    bi = bytes(b)
                    ds = sub.create_dataset(str(i), data=np.frombuffer(bi, dtype=np.uint8))
                    ds.attrs["__ELEM_TYPE__"] = "bytes_uint8"
            else:
                self._ensure_new_slot(g, key)
                sub = g.require_group(key)
                sub.attrs["__TYPE__"] = "list"
                for i, item in enumerate(value):
                    self._put(sub, str(i), item)
            return

        # str
        if isinstance(value, str):
            self._ensure_new_slot(g, key)
            g.create_dataset(key, data=value, dtype=self.STR_DTYPE)
            return

        # bytes / bytearray
        if isinstance(value, (bytes, bytearray)):
            self._ensure_new_slot(g, key)
            ds = g.create_dataset(key, data=np.frombuffer(bytes(value), dtype=np.uint8))
            ds.attrs["__BYTES__"] = True
            return

        # numbers / bools
        if self._is_number_like(value):
            self._ensure_new_slot(g, key)
            g.create_dataset(key, data=value)
            return

        # None
        if value is None:
            self._ensure_new_slot(g, key)
            sub = g.require_group(key)
            sub.attrs["__NONE__"] = True
            return

        raise TypeError(f"Unsupported type for key {key!r}: {type(value)}")

    def _write_into(self, group: h5py.Group, source: Mapping[str, Any]) -> None:
        """辞書の各キー/値を HDF5 へ再帰的に書き込む。"""
        for k, v in source.items():
            self._put(group, str(k), v)

    @staticmethod
    def _dataset_to_python(ds: h5py.Dataset) -> Any:
        """データセットを Python 値へ復号する。"""
        # Strings (variable-length UTF-8)
        if h5py.check_string_dtype(ds.dtype) is not None:
            data = ds.asstr()[...]
            if ds.shape == ():
                return data.item()
            return data.tolist()

        # bytes
        if bool(ds.attrs.get("__BYTES__", False)) or ds.attrs.get("__ELEM_TYPE__") == "bytes_uint8":
            arr = np.asarray(ds[...], dtype=np.uint8)
            return bytes(arr.tobytes())

        # torch tensor marker
        if bool(ds.attrs.get("__TORCH__", False)):
            arr = ds[...]
            t = torch.from_numpy(arr)
            return t

        # scalar numbers / bools
        if ds.shape == ():
            val = ds[()]
            if isinstance(val, np.generic):
                return val.item()
            return val

        # other arrays stay as numpy.ndarray
        return ds[...]

    @classmethod
    def _group_to_python(cls, g: h5py.Group) -> Any:
        """グループを再帰的に辿って Python 値へ復号する。"""
        # None marker
        if bool(g.attrs.get("__NONE__", False)):
            return None

        # list marker
        if g.attrs.get("__TYPE__") == "list":
            if int(g.attrs.get("__LENGTH__", -1)) == 0 and len(g) == 0:
                return []
            seq = []
            for k in sorted(g.keys(), key=lambda s: int(s) if s.isdigit() else s):
                obj = g[k]
                if isinstance(obj, h5py.Dataset):
                    seq.append(cls._dataset_to_python(obj))
                else:
                    seq.append(cls._group_to_python(obj))
            return seq

        # dict
        out: dict[str, Any] = {}
        for k, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                out[k] = cls._dataset_to_python(obj)
            else:
                out[k] = cls._group_to_python(obj)
        return out

    # ------------ Public Methods ------------
    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """保持している辞書ソースを与えられたグループへ再帰的に書き込む。"""
        self._write_into(group, self._source)
        return group

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """グループから読み取り、新しいインスタンスを返す。"""
        source = cls._group_to_python(group)
        if not isinstance(source, dict):
            raise TypeError("Top-level object must be a dict to initialize Hdf5Container")
        return cls(source)

    def save_as_hdf5(self, file_path: str) -> None:
        """このインスタンスを HDF5 ファイルへ保存する。"""
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """HDF5 ファイルから読み込み、新しいインスタンスを返す。"""
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
