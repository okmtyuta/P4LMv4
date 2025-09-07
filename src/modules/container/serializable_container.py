from __future__ import annotations

"""
辞書/HDF5 の両方へ保存・復元できるシリアライズ可能コンテナ。
"""

from typing import Self

import h5py

from src.modules.container.hdf5_container import Hdf5Container
from src.modules.container.hdf5_io import HDF5IO
from src.modules.container.serializable_object import SerializableObject


class SerializableContainer(SerializableObject, HDF5IO):
    """辞書と HDF5 の双方に対応するコンテナ基底。"""

    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """自身を HDF5 グループへ書き込む。"""
        # Convert to dictionary first, then to HDF5
        data_dict = self.to_dict()
        storage = Hdf5Container(data_dict)
        return storage.to_hdf5_group(group)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """HDF5 グループからインスタンスを復元する。"""
        # Read from HDF5, then convert from dictionary
        storage = Hdf5Container.from_hdf5_group(group)
        return cls.from_dict(dict(storage._source))

    def save_as_hdf5(self, file_path: str) -> None:
        """このオブジェクトを HDF5 ファイルへ保存する。"""
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """HDF5 ファイルから読み込み、インスタンスを生成する。"""
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
