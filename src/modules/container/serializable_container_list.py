from typing import Self

"""
HDF5 へ保存・復元できるシーケンスコンテナ。
要素は HDF5IO インタフェースを実装している必要があります。
"""

import h5py

from src.modules.container.hdf5_io import HDF5IO
from src.modules.container.sequence_container import SequenceContainer


class SerializableContainerList[T](SequenceContainer[T], HDF5IO):
    """HDF5 シリアライズに対応したシーケンスコンテナ。"""

    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """このリストを HDF5 グループへ書き込む。"""
        # Store metadata
        group.attrs["__TYPE__"] = "SerializableContainerList"
        group.attrs["__LENGTH__"] = len(self._data)

        # Store each element
        for i, item in enumerate(self._data):
            if not isinstance(item, HDF5IO):
                raise TypeError(f"Element at index {i} does not implement HDF5IO interface")

            # Create subgroup for each element
            item_group = group.create_group(str(i))
            # Store the class name for deserialization
            item_group.attrs["__CLASS__"] = f"{item.__class__.__module__}.{item.__class__.__name__}"
            # Let the item serialize itself
            item.to_hdf5_group(item_group)

        return group

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """HDF5 グループからリストを復元する。"""
        # Check if this is the correct type
        if group.attrs.get("__TYPE__") != "SerializableContainerList":
            raise TypeError("Group does not contain SerializableContainerList data")

        length = int(group.attrs.get("__LENGTH__", 0))
        items = []

        # Restore each element
        for i in range(length):
            item_group = group[str(i)]

            # Get the class information
            class_name = (
                item_group.attrs["__CLASS__"].decode()
                if isinstance(item_group.attrs["__CLASS__"], bytes)
                else item_group.attrs["__CLASS__"]
            )

            # Import and get the class
            module_name, class_name = class_name.rsplit(".", 1)
            module = __import__(module_name, fromlist=[class_name])
            item_class = getattr(module, class_name)

            # Restore the item using its from_hdf5_group method
            if not issubclass(item_class, HDF5IO):
                raise TypeError(f"Class {item_class} does not implement HDF5IO interface")

            item = item_class.from_hdf5_group(item_group)
            items.append(item)

        return cls(items)

    def save_as_hdf5(self, file_path: str) -> None:
        """このリストを HDF5 ファイルへ保存する。"""
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """HDF5 ファイルから読み込み、新しいインスタンスを返す。"""
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
