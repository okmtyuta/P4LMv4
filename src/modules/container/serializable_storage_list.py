from typing import Self

import h5py

from src.modules.container.hdf5_io import HDF5IO
from src.modules.container.sequence_container import SequenceContainer


class SerializableStorageList[T](SequenceContainer[T], HDF5IO):
    """A sequence container that can be serialized to HDF5 format.

    This class assumes that all elements implement the HDF5IO interface.
    """

    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """Write this list to an HDF5 group.

        Args:
            group: HDF5 group to write data into.

        Returns:
            The HDF5 group that was written to.
        """
        # Store metadata
        group.attrs["__TYPE__"] = "SerializableStorageList"
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
        """Create a list from HDF5 group data.

        Args:
            group: HDF5 group containing the list data.

        Returns:
            New instance of this class restored from HDF5 data.
        """
        # Check if this is the correct type
        if group.attrs.get("__TYPE__") != "SerializableStorageList":
            raise TypeError("Group does not contain SerializableStorageList data")

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
        """Save this list to an HDF5 file.

        Args:
            file_path: Path to the HDF5 file to create.
        """
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """Load a list from an HDF5 file.

        Args:
            file_path: Path to the HDF5 file to read from.

        Returns:
            New instance of this class loaded from the file.
        """
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
