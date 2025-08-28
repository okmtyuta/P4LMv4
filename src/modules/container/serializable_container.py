from __future__ import annotations

from typing import Self

import h5py

from src.modules.container.hdf5_container import Hdf5Container
from src.modules.container.hdf5_io import HDF5IO
from src.modules.container.serializable_object import SerializableObject


class SerializableContainer(SerializableObject, HDF5IO):
    """A serializable object that can be persisted to both dictionary and HDF5 formats.

    This class combines the dictionary serialization capabilities of SerializableObject
    with HDF5 storage functionality, providing flexible data persistence options.
    """

    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """Write this object to an HDF5 group.

        Args:
            group: HDF5 group to write data into.

        Returns:
            The HDF5 group that was written to.
        """
        # Convert to dictionary first, then to HDF5
        data_dict = self.to_dict()
        storage = Hdf5Container(data_dict)
        return storage.to_hdf5_group(group)

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Create an instance from HDF5 group data.

        Args:
            group: HDF5 group containing the object data.

        Returns:
            New instance of this class restored from HDF5 data.
        """
        # Read from HDF5, then convert from dictionary
        storage = Hdf5Container.from_hdf5_group(group)
        return cls.from_dict(dict(storage._source))

    def save_as_hdf5(self, file_path: str) -> None:
        """Save this object to an HDF5 file.

        Args:
            file_path: Path to the HDF5 file to create.
        """
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """Load an object from an HDF5 file.

        Args:
            file_path: Path to the HDF5 file to read from.

        Returns:
            New instance of this class loaded from the file.
        """
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
