#!/usr/bin/env python3
from typing import Any, Mapping, Self, Sequence

import h5py
import numpy as np
import torch

from src.modules.container.hdf5_io import HDF5IO


class Hdf5Storage(HDF5IO):
    """Serialize/deserialize Python dict structures to/from h5py.Group.

    Attributes and markers used:
    - __TYPE__: "list" for list containers
    - __LENGTH__: 0 for empty lists
    - __NONE__: True to mark a None placeholder group
    - __BYTES__: True to mark a bytes dataset
    - __ELEM_TYPE__: "bytes_uint8" to mark list elements stored as uint8
    """

    STR_DTYPE = h5py.string_dtype(encoding="utf-8")

    def __init__(self, source: Mapping[str, Any]) -> None:
        """Initialize with a source dict to be written or held.

        - For writing: call instance method to_hdf5_group(group) to persist this source.
        - For reading: use Hdf5Storage.from_hdf5_group(group) which returns an instance whose
          _source holds the reconstructed dict.
        """
        # Take a shallow copy to snapshot current content
        self._source: Mapping[str, Any] = dict(source)

    # ------------ Private Methods ------------
    def _ensure_new_slot(self, g: h5py.Group, key: str) -> None:
        if "/" in key:
            raise ValueError(f"HDF5 key cannot contain '/': {key!r}")

    @staticmethod
    def _is_number_like(x: Any) -> bool:
        return isinstance(x, (int, float, bool, np.integer, np.floating, np.bool_))

    @classmethod
    def _all_number_like(cls, seq: Sequence[Any]) -> bool:
        return all(cls._is_number_like(x) for x in seq)

    @staticmethod
    def _all_str(seq: Sequence[Any]) -> bool:
        return all(isinstance(x, str) for x in seq)

    @staticmethod
    def _all_bytes(seq: Sequence[Any]) -> bool:
        return all(isinstance(x, (bytes, bytearray, np.void)) for x in seq)

    def _put(self, g: h5py.Group, key: str, value: Any) -> None:
        # dict
        if isinstance(value, Mapping):
            self._ensure_new_slot(g, key)
            sub = g.require_group(key)
            self._write_into(sub, value)
            return

        # torch.Tensor
        if isinstance(value, torch.Tensor):
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
        for k, v in source.items():
            self._put(group, str(k), v)

    @staticmethod
    def _dataset_to_python(ds: h5py.Dataset) -> Any:
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
        """Write the initialized source into the given group recursively and return the group."""
        self._write_into(group, self._source)
        return group

    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """Read from group and return a new instance with the loaded data."""
        source = cls._group_to_python(group)
        if not isinstance(source, dict):
            raise TypeError("Top-level object must be a dict to initialize Hdf5Storage")
        return cls(source)

    def save_as_hdf5(self, file_path: str) -> None:
        """Save this instance to an HDF5 file."""
        with h5py.File(file_path, "w") as f:
            self.to_hdf5_group(f)

    @classmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """Load data from an HDF5 file and return a new instance."""
        with h5py.File(file_path, "r") as f:
            return cls.from_hdf5_group(f)
