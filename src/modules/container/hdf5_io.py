#!/usr/bin/env python3
"""
HDF5 との入出力インタフェース定義。
"""
from abc import ABC, abstractmethod
from typing import Self

import h5py


class HDF5IO(ABC):
    """HDF5ファイルの読み書きを行うための抽象クラス。

    このクラスを継承するクラスは、HDF5グループとの間でデータの
    シリアライゼーション/デシリアライゼーションを行う機能を提供する。
    """

    @abstractmethod
    def to_hdf5_group(self, group: h5py.Group) -> h5py.Group:
        """オブジェクトのデータをHDF5グループに保存する。

        Args:
            group: データを保存するHDF5グループ
        """
        pass

    @classmethod
    @abstractmethod
    def from_hdf5_group(cls, group: h5py.Group) -> Self:
        """HDF5グループからデータを読み込んでオブジェクトを初期化する。

        Args:
            group: データを読み込むHDF5グループ
        """
        pass

    @abstractmethod
    def save_as_hdf5(self, file_path: str) -> None:
        """オブジェクトをHDF5ファイルとして保存する。

        Args:
            file_path: 保存先のファイルパス
        """
        pass

    @classmethod
    @abstractmethod
    def load_from_hdf5(cls, file_path: str) -> Self:
        """HDF5ファイルからデータを読み込んでオブジェクトを初期化する。

        Args:
            file_path: 読み込むファイルのパス
        """
        pass
