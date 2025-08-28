import random
import secrets
from typing import Iterable, Optional, Self

import polars as pl

from src.modules.container.serializable_container_list import SerializableContainerList
from src.modules.protein.protein import Protein
from src.modules.protein.protein_types import ProteinProps


class ProteinList(SerializableContainerList[Protein]):
    """Sequence container for handling collections of Protein objects.

    This class extends SerializableContainerList to provide specialized functionality
    for working with protein data, including CSV loading capabilities and HDF5 I/O.
    Inherits all list-like operations from SequenceContainer and HDF5 I/O from HDF5IO.
    """

    def __init__(self, iterable: Iterable[Protein]) -> None:
        """Initialize ProteinList with an iterable of Protein objects.

        Args:
            iterable: Iterable of Protein objects
        """
        super().__init__(iterable)

    @classmethod
    def from_csv(cls, path: str) -> "ProteinList":
        """Load ProteinList from CSV file using Polars.

        The 'index' column will be used as the key, and all other columns
        including 'seq' will be stored in props for each protein.

        Args:
            path: Path to the CSV file

        Returns:
            ProteinList containing Protein instances from CSV data
        """
        df = pl.read_csv(path)

        proteins = []
        for row in df.iter_rows(named=True):
            # Use index as key
            key = str(row["index"])

            # All columns including seq go to props (except index)
            props: ProteinProps = {k: v for k, v in row.items() if k != "index"}

            # Create Protein Directly
            protein = Protein(key=key, props=props, representations=None, processed=None, predicted={})
            proteins.append(protein)

        return cls(proteins)

    def shuffle(self, seed: Optional[int] = None) -> Self:
        """要素順を破壊的にシャッフルする。

        - インプレースで順序を変更します。
        - `seed` が `None` の場合は内部でランダムに決定します（`secrets.randbits(64)`）。

        Args:
            seed: 乱数シード。`None` 可（未指定時は内部で自動決定）。

        Returns:
            self（メソッドチェーン可能）。
        """
        if seed is None:
            seed = secrets.randbits(64)
        rng = random.Random(seed)
        rng.shuffle(self._data)

        print("====")
        print("====")
        print("seed is", seed)
        print("====")
        print("====")

        return self
