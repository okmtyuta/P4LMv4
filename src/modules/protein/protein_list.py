from typing import Iterable

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

    def __init__(self, proteins: Iterable[Protein]) -> None:
        """Initialize ProteinList with an iterable of Protein objects.

        Args:
            proteins: Iterable of Protein objects
        """
        super().__init__(proteins)

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

            # Create Protein directly
            protein = Protein(key=key, props=props, representations=None, processed=None, predicted={})
            proteins.append(protein)

        return cls(proteins)
