from typing import Optional, Union

import torch

from src.modules.container.serializable_storage import SerializableStorage
from src.modules.protein.protein_types import ProteinProps


class Protein(SerializableStorage):
    """Protein class that handles protein data with sequence, properties, and representations.

    This class manages protein information including:
    - Sequence data identifier (key)
    - Properties like retention time, length etc. (props)
    - Optional tensor representations for ML models
    - Optional processed tensor data (representations with positional encoding and aggregation)
    - Optional predicted properties

    It inherits from SerializableStorage to support both dictionary and HDF5 serialization.
    """

    def __init__(
        self,
        key: str,
        props: ProteinProps,
        representations: Optional[torch.Tensor] = None,
        processed: Optional[torch.Tensor] = None,
        predicted: Optional[ProteinProps] = None,
    ) -> None:
        """Initialize Protein with key, properties and optional tensor data.

        Args:
            key: The protein identifier
            props: Dictionary of protein properties
            representations: Optional tensor representations for ML models
            processed: Optional processed tensor data
            predicted: Optional predicted properties dictionary
        """
        self.key = key
        self.props = props
        self.representations = representations
        self.processed = processed
        self.predicted = predicted if predicted is not None else {}

    @property
    def seq(self) -> str:
        value = self.props.get("seq")
        if not isinstance(value, str):
            raise TypeError("Protein.props['seq'] must be a str")
        return value

    def read_props(self, name: str) -> Union[str, int, float]:
        """Read a specific property by name.

        Args:
            name: The property name to read

        Returns:
            The property value

        Raises:
            RuntimeError: If the property doesn't exist or is None
        """
        if name not in self.props:
            raise RuntimeError(f"Prop {name} is not readable")

        prop = self.props[name]
        if prop is None:
            raise RuntimeError(f"Prop {name} is not readable")

        return prop

    def set_props(self, props: ProteinProps) -> "Protein":
        """Set the protein properties.

        Args:
            props: Dictionary of properties to set

        Returns:
            Self for method chaining
        """
        self.props = props
        return self

    def set_representations(self, representations: torch.Tensor) -> "Protein":
        """Set the tensor representations.

        Args:
            representations: The representations tensor to set

        Returns:
            Self for method chaining
        """
        self.representations = representations
        return self

    def set_processed(self, processed: torch.Tensor) -> "Protein":
        """Set the processed tensor data.

        Args:
            processed: The processed tensor to set

        Returns:
            Self for method chaining
        """
        self.processed = processed
        return self

    def set_predicted(self, predicted: ProteinProps) -> "Protein":
        """Set the predicted properties.

        Args:
            predicted: Dictionary of predicted properties

        Returns:
            Self for method chaining
        """
        self.predicted = predicted
        return self

    def get_representations(self) -> torch.Tensor:
        """Get the tensor representations.

        Returns:
            torch.Tensor: The representations tensor

        Raises:
            RuntimeError: If representations are None
        """
        if self.representations is None:
            raise RuntimeError("Protein representations unavailable")

        return self.representations

    def get_processed(self) -> torch.Tensor:
        """Get the processed tensor data (representations with positional encoding and aggregation).

        Returns:
            torch.Tensor: The processed tensor

        Raises:
            RuntimeError: If processed data is None
        """
        if self.processed is None:
            raise RuntimeError("Protein processed unavailable")

        return self.processed
