from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class LearnableAbsolutePositionalAdder(DataProcess):
    """Add a learnable absolute positional bias per position (scalar per position).

    For a sequence of length L and representations reps (L, D), this adds a scalar
    bias b[p] to every channel at position p (0-indexed internally).

    - Parameters are defined up to `max_length`. If L > max_length, the last bias
      value (b[max_length-1]) is reused for positions beyond max_length.
    - Initialized with zeros so the initial transform is identity.
    """

    def __init__(self, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._max_len = int(max_length)
        self._bias = nn.Parameter(torch.zeros(self._max_len, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        return [self._bias]

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        idx = torch.arange(L, device=device)
        idx = torch.clamp(idx, max=self._max_len - 1)

        b = self._bias.to(device=device, dtype=dtype)  # (max_len,)
        pos_bias = b[idx]  # (L,)

        out = reps + pos_bias[:, None]
        return protein.set_processed(processed=out)


class LearnableAbsolutePositionalScaler(DataProcess):
    """Apply a learnable absolute positional scale per position (scalar per position).

    For a sequence of length L and representations reps (L, D), this multiplies
    reps[p, :] by a gate g[p] = 1 + s[p], where s[p] is a learnable scalar. Using
    zero initialization yields identity at start of training.

    - Parameters are defined up to `max_length`. If L > max_length, the last scale
      value is reused for positions beyond max_length.
    """

    def __init__(self, max_length: int) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._max_len = int(max_length)
        self._scale = nn.Parameter(torch.zeros(self._max_len, dtype=torch.float32))

    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        return [self._scale]

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_processed()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        idx = torch.arange(L, device=device)
        idx = torch.clamp(idx, max=self._max_len - 1)

        s = self._scale.to(device=device, dtype=dtype)  # (max_len,)
        gate = 1.0 + s[idx]  # (L,)

        out = reps * gate[:, None]
        return protein.set_processed(processed=out)
