import math
from typing import List

import torch
from torch import nn

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class LearnableFourierPositionalEncoder(DataProcess):
    """Learnable Fourier positional gating.

    For a sequence of length L with representations reps (L, D), this process builds
    K learnable Fourier bases with frequencies w_k and phases φ_k, computes
    Φ(p) = [sin(p·w_k + φ_k), cos(p·w_k + φ_k)]_{k=1..K} ∈ R^{2K} per position p (1..L),
    then a scalar gating g(p) = 1 + s · Φ(p) · v, where v ∈ R^{2K} is learnable and s is
    a fixed scale. The final output is out[p, :] = reps[p, :] * g(p).

    Args:
        num_bases: K, number of Fourier bases.
        min_period: Minimum period (>0). Shortest wavelength ≈ min_period.
        max_period: Maximum period (>min_period). Longest wavelength ≈ max_period.
        projection_scale: s, scale factor for the projection output.
    """

    def __init__(self, num_bases: int, min_period: float, max_period: float, projection_scale: float) -> None:
        if num_bases <= 0:
            raise ValueError("num_bases must be positive")
        if min_period <= 0 or max_period <= 0:
            raise ValueError("periods must be positive")
        if not (min_period < max_period):
            raise ValueError("min_period must be < max_period")

        self._K = int(num_bases)
        self._scale = float(projection_scale)

        # Initialize frequencies using log-spaced periods between [min_period, max_period]
        periods = torch.logspace(
            start=math.log10(min_period), end=math.log10(max_period), steps=self._K, dtype=torch.float32
        )
        init_w = (2.0 * math.pi) / periods  # angular frequencies
        init_log_w = torch.log(init_w)  # store in log-space for stability/positivity

        # Learnable parameters
        self._log_w = nn.Parameter(init_log_w)  # (K,)
        self._phase = nn.Parameter(torch.zeros(self._K, dtype=torch.float32))  # (K,)
        self._proj = nn.Parameter(torch.randn(2 * self._K, dtype=torch.float32) * 0.01)  # (2K,)

    # Optional helper to expose parameters for optimizers if統合する場合
    def parameters(self) -> List[nn.Parameter]:  # type: ignore[override]
        return [self._log_w, self._phase, self._proj]

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_representations()  # (L, D)
        L, _D = reps.shape

        device = reps.device
        dtype = reps.dtype if reps.dtype.is_floating_point else torch.float32

        # Positions 1..L
        p = torch.arange(1, L + 1, dtype=dtype, device=device)  # (L,)

        # Frequencies and phases (ensure on same device/dtype)
        w = torch.exp(self._log_w).to(device=device, dtype=dtype)  # (K,)
        phi = self._phase.to(device=device, dtype=dtype)  # (K,)

        # Features Φ(p): (L, 2K)
        angles = p[:, None] * w[None, :] + phi[None, :]  # (L, K)
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (L, 2K)

        # Scalar gate per position: g(p) = 1 + s * Φ(p) · v
        v = self._proj.to(device=device, dtype=dtype)  # (2K,)
        gate = 1.0 + self._scale * (feats @ v)  # (L,)

        out = reps * gate[:, None]
        return protein.set_processed(processed=out)
