import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class RoPEAnglesCache:
    """Cache for (cos, sin) angle tensors keyed by (length, dim, reversed)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, bool], tuple[torch.Tensor, torch.Tensor]] = {}

    def read(self, length: int, dim: int, reversed: bool) -> tuple[torch.Tensor, torch.Tensor] | None:
        return self._cache.get((length, dim, reversed))

    def set(
        self,
        length: int,
        dim: int,
        reversed: bool,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> None:
        self._cache[(length, dim, reversed)] = (cos, sin)


class _BaseRoPEPositionalEncoder(DataProcess):
    """Rotary Positional Embedding (RoPE) applied to representations (L, D).

    - D の偶数ペアごとに 2 次元平面回転を適用します。
    - 奇数次元の場合、最後の 1 チャンネルはそのまま維持します。
    - 位置は 1 始まり（1..L）。`reversed=True` の場合は L..1 の順序。
    """

    def __init__(self, theta_base: float) -> None:
        self._theta_base = float(theta_base)
        self._cache = RoPEAnglesCache()

    def _positions(self, length: int, reversed: bool) -> torch.Tensor:
        if reversed:
            return torch.arange(length, 0, -1, dtype=torch.float32)
        return torch.arange(1, length + 1, dtype=torch.float32)

    def _cos_sin(self, length: int, dim: int, reversed: bool) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._cache.read(length=length, dim=dim, reversed=reversed)
        if cached is not None:
            return cached

        L, D = length, dim
        half = D // 2
        if half == 0:
            # D == 0 or 1 の場合にも対応
            cos = torch.ones((L, 0), dtype=torch.float32)
            sin = torch.zeros((L, 0), dtype=torch.float32)
            self._cache.set(length, dim, reversed, cos, sin)
            return cos, sin

        # 位置 (1..L) または (L..1)
        positions = self._positions(length=L, reversed=reversed)  # (L,)
        p_col = positions[:, None]  # (L, 1)

        # 周波数（ペア単位）。inv_freq = base^{-2m/D}
        m = torch.arange(0, half, dtype=torch.float32)  # (half,)
        inv_freq = self._theta_base ** (-2.0 * m / float(D))  # (half,)

        angles = p_col * inv_freq[None, :]  # (L, half)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self._cache.set(length, dim, reversed, cos, sin)
        return cos, sin

    def _apply_rope(self, reps: torch.Tensor, reversed: bool) -> torch.Tensor:
        """Apply RoPE rotation to representations.

        Args:
            reps: (L, D) tensor
            reversed: whether to use reversed positions

        Returns:
            Rotated tensor with the same shape as reps.
        """
        L, D = reps.shape
        cos, sin = self._cos_sin(length=L, dim=D, reversed=reversed)  # (L, half)
        half = D // 2
        if half == 0:
            return reps

        x_even = reps[:, 0 : 2 * half : 2]
        x_odd = reps[:, 1 : 2 * half : 2]

        # 回転: [x_even, x_odd] -> [x_even*cos - x_odd*sin, x_even*sin + x_odd*cos]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos

        out = reps.clone()
        out[:, 0 : 2 * half : 2] = rot_even
        out[:, 1 : 2 * half : 2] = rot_odd
        # 奇数次元の最終チャネルはそのまま保持
        return out


class RoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_representations()
        out = self._apply_rope(reps, False)
        return protein.set_processed(processed=out)


class ReversedRoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_representations()
        out = self._apply_rope(reps, True)
        return protein.set_processed(processed=out)


class BidirectionalRoPEPositionalEncoder(_BaseRoPEPositionalEncoder):
    def _act(self, protein: Protein) -> Protein:
        reps = protein.get_representations()
        out_normal = self._apply_rope(reps, False)
        out_reversed = self._apply_rope(reps, True)
        out = torch.cat([out_normal, out_reversed], dim=1)
        return protein.set_processed(processed=out)
