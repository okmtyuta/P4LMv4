import math

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class LogSumExpAggregator(DataProcess):
    """温度付き LogSumExp による非学習集約（長さ不変型）。

    入力: processed (L, D)
    出力: (D,)

    定義: y = τ · [logsumexp(x/τ, dim=0) - log(L)]
    - τ が大きいほど平均に近く、小さいほど最大に近い。
    - `- log(L)` により長さに対してスケール不変（平均に整合）。
    """

    def __init__(self, tau: float) -> None:
        if tau <= 0:
            raise ValueError("tau must be positive")
        self._tau = float(tau)

    @property
    def dim_factor(self) -> int:
        return 1

    def _act(self, protein: Protein) -> Protein:
        x = protein.get_processed()  # (L, D)
        if x.ndim != 2:
            raise ValueError("processed must be a 2D tensor (L, D)")
        L, _D = x.shape
        if L == 0:
            # 空列への安全対策（ゼロを返す）。通常は起こらない想定。
            return protein.set_processed(processed=torch.zeros(_D, dtype=x.dtype, device=x.device))

        tau = torch.as_tensor(self._tau, dtype=x.dtype, device=x.device)
        z = torch.logsumexp(x / tau, dim=0)  # (D,)
        y = tau * (z - math.log(L))
        return protein.set_processed(processed=y)
