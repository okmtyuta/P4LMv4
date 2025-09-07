"""
先頭/中央/末尾の3区間平均を連結する非学習 Aggregator。
"""

import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class EndsSegmentMeanAggregator(DataProcess):
    """先頭N・中央・末尾M の平均を連結して返す非学習 Aggregator。"""

    def __init__(self, head_len: int, tail_len: int) -> None:
        if head_len < 0 or tail_len < 0:
            raise ValueError("head_len and tail_len must be non-negative")
        self._head = int(head_len)
        self._tail = int(tail_len)

    @property
    def dim_factor(self) -> int:  # 出力は [head, center, tail] の連結で 3D
        return 3

    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        """安全な平均（長さ0ならゼロベクトル）。"""
        # x: (len, D) or (0, D)
        if x.numel() == 0:
            # 形だけ整えるゼロベクトルを返す
            return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        return torch.mean(x, dim=0)

    def _act(self, protein: Protein) -> Protein:
        """3 区間の平均を計算して (3D,) に連結。"""
        data = protein.get_processed()  # (L, D)
        L, D = data.shape

        head_len = min(self._head, L)
        rem = L - head_len
        tail_len = min(self._tail, rem)
        center_len = rem - tail_len

        head = data[0:head_len]
        center = data[head_len : head_len + center_len]
        tail = data[head_len + center_len : L]

        mh = self._safe_mean(head)  # (D,)
        mc = self._safe_mean(center)  # (D,)
        mt = self._safe_mean(tail)  # (D,)

        out = torch.cat([mh, mc, mt], dim=0)  # (3D,)
        return protein.set_processed(processed=out)
