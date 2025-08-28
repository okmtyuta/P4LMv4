import torch

from src.modules.data_process.data_process import DataProcess
from src.modules.protein.protein import Protein


class EndsSegmentMeanAggregator(DataProcess):
    """先頭N・中央・末尾M の平均を連結して返す非学習Aggregator。

    - 入力: processed (L, D)
    - 出力: (3D,) = [mean_head, mean_center, mean_tail] の連結
    - パラメータ: head_len, tail_len は固定整数（学習なし）

    仕様:
    - L が短い場合でも安全に動作するよう、区間は重複しないように切り出す。
      具体的には、先に先頭 head_len を確保し、残りから末尾 tail_len を確保する。
      残差が中央区間となる（長さ0の区間はゼロベクトル平均とみなす）。
    """

    def __init__(self, head_len: int, tail_len: int) -> None:
        if head_len < 0 or tail_len < 0:
            raise ValueError("head_len and tail_len must be non-negative")
        self._head = int(head_len)
        self._tail = int(tail_len)

    @property
    def dim_factor(self) -> int:  # 出力は [head, center, tail] の連結で 3D
        return 3

    def _safe_mean(self, x: torch.Tensor) -> torch.Tensor:
        # x: (len, D) or (0, D)
        if x.numel() == 0:
            # 形だけ整えるゼロベクトルを返す
            return torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        return torch.mean(x, dim=0)

    def _act(self, protein: Protein) -> Protein:
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
