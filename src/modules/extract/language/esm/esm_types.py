"""
ESM で使用する型定義。
"""

from typing import Literal, TypedDict

import torch

ESMModelName = Literal["esm2", "esm1b"]


class ESMModelResult(TypedDict):
    """ESM の forward 出力（最小限）。"""

    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor
