from typing import Literal, TypedDict

import torch

ESMModelName = Literal["esm2", "esm1b"]


class ESMModelResult(TypedDict):
    logits: torch.Tensor
    representations: dict[int, torch.Tensor]
    attentions: torch.Tensor
    contacts: torch.Tensor
