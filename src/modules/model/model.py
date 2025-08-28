from abc import ABC, abstractmethod

import torch


class Model(torch.nn.Module, ABC):
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return super().__call__(input)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
