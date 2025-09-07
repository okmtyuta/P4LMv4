"""
モデルの抽象基底。`torch.nn.Module` を継承する薄いラッパです。
"""

from abc import ABC, abstractmethod

import torch


class Model(torch.nn.Module, ABC):
    """共通インタフェースを提供する抽象モデル基底クラス。"""

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        """`forward` を呼び出すエイリアス。"""
        return super().__call__(input)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を実装する。"""
        raise NotImplementedError
