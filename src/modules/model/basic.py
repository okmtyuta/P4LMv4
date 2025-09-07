"""
全結合の浅い回帰モデル（デモ用途）。
"""

import torch

from src.modules.model.model import Model


class BasicModel(Model):
    """入力次元を受け取り、MLPで1出力を予測する基本モデル。"""

    def __init__(self, input_dim: int) -> None:
        """層構成を初期化する。"""
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        """順伝播を実装する。"""
        return self.layers(x)
