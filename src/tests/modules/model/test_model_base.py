#!/usr/bin/env python3
import torch

from src.modules.model.model import Model


class IdentityModel(Model):
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x


def test_model_call_invokes_forward():
    m = IdentityModel()
    x = torch.randn(2, 3)
    y = m(x)
    assert torch.allclose(x, y)
