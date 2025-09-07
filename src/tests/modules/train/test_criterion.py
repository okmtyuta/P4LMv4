#!/usr/bin/env python3
import numpy as np
import torch

from src.modules.train.criterion import Criterion


def test_criterion_basic_relations():
    # y = 2x -> 完全相関
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
    y = 2.0 * x

    mse = Criterion.mean_squared_error(output=x, label=y)
    rmse = Criterion.root_mean_squared_error(output=x, label=y)
    mae = Criterion.mean_absolute_error(output=x, label=y)
    pr = Criterion.pearsonr(output=x, label=y)

    # 相関は1
    assert np.isclose(float(pr), 1.0, atol=1e-6)
    # MSE, MAE は非負
    assert mse.item() >= 0
    assert mae.item() >= 0
    # RMSE^2 ≈ MSE
    assert np.isclose(rmse.item() ** 2, mse.item(), rtol=1e-6)
