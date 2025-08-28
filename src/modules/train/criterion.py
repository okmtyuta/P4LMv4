from typing import Literal, TypedDict

import numpy as np
import torch
from scipy import stats

CriteriaName = Literal["root_mean_squared_error", "mean_squared_error", "mean_absolute_error", "pearsonr"]
criteria_names: list[CriteriaName] = [
    "root_mean_squared_error",
    "mean_squared_error",
    "mean_absolute_error",
    "pearsonr",
]


class Criteria(TypedDict):
    root_mean_squared_error: float
    mean_squared_error: float
    mean_absolute_error: float
    pearsonr: float


class Criterion:
    _mse_loss = torch.nn.MSELoss()
    _l1_loss = torch.nn.L1Loss()

    @classmethod
    def root_mean_squared_error(cls, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = cls.mean_squared_error(output=output, label=label).sqrt()
        return loss

    @classmethod
    def mean_squared_error(cls, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = cls._mse_loss(output, label)
        return loss

    @classmethod
    def mean_absolute_error(cls, output: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss: torch.Tensor = cls._l1_loss(output, label)
        return loss

    @classmethod
    def pearsonr(cls, output: torch.Tensor, label: torch.Tensor) -> np.float32:
        correlation = stats.pearsonr(output.detach(), label.detach()).correlation
        return correlation

    @classmethod
    def call(cls, output: torch.Tensor, label: torch.Tensor) -> Criteria:
        root_mean_squared_error = cls.root_mean_squared_error(output=output, label=label)
        mean_squared_error = cls.mean_squared_error(output=output, label=label)
        mean_absolute_error = cls.mean_absolute_error(output=output, label=label)
        pearsonr = cls.pearsonr(output=output, label=label)

        return {
            "root_mean_squared_error": root_mean_squared_error.item(),
            "mean_squared_error": mean_squared_error.item(),
            "mean_absolute_error": mean_absolute_error.item(),
            "pearsonr": pearsonr.item(),
        }

    # def delta(cls, output: torch.Tensor, label: torch.Tensor, alpha: float = 0.95):
    #     output_mean = torch.mean(output)
    #     label_mean = torch.mean(label)
    #     output_var = torch.var(output)
    #     label_var = torch.var(label)

    #     diff = output_mean - label_mean
    #     upper = diff + 1.96 * torch.sqrt((output_var / len(output)) + (label_var / len(label)))
    #     lower = diff - 1.96 * torch.sqrt((output_var / len(output)) + (label_var / len(label)))

    #     print(lower, upper)
