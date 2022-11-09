from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class NeuralNetwork(torch.nn.Module):
    __slots__ = (
        "criterion",
        "layers",
        "lr_scheduler",
        "optimizer",
    )

    def __init__(
        self,
        layers: [torch.nn.Module],
    ):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        return self.layers[-1].out_features
