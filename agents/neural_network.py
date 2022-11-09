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
        self.__num_inputs = (
            self.layers[0].in_features
            if hasattr(self.layers[0], "in_features")
            else self.layers[0].in_channels
        )
        self.__num_outputs = (
            self.layers[-1].out_features
            if hasattr(self.layers[-1], "out_features")
            else self.layers[-1].out_channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    @property
    def in_features(self) -> int:
        return self.__num_inputs

    @property
    def out_features(self) -> int:
        return self.__num_outputs
