from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor


class NeuralNetwork(torch.nn.Module):
    @dataclass(kw_only=True)
    class Builder:
        layers: [torch.nn.Module] | torch.nn.Module

        def build(self) -> NeuralNetwork:
            return NeuralNetwork(self.layers)

    __slots__ = (
        "criterion",
        "layers",
        "lr_scheduler",
        "optimizer",
    )

    def __init__(
        self,
        layers: [torch.nn.Module] | torch.nn.Module,
    ):
        super().__init__()
        if isinstance(layers, torch.nn.Module):
            layers = [layers]

        self.layers = torch.nn.Sequential(*layers)
        self.__num_inputs = 0

        for i in range(0, len(self.layers)):
            layer = layers[i]
            if hasattr(layer, "in_features"):
                self.__num_inputs = layer.in_features
                break
            elif hasattr(layer, "in_channels"):
                self.__num_inputs = layer.in_channels
                break
        else:
            raise ValueError("No layer with in_features or in_channels found")

        self.__num_outputs = 0
        for i in range(1, len(self.layers) + 1):
            layer = layers[-i]
            if hasattr(layer, "out_features"):
                self.__num_outputs = layer.out_features
                break
            elif hasattr(layer, "out_channels"):
                self.__num_outputs = layer.out_channels
                break
        else:
            raise ValueError("No layer with out_features or out_channels found")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    @property
    def in_features(self) -> int:
        return self.__num_inputs

    @property
    def out_features(self) -> int:
        return self.__num_outputs

    def __iadd__(self, other: torch.nn.Module):
        self.layers.append(other)
        return self

    def __getitem__(self, item):
        return self.layers[item]
