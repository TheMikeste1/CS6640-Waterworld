from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Callable, Iterable, Union

import torch

from agents import ModuleBuilder, NeuralNetwork


class CriticNetwork(NeuralNetwork):
    @dataclass(kw_only=True)
    class Builder(NeuralNetwork.Builder):
        obs_layers: Union[
            Iterable[ModuleBuilder | Callable[[], torch.nn.Module]],
            ModuleBuilder,
            Callable[[], torch.nn.Module],
        ]
        action_layers: Union[
            Iterable[ModuleBuilder | Callable[[], torch.nn.Module]],
            ModuleBuilder,
            Callable[[], torch.nn.Module],
        ]

        def build(self) -> CriticNetwork:
            if isinstance(self.layers, Iterable):
                layers = [layer() for layer in self.layers]
            else:
                layers = self.layers()
            if isinstance(self.obs_layers, Iterable):
                obs_layers = [layer() for layer in self.obs_layers]
            else:
                obs_layers = self.obs_layers()
            if isinstance(self.action_layers, Iterable):
                action_layers = [layer() for layer in self.action_layers]
            else:
                action_layers = self.action_layers()
            return CriticNetwork(layers, obs_layers, action_layers)

    def __init__(self, layers: Iterable[torch.nn.Module] | torch.nn.Module,
                 obs_layers: Iterable[torch.nn.Module] | torch.nn.Module,
                 action_layers: Iterable[torch.nn.Module] | torch.nn.Module):
        super().__init__(layers)
        if isinstance(obs_layers, torch.nn.Module):
            obs_layers = [obs_layers]
        self.obs_layers = torch.nn.Sequential(*obs_layers)
        if isinstance(action_layers, torch.nn.Module):
            action_layers = [action_layers]
        self.action_layers = torch.nn.Sequential(*action_layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self.obs_layers(obs)
        action = self.action_layers(action)
        x = obs + action
        x = self.layers(x)
        return x

    def clone(self):
        layers = list(copy.deepcopy(self.layers))
        obs_layers = list(copy.deepcopy(self.obs_layers))
        action_layers = list(copy.deepcopy(self.action_layers))
        return CriticNetwork(layers, obs_layers, action_layers)
