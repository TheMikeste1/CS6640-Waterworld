import copy
from typing import Iterable

import torch

from agents import NeuralNetwork


class CriticNetwork(NeuralNetwork):
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
        x = torch.cat([obs, action], dim=-1)
        x = self.layers(x)
        return x

    def clone(self):
        layers = list(copy.deepcopy(self.layers))
        obs_layers = list(copy.deepcopy(self.obs_layers))
        action_layers = list(copy.deepcopy(self.action_layers))
        return CriticNetwork(layers, obs_layers, action_layers)
