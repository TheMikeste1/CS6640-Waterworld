from __future__ import annotations
from typing import Any, TYPE_CHECKING

import torch

from agents import AbstractAgent


class RandomAgent(AbstractAgent):
    def __call__(self, obs) -> (torch.Tensor, Any):
        return torch.from_numpy(self.env.action_space(self.env_name).sample()), None

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
