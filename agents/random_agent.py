from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

from agents import AbstractAgent

if TYPE_CHECKING:
    import pettingzoo as pz


class RandomAgent(AbstractAgent):
    @dataclass(kw_only=True)
    class Builder(AbstractAgent.Builder):
        def build(self, env: pz.AECEnv) -> RandomAgent:
            return RandomAgent(env, self.env_name, self.name)

    def __call__(self, obs) -> (torch.Tensor, Any):
        return torch.from_numpy(self.env.action_space(self.env_name).sample()), None

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
