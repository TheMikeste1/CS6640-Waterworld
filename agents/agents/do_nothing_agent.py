from typing import Any

import torch

from ..agents import AbstractAgent


class DoNothingAgent(AbstractAgent):
    def __call__(self, obs) -> (torch.Tensor, Any):
        return (
            torch.zeros(
                self.env.action_space(self.env_name).shape, dtype=torch.float32
            ),
            None,
        )

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
