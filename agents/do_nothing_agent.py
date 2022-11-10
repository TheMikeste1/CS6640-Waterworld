from typing import Any

import numpy as np

from agents import AbstractAgent


class DoNothingAgent(AbstractAgent):
    def __call__(self, obs) -> (np.ndarray, Any):
        return (
            np.zeros(self.env.action_space(self.env_name).shape, dtype=np.float32),
            None,
        )

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
