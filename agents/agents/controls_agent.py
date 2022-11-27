from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from ..agents import AbstractAgent

if TYPE_CHECKING:
    import pettingzoo as pz


class ControlsAgent(AbstractAgent):
    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        name: str,
        device: torch.device | str = None,
    ):
        super().__init__(env, env_name, name)

        self.num_sensors: int = env.unwrapped.env.n_sensors
        self.speed_features: bool = env.unwrapped.env.speed_features

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)

    def __call__(self, obs) -> (torch.Tensor, Any):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # If obs is not in batch form, add a batch dimension
        while len(obs.shape) < 3:
            obs = obs.unsqueeze(-2)
        obs = obs.to(self.device)
        actions, *_ = torch.nn.Module.__call__(self, obs)

        return actions, None

    def forward(self, obs):
        # Check if food is in view
        target_obs = obs[..., 2 * self.num_sensors : 3 * self.num_sensors]
        if target_obs.sum() == self.num_sensors:
            # Don't move if there is no food unless there is poison
            start_i = 3 * self.num_sensors + (
                self.num_sensors if self.speed_features else 0
            )
            # Check if poison is in view
            target_obs = obs[..., start_i : start_i + self.num_sensors]
            if target_obs.sum() == self.num_sensors:
                # There's nothing, don't move
                return (
                    torch.zeros((*obs.shape[:-1], self.out_features)).to(self.device),
                    None,
                )
            # No food, but poison, move away from it
            min_index = target_obs.argmin(dim=-1)
            percent_of_circle = -min_index / self.num_sensors
            radians = percent_of_circle * 2 * np.pi
            x = torch.cos(radians)
            y = torch.sin(radians)
            actions = torch.stack([x, y], dim=-1)
            return actions, None

        min_index = target_obs.argmin(dim=-1)
        percent_of_circle = min_index / self.num_sensors
        radians = percent_of_circle * 2 * np.pi
        x = torch.cos(radians)
        y = torch.sin(radians)
        actions = torch.stack([x, y], dim=-1)
        return actions

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
