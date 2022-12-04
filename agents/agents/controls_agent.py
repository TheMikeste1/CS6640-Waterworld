from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from ..agents import AbstractAgent

if TYPE_CHECKING:
    import pettingzoo as pz


class ControlsAgent(AbstractAgent):
    @dataclass(kw_only=True)
    class Builder(AbstractAgent.Builder):
        def build(self, env: pz.AECEnv) -> ControlsAgent:
            return ControlsAgent(env, self.env_name, self.name)

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
        food_obs = obs[..., 2 * self.num_sensors : 3 * self.num_sensors]
        # The location of poison observations depends on if speed features are used
        start_i = 3 * self.num_sensors + (
            self.num_sensors if self.speed_features else 0
        )
        # Check if poison is in view
        poison_obs = obs[..., start_i : start_i + self.num_sensors]
        rows_no_food = torch.all(food_obs == 1.0, dim=-1)
        rows_no_poison = torch.all(poison_obs == 1.0, dim=-1)
        rows_only_poison = rows_no_food & ~rows_no_poison
        rows_nothing = torch.logical_and(rows_no_food, rows_no_poison)

        # Target food if there's food, otherwise target poison
        min_index = food_obs.argmin(dim=-1)
        min_index[rows_only_poison] = poison_obs[rows_only_poison].argmin(dim=-1)

        percent_of_circle = min_index / self.num_sensors
        # If we're targeting poison, we want to turn the opposite direction
        percent_of_circle[rows_only_poison] += 0.5

        radians = percent_of_circle * 2 * np.pi
        x = torch.cos(radians)
        y = torch.sin(radians)
        actions = torch.stack([x, y], dim=-1)
        actions[rows_nothing] = torch.tensor([0.0, 0.0], device=self.device)
        return actions

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
