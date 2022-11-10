from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch

from agents.step_data import StepData

if TYPE_CHECKING:
    import numpy as np
    import pettingzoo as pz


class AbstractAgent(ABC, torch.nn.Module):
    __slots__ = ("env", "env_name", "name")

    def __init__(self, env: pz.AECEnv, env_name: str, name: str):
        torch.nn.Module.__init__(self)
        self.env = env
        self.env_name = env_name
        self.name = name
        self.device = torch.device("cpu")

    @abstractmethod
    def __call__(self, obs) -> (np.ndarray, Any):
        pass

    def post_step(self, data: StepData):
        pass

    def post_episode(self):
        pass

    def on_train(self) -> float:
        return 0.0

    @property
    def in_features(self):
        raise NotImplementedError

    @property
    def out_features(self):
        raise NotImplementedError

    def update(self, batch_size: int):
        pass

    def apply_loss(self, old_policy_targets, new_policy_targets):
        pass

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)
