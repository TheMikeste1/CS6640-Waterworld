from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import torch

from agents.step_data import StepData

if TYPE_CHECKING:
    import numpy as np
    import pettingzoo as pz


class AbstractAgent(ABC, torch.nn.Module):
    __slots__ = ("env", "name")

    def __init__(self, env: pz.AECEnv, name: str):
        torch.nn.Module.__init__(self)
        self.env = env
        self.name = name

    @abstractmethod
    def __call__(self, obs) -> (np.ndarray, Any):
        pass

    def post_step(self, data: StepData):
        pass

    def post_episode(self):
        pass

    def on_train(self) -> float:
        return 0.0
