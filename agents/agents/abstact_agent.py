from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    import pettingzoo as pz

    from agents import StepData


class AbstractAgent(ABC, torch.nn.Module):
    @dataclass(kw_only=True)
    class Builder(ABC):
        env_name: str
        name: str

        @abstractmethod
        def build(self, env: pz.AECEnv) -> AbstractAgent:
            pass

    __slots__ = ("env", "env_name", "name")

    def __init__(self, env: pz.AECEnv, env_name: str, name: str):
        torch.nn.Module.__init__(self)
        self.env = env
        self.env_name = env_name
        self.name = name
        self.device = torch.device("cpu")
        self.should_explore = True

    @abstractmethod
    def __call__(self, obs) -> (torch.Tensor, Any):
        pass

    def post_step(self, data: StepData):
        pass

    def post_episode(self):
        pass

    def on_train(self) -> float:
        return dict()

    @property
    def in_features(self):
        raise NotImplementedError

    @property
    def out_features(self):
        raise NotImplementedError

    def update(self, batch_size: int):
        pass

    def apply_loss(self, *args, **kwargs):
        pass

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def reset(self):
        pass


AgentBuilder = AbstractAgent.Builder
