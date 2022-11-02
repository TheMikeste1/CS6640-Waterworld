from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from agents import AbstractAgent

if TYPE_CHECKING:
    import numpy as np
    import pettingzoo as pz


class NNAgent(AbstractAgent, torch.nn.Module):
    __slots__ = ("model", "device")

    def __init__(
        self, env: pz.AECEnv, model: torch.nn.Module, auto_select_device: bool = True
    ):
        AbstractAgent.__init__(self, env)
        torch.nn.Module.__init__(self)

        self.model = model
        self.device = torch.device(
            "cuda" if auto_select_device and torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def __call__(self, name, obs) -> np.ndarray:
        obs = torch.tensor(obs, device=self.device)
        out = torch.nn.Module.__call__(self, obs)
        out = out.detach().cpu().numpy()
        return out

    def forward(self, x):
        return self.model(x)
