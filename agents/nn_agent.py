from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import numpy as np
    import pettingzoo as pz


class NNAgent(AbstractAgent, torch.nn.Module):
    __slots__ = ("model", "optimizer", "loss", "lr_scheduler", "memory", "device")

    def __init__(
        self,
        env: pz.AECEnv,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.nn.Module,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        auto_select_device: bool = True,
        memory: Memory = None,
    ):
        AbstractAgent.__init__(self, env)
        torch.nn.Module.__init__(self)

        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.lr_scheduler = lr_scheduler

        self.memory = memory if memory is not None else Memory(1024)

        self.device = torch.device(
            "cuda" if auto_select_device and torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def __call__(self, name, obs) -> np.ndarray:
        self.model.eval()
        obs = torch.tensor(obs, device=self.device)
        out = torch.nn.Module.__call__(self, obs)
        out = out.detach().cpu().numpy()
        return out

    def forward(self, x):
        return self.model(x)

    def update(self, batch_size: int = 1) -> float:
        self.model.train()
        state, action, reward, new_state, terminated = self.memory.sample(batch_size)
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        # action = torch.tensor(action, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, device=self.device).unsqueeze(0)
        new_state = torch.tensor(new_state, device=self.device).unsqueeze(0)
        # terminated = torch.tensor(terminated, device=self.device).unsqueeze(0)

        old_targets = self.model(state)
        # TODO: Add get_old_targets and get_new_targets to model
        new_targets = reward + self.model(new_state)

        loss = self.loss(old_targets, new_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.item()
