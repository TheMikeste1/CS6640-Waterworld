from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import pettingzoo as pz

# TODO: Refactor to pass in name of agent
class QNNAgent(AbstractAgent, torch.nn.Module):
    __slots__ = (
        "value_model",
        "policy_models",
        "optimizer",
        "criterion",
        "lr_scheduler",
        "memory",
        "device",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        value_model: torch.nn.Module,
        policy_models: [torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        auto_select_device: bool = True,
        memory: Memory = None,
    ):
        AbstractAgent.__init__(self, env)
        torch.nn.Module.__init__(self)

        # Assert the value_model inputs are the same as the observation space
        assert (
            value_model.in_features
            == env.observation_space(env.possible_agents[0]).shape[0]
        ), "The value_model input size must match the observation space size"

        # Assert there are the same number of policy_models as actions
        assert (
            len(policy_models) == env.action_space(env.possible_agents[0]).shape[0]
        ), "There must be a policy_model for each action"

        # Assert the policy_models inputs are the same as value_model's output
        for policy_model in policy_models:
            assert (
                policy_model.in_features == value_model.out_features
            ), "The policy_model input size must match the value_model output size"

        self.value_model = value_model
        self.policy_models = torch.nn.ModuleList(policy_models)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler

        self.memory = memory if memory is not None else Memory(1024)

        self.device = torch.device(
            "cuda" if auto_select_device and torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def __call__(self, name, obs) -> np.ndarray:
        self.eval()
        obs = torch.tensor(obs, device=self.device)
        out = torch.nn.Module.__call__(self, obs, name)
        return out

    def forward(self, x, name):
        value = self.value_model(x)
        actions = np.zeros(len(self.policy_models), dtype=np.float32)
        action_space = self.env.action_space(name)
        for i, policy_model in enumerate(self.policy_models):
            low = action_space.low[i]
            high = action_space.high[i]
            step_size = (high - low) / policy_model.out_features
            desired_step = torch.argmax(policy_model(value)).item()
            actions[i] = desired_step * step_size + low
        return actions

    def update(self, batch_size: int = 1) -> float:
        self.train()
        state, action, reward, new_state, terminated = self.memory.sample(batch_size)
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, device=self.device).unsqueeze(0)
        new_state = torch.tensor(new_state, device=self.device).unsqueeze(0)

        # old_targets = self.model(state)
        # # TODO: Add get_old_targets and get_new_targets to model
        # new_targets = reward + self.model(new_state)
        #
        # loss = self.criterion(old_targets, new_targets)
        #
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # if self.lr_scheduler is not None:
        #     self.lr_scheduler.step()
        # return loss.item()
        return 0
