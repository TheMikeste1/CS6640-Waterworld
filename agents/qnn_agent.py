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
        "optimizers",
        "criteria",
        "lr_schedulers",
        "memory",
        "device",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        name: str,
        value_model: torch.nn.Module,
        policy_models: [torch.nn.Module],
        optimizers: [torch.optim.Optimizer],
        criteria: [torch.nn.Module],
        lr_schedulers: [torch.optim.lr_scheduler._LRScheduler] = None,
        auto_select_device: bool = True,
        memory: Memory = None,
    ):
        AbstractAgent.__init__(self, env, name)
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
        self.optimizers = optimizers
        self.criteria = criteria
        if lr_schedulers is None:
            lr_schedulers = [None] * len(optimizers)
        self.lr_schedulers = lr_schedulers

        self.memory = memory if memory is not None else Memory(1024)

        self.device = torch.device(
            "cuda" if auto_select_device and torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def __call__(self, obs) -> np.ndarray:
        self.eval()
        obs = torch.tensor(obs, device=self.device)
        policy_outs = torch.nn.Module.__call__(self, obs)
        actions = np.zeros(len(self.policy_models), dtype=np.float32)
        action_space = self.env.action_space(self.name)
        for i, (pm, values) in enumerate(policy_outs.items()):
            low = action_space.low[i]
            high = action_space.high[i]
            # pm.out_features - 1 so we can reach the high value
            step_size = (high - low) / (pm.out_features - 1)
            desired_step = torch.argmax(values).item()
            actions[i] = desired_step * step_size + low
        return actions

    def _call_no_parse(self, obs) -> dict:
        return torch.nn.Module.__call__(self, obs)

    def forward(self, x) -> dict:
        value = self.value_model(x)
        actions = {pm: pm(value) for pm in self.policy_models}
        return actions

    def update(self, batch_size: int = 1):
        self.train()
        state, action, reward, new_state, terminated = self.memory.sample(batch_size)
        state = torch.tensor(state, device=self.device).unsqueeze(0)
        action = torch.tensor(action, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward, device=self.device).unsqueeze(0)
        new_state = torch.tensor(new_state, device=self.device).unsqueeze(0)

        # FIXME: I'm currently crashing, probably because I'm traversing the
        #  value network twice.
        #  I likely need to get targets for the value network and the policy networks
        #  separately.
        old_targets = self.get_old_targets(state, action)
        new_targets = self.get_new_targets(reward, new_state)

        losses = dict()
        for i, (pm, old_target) in enumerate(old_targets.items()):
            new_target = new_targets[pm]
            optimizer = self.optimizers[i]
            criterion = self.criteria[i]
            lr_scheduler = self.lr_schedulers[i]

            loss = criterion(old_target, new_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                # Don't know why this is giving me unresolved reference;
                # I've already checked it's not none.
                # noinspection PyUnresolvedReferences
                lr_scheduler.step()
            losses[pm] = loss.item()

        return losses

    def get_old_targets(self, state, action):
        # Parse the actions to determine which was taken
        original_out = self._call_no_parse(state)
        action_space = self.env.action_space(self.name)
        for i, pm in enumerate(original_out.keys()):
            low = action_space.low[i]
            high = action_space.high[i]
            # pm.out_features - 1 so we can reach the high value
            step_size = (high - low) / (pm.out_features - 1)
            action[..., i] = (action[..., i] - action_space.low[i]) / step_size
        # Return the current value for the action taken
        # FIXME: I need to test that action[..., i] is always a whole number
        action = action.long()
        old_targets = {
            # Take the value for the action taken
            pm: torch.take(values, action[..., i]).float()
            for i, (pm, values) in enumerate(original_out.items())
        }
        return old_targets

    def get_new_targets(self, reward, new_state):
        new_targets = {
            pm: (reward + torch.amax(values, dim=2)).float()
            for pm, values in self._call_no_parse(new_state).items()
        }
        return new_targets
