from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import pettingzoo as pz
    from agents.step_data import StepData
    from agents.neural_network import NeuralNetwork


class QNNAgent(AbstractAgent, torch.nn.Module):
    __slots__ = (
        "batch_size",
        "device",
        "memory",
        "policy_models",
        "value_model",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        name: str,
        value_model: NeuralNetwork,
        policy_models: [NeuralNetwork],
        auto_select_device: bool = True,
        memory: Memory | int = None,
        batch_size: int = 1,
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

        if memory is None:
            memory = Memory(2048)
        elif isinstance(memory, int):
            memory = Memory(memory)
        self.memory = memory
        self.batch_size = batch_size
        self.device = torch.device(
            "cuda" if auto_select_device and torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def __call__(self, obs: np.ndarray | torch.Tensor) -> np.ndarray:
        self.eval()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        obs = obs.to(self.device).unsqueeze(0)
        policy_outs = torch.nn.Module.__call__(self, obs)
        actions = np.zeros(len(self.policy_models), dtype=np.float32)
        action_space = self.env.action_space(self.name)
        for i, (pm, values) in enumerate(policy_outs.items()):
            low = action_space.low[i]
            high = action_space.high[i]
            # pm.out_features - 1 so we can reach the high value, minus another 1
            # so feature[0] is 0 (do nothing)
            step_size = (high - low) / (pm.out_features - 2)
            desired_step = torch.argmax(values)
            # If desired_step == 0, keep it that way so we do nothing
            actions[i] = (desired_step == 0) * (low + step_size * desired_step)
        return actions

    def _call_policies(self, value: torch.Tensor) -> dict:
        return {pm: pm(value) for pm in self.policy_models}

    def forward(self, x) -> dict:
        value = self.value_model(x)
        actions = self._call_policies(value)
        return actions

    def get_new_policy_targets(self, reward, new_value: torch.Tensor):
        new_targets = {
            pm: (reward + torch.amax(values, dim=2)).float()
            for pm, values in self._call_policies(new_value).items()
        }
        return new_targets

    def get_old_policy_targets(self, value: torch.Tensor, action):
        # Parse the actions to determine which was taken
        original_out = self._call_policies(value)
        action_space = self.env.action_space(self.name)
        for i, pm in enumerate(original_out.keys()):
            low = action_space.low[i]
            high = action_space.high[i]
            # pm.out_features - 1 so we can reach the high value,
            # minus another 1 so feature[0] is 0 (do nothing)
            step_size = (high - low) / (pm.out_features - 2)
            action[..., i] = (
                # If the action is 0, we want to use the first feature,
                # which is hard-set to be 0 (do nothing)
                (action[..., i] != 0)
                * (action[..., i] - action_space.low[i])
                / step_size
            )
        # Return the current value for the action taken
        # FIXME: I need to test that action[..., i] is always a whole number
        action = action.long()
        old_targets = {
            # Take the value for the action taken
            pm: torch.take(values, action[..., i]).float()
            for i, (pm, values) in enumerate(original_out.items())
        }
        return old_targets

    def post_episode(self):
        self.update(self.batch_size)

    def post_step(self, data: StepData):
        self.memory.add(
            (data.state, data.action, data.reward, data.next_state, data.terminated)
        )

    def update(self, batch_size: int = 1):
        self.train()
        state, action, reward, new_state, terminated = self.memory.sample(batch_size)
        state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        action = torch.from_numpy(action).to(self.device).unsqueeze(0)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(0)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(0)

        old_value_targets = self.value_model(state)
        new_value_targets = self.value_model(new_state)
        self.value_model.step(old_value_targets, new_value_targets)

        value = self.value_model(state).detach()
        new_value = self.value_model(new_state).detach()
        old_policy_targets = self.get_old_policy_targets(value, action)
        new_policy_targets = self.get_new_policy_targets(reward, new_value)

        losses = dict()
        for pm, old_target in old_policy_targets.items():
            new_target = new_policy_targets[pm]
            losses[pm] = pm.step(old_target, new_target)
        return losses
