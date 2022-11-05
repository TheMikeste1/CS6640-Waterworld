from __future__ import annotations
from typing import Any, TYPE_CHECKING

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
        # for policy_model in policy_models:
        #     assert (
        #         policy_model.in_features == value_model.out_features
        #     ), "The policy_model input size must match the value_model output size"

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

    def __call__(self, obs: np.ndarray | torch.Tensor) -> (np.ndarray, Any):
        self.eval()
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # If obs is not in batch form, add a batch dimension
        if len(obs.shape) < 2:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        policy_outs = torch.nn.Module.__call__(self, obs)

        actions = np.array(
            [
                torch.argmax(po, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.long, copy=False)
                for po in policy_outs
            ]
        ).squeeze()

        action_values = self._action_to_action_values(actions)
        return action_values, actions

    def _action_to_action_values(self, action: [torch.Tensor]) -> torch.Tensor:
        action_space = self.env.action_space(self.name)
        step_sizes = self._calculate_step_size(action_space)
        return (action * step_sizes + action_space.low).astype(
            action_space.dtype, copy=False
        )

    def _action_values_to_action(
        self, action_values: np.ndarray
    ) -> np.ndarray[torch.Tensor]:
        action_space = self.env.action_space(self.name)
        step_sizes = self._calculate_step_size(action_space)
        return (action_values - action_space.low) / step_sizes

    def _calculate_step_size(self, action_space):
        out_features = np.array([pm.out_features for pm in self.policy_models])
        # out_features - 1 because the zeroth feature will give us the low.
        # For example, if we have 10 out features, the 0th feature will give us the low,
        # and the 9th feature will give us the high.
        step_sizes = (action_space.high - action_space.low) / (out_features - 1)
        return step_sizes

    def _call_policies(self, value: torch.Tensor) -> list[torch.Tensor]:
        # Concatenates the output of each policy model
        return [pm(value).squeeze() for pm in self.policy_models]

    def forward(self, x) -> [torch.Tensor]:
        # value = self.value_model(x)
        actions = self._call_policies(x)
        return actions

    def get_new_policy_targets(self, reward, new_value: torch.Tensor):
        return [
            (reward.detach() + torch.amax(v, dim=-1)).squeeze()
            for v in self._call_policies(new_value)
        ]

    def get_old_policy_targets(self, value: torch.Tensor, action):
        # Parse the actions to determine which was taken
        old_targets = self._call_policies(value)
        # From the original_out, gather the output of the policy that was taken.

        # We need to transpose the actions to properly align the dimensions.
        action = torch.tensor(action.T, device=self.device, dtype=torch.int64)
        old_targets = [
            # Given the index of the action,
            # unsqueeze the action to add the target dimension at the end
            # then gather the values at those indices.
            # Finally, squeeze the result to put all the values in the same dimension.
            o.gather(-1, a.unsqueeze(-1)).squeeze()
            for o, a in zip(old_targets, action)
        ]
        return old_targets

    def post_episode(self):
        self.update(self.batch_size)

    def post_step(self, data: StepData):
        self.memory.add(
            (
                data.state,
                data.action,
                data.reward,
                data.next_state,
                data.terminated,
                data.agent_info,
            )
        )

    def update(self, batch_size: int = 1):
        self.train()
        state, action, reward, new_state, terminated, action_index = self.memory.sample(
            batch_size
        )
        state = torch.from_numpy(state).to(self.device).unsqueeze(0)
        # action = torch.from_numpy(action).to(self.device).unsqueeze(0)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(0)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(0)

        # old_value_targets = self.value_model(state)
        # new_value_targets = self.value_model(new_state)
        # self.value_model.step(old_value_targets, new_value_targets)

        # value = self.value_model(state).detach()
        # new_value = self.value_model(new_state).detach()
        value = state.detach()
        new_value = new_state.detach()
        old_policy_targets = self.get_old_policy_targets(value, action_index)
        new_policy_targets = self.get_new_policy_targets(reward, new_value)

        losses = {
            pm: pm.step(old_target, new_target)
            for pm, old_target, new_target in zip(
                self.policy_models, old_policy_targets, new_policy_targets
            )
        }
        return losses
