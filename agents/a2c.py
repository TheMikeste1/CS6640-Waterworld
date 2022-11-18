from __future__ import annotations
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from agents import AbstractAgent

if TYPE_CHECKING:
    import pettingzoo as pz

    from agents import NeuralNetwork


class A2CAgent(AbstractAgent):
    __slots__ = ("advantage_network", "device", "shared_network", "policy_networks")

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        shared_network: NeuralNetwork,
        advantage_network: NeuralNetwork,
        policy_networks: [NeuralNetwork],
        device: str | torch.device = None,
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)

        assert (
            shared_network.in_features == env.observation_space(self.env_name).shape[0]
        ), (
            "A2C shared_network in_features must match env.observation_space. "
            f"Got {shared_network.in_features}, expected "
            f"{env.observation_space(self.env_name).shape[0]}"
        )

        assert advantage_network.in_features == shared_network.out_features, (
            "A2C advantage_network in_features must match shared_network out_features. "
            f"Got {advantage_network.in_features}, expected "
            f"{shared_network.out_features}"
        )

        assert all(
            pm.in_features == shared_network.out_features for pm in policy_networks
        ), "A2C policy_networks in_features must match shared_network out_features"

        assert (
            advantage_network.out_features == 1
        ), "A2C advantage_network out_features must be exactly 1"

        assert len(policy_networks) == env.action_space(self.env_name).shape[0], (
            "A2C must have the same number of policy_networks "
            "as env.action_space has actions. "
            f"Got {len(policy_networks)}, expected "
            f"{env.action_space(self.env_name).shape[0]}"
        )

        self.shared_network = shared_network
        self.advantage_network = advantage_network
        self.policy_networks = torch.nn.ModuleList(policy_networks)

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
        action_probs, _ = torch.nn.Module.__call__(self, obs)

        categories = torch.distributions.Categorical(action_probs)
        actions_to_take = categories.sample()
        probabilities = categories.log_prob(actions_to_take)
        actions = self._action_to_action_values(actions_to_take)
        return actions, probabilities

    def _action_to_action_values(self, action: torch.Tensor) -> torch.Tensor:
        action_space = self.env.action_space(self.env_name)
        step_sizes = self._calculate_step_size(action_space).to(self.device)
        low = torch.tensor(action_space.low).to(self.device)
        return (action * step_sizes + low).to(torch.float32)

    def _calculate_step_size(self, action_space):
        out_features = torch.tensor([pm.out_features for pm in self.policy_networks])
        # out_features - 1 because the zeroth feature will give us the low.
        # For example, if we have 10 out features, the 0th feature will give us the low,
        # and the 9th feature will give us the high.
        step_sizes = (action_space.high - action_space.low) / (out_features - 1)
        return step_sizes

    def _call_policies(self, value: torch.Tensor) -> torch.Tensor:
        # Concatenates the output of each policy model
        policy_outs = [pm(value) for pm in self.policy_networks]
        return torch.cat(policy_outs, dim=-2)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        shared_out = self.shared_network(x)
        advantage = self.advantage_network(shared_out)
        policy_out = self._call_policies(shared_out)
        return policy_out, advantage

    @property
    def in_features(self):
        return self.shared_network.in_features

    @property
    def out_features(self):
        # Advantage layer + actions from policy layers
        return 1 + len(self.policy_networks)
