from __future__ import annotations
from typing import Any, TYPE_CHECKING

import torch

from agents import AbstractAgent

if TYPE_CHECKING:
    import pettingzoo as pz

    from agents import NeuralNetwork


class A2CAgent(AbstractAgent):
    __slots__ = ("advantage_layers", "shared_layers", "policy_layers")

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        shared_layers: NeuralNetwork,
        advantage_layers: NeuralNetwork,
        policy_layers: [NeuralNetwork],
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)

        assert shared_layers.in_features == env.observation_space(
            self.env_name
        ), "A2C shared_layers in_features must match env.observation_space"

        assert (
            advantage_layers.in_features == shared_layers.out_features
        ), "A2C advantage_layers in_features must match shared_layers out_features"

        assert all(
            pm.in_features == shared_layers.out_features for pm in policy_layers
        ), "A2C policy_layers in_features must match shared_layers out_features"

        assert (
            advantage_layers.out_features == 1
        ), "A2C advantage_layers out_features must be 1"

        assert len(policy_layers) == env.action_space(self.env_name), (
            "A2C must have the same number of policy_layers "
            "as env.action_space has actions"
        )

        self.shared_layers = shared_layers
        self.advantage_layers = advantage_layers
        self.policy_layers = policy_layers

    def __call__(self, obs) -> (torch.Tensor, Any):
        pass

    @property
    def in_features(self):
        return self.shared_layers.in_features

    @property
    def out_features(self):
        # Advantage layer + actions from policy layers
        return 1 + len(self.policy_layers)
