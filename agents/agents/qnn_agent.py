from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, TYPE_CHECKING, Union

import torch
import numpy as np
from torch import Tensor

from ..agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import pettingzoo as pz
    from agents import StepData
    from agents.neural_network import NeuralNetwork

    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler


class QNNAgent(AbstractAgent):
    @dataclass(kw_only=True)
    class Builder(AbstractAgent.Builder):
        policy_networks: [NeuralNetwork.Builder]
        optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ]
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ]
        lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] | None = None
        optimizer_kwargs: dict = None
        criterion_kwargs: dict = None
        lr_scheduler_kwargs: dict = None
        device: str | torch.device = None
        memory: Memory | int = None
        batch_size: int = 1
        gamma: float = 0.99
        epsilon: float = 0.1

        def build(self, env: pz.AECEnv) -> QNNAgent:
            kwargs = self.__dict__.copy()
            kwargs["env"] = env
            kwargs["policy_networks"] = [
                builder.build() for builder in self.policy_networks
            ]
            return QNNAgent(**kwargs)

    __slots__ = (
        "batch_size",
        "criterion",
        "gamma",
        "optimizer",
        "lr_scheduler",
        "device",
        "enable_explore",
        "epsilon",
        "memory",
        "policy_networks",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        policy_networks: [NeuralNetwork],
        optimizer_factory: Callable[[Iterable[Tensor], ...], torch.optim.Optimizer],
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ],
        lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None,
        optimizer_kwargs: dict = None,
        criterion_kwargs: dict = None,
        lr_scheduler_kwargs: dict = None,
        device: str | torch.device = None,
        memory: Memory | int = None,
        batch_size: int = 1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)
        if lr_scheduler_factory is None and lr_scheduler_kwargs is not None:
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

        # Assert there are the same number of policy_networks as actions
        assert (
            len(policy_networks) == env.action_space(env.possible_agents[0]).shape[0]
        ), "There must be a policy_model for each action"

        self.policy_networks = torch.nn.ModuleList(policy_networks)

        self.optimizer = optimizer_factory(
            self.parameters(), **(optimizer_kwargs or {})
        )
        self.criterion = criterion_factory(**(criterion_kwargs or {}))
        self.lr_scheduler = (
            lr_scheduler_factory(self.optimizer, **(lr_scheduler_kwargs or {}))
            if lr_scheduler_factory
            else None
        )

        if memory is None:
            memory = Memory(2048)
        elif isinstance(memory, int):
            memory = Memory(memory)
        self.memory = memory
        self.batch_size = batch_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)
        self.epsilon = epsilon
        self.enable_explore = True
        self.gamma = gamma

    def __call__(self, obs: np.ndarray | torch.Tensor) -> (torch.Tensor, Any):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # If obs is not in batch form, add a batch dimension
        while len(obs.shape) < 3:
            obs = obs.unsqueeze(-2)
        if self.enable_explore and np.random.random() < self.epsilon:
            actions = self._get_random_actions(num_actions=obs.shape[0]).squeeze()
            actions = torch.from_numpy(actions)
            action_values = self._action_to_action_values(actions)
            return action_values, actions
        self.eval()
        obs = obs.to(self.device)
        policy_outs = torch.nn.Module.__call__(self, obs)

        actions = torch.argmax(policy_outs, dim=-1).cpu()

        action_values = self._action_to_action_values(actions)
        return action_values, actions.squeeze().long()

    def _action_to_action_values(self, action: torch.Tensor) -> torch.Tensor:
        action_space = self.env.action_space(self.env_name)
        step_sizes = self._calculate_step_size(action_space)
        return (action * step_sizes + action_space.low).to(torch.float32)

    def _action_values_to_action(
        self, action_values: np.ndarray
    ) -> np.ndarray[torch.Tensor]:
        action_space = self.env.action_space(self.env_name)
        step_sizes = self._calculate_step_size(action_space)
        return (action_values - action_space.low) / step_sizes

    def _get_random_actions(self, num_actions: int = 1) -> np.ndarray:
        return np.array(
            [
                np.random.randint(0, pm.out_features, size=num_actions)
                for pm in self.policy_networks
            ],
            dtype=np.long,
        ).T

    def _calculate_step_size(self, action_space):
        out_features = np.array([pm.out_features for pm in self.policy_networks])
        # out_features - 1 because the zeroth feature will give us the low.
        # For example, if we have 10 out features, the 0th feature will give us the low,
        # and the 9th feature will give us the high.
        step_sizes = (action_space.high - action_space.low) / (out_features - 1)
        return step_sizes

    def _call_policies(self, value: torch.Tensor) -> torch.Tensor:
        # Concatenates the output of each policy model
        policy_outs = [pm(value) for pm in self.policy_networks]
        return torch.cat(policy_outs, dim=-2)

    def forward(self, x) -> torch.Tensor:
        # value = self.value_model(x)
        actions = self._call_policies(x)
        return actions

    def get_new_policy_targets(self, reward, new_value: torch.Tensor):
        reward = reward.detach()
        policy_out = self._call_policies(new_value)
        new_targets = reward + self.gamma * torch.amax(policy_out, dim=-1)
        return new_targets.float()

    def get_old_policy_targets(self, value: torch.Tensor, action):
        # Parse the actions to determine which was taken
        old_targets = self._call_policies(value)
        # From the original_out, gather the output of the policy that was taken.

        # We need to transpose the actions to properly align the dimensions.
        action = torch.from_numpy(action).long().to(self.device)
        old_targets = old_targets.take(action)
        return old_targets

    @property
    def in_features(self):
        return self.policy_networks[0].in_features

    def on_train(self):
        return self.update(self.batch_size)

    @property
    def out_features(self):
        return self.policy_networks[0].out_features

    def post_step(self, data: StepData):
        agent_info = data.agent_info
        if isinstance(agent_info, torch.Tensor):
            agent_info = agent_info.detach().cpu().numpy()
        self.memory.add(
            (
                data.state,
                data.action,
                data.reward,
                data.next_state,
                data.terminated,
                agent_info,
            )
        )

    def update(self, batch_size: int = 1):
        self.train()
        state, _, reward, new_state, _, action_index = self.memory.sample(batch_size)
        state = torch.from_numpy(state).to(self.device).unsqueeze(1)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(1)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(1)

        value = state.detach()
        new_value = new_state.detach()
        old_policy_targets = self.get_old_policy_targets(value, action_index)
        new_policy_targets = self.get_new_policy_targets(reward, new_value)

        # Calculate loss
        loss = self.apply_loss(old_policy_targets, new_policy_targets)
        return {
            "loss": loss,
        }

    def apply_loss(
        self, old_policy_targets: torch.Tensor, new_policy_targets: torch.Tensor
    ):
        old_policy_targets = old_policy_targets.to(self.device)
        new_policy_targets = new_policy_targets.to(self.device)
        loss = self.criterion(old_policy_targets, new_policy_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()
