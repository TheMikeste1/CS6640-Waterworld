from __future__ import annotations
from typing import Any, Callable, Iterable, TYPE_CHECKING, Union

import numpy as np
import torch

from agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import pettingzoo as pz

    from agents import NeuralNetwork
    from agents.step_data import StepData

    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler


class A2CAgent(AbstractAgent):
    __slots__ = (
        "advantage_network",
        "batch_size",
        "criterion",
        "critic_loss_weight",
        "device",
        "gamma",
        "lr_scheduler",
        "memory",
        "optimizer",
        "policy_networks",
        "shared_network",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        shared_network: NeuralNetwork,
        advantage_network: NeuralNetwork,
        policy_networks: [NeuralNetwork],
        optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ],
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
        critic_loss_weight: float = 0.5,
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)
        if lr_scheduler_factory is None and lr_scheduler_kwargs is not None:
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

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

        # Verify the last layer of each pm is not a softmax;
        # we will apply the softmax ourselves
        for pm in policy_networks:
            assert not isinstance(pm[-1], torch.nn.Softmax) and not isinstance(
                pm[-1], torch.nn.LogSoftmax
            ), (
                "A2C policy_networks should not have a softmax layer "
                "as the final layer. "
                "The softmax will be applied automatically."
            )

        self.shared_network = shared_network
        self.advantage_network = advantage_network
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
        self.gamma = gamma
        self.critic_loss_weight = critic_loss_weight

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

    def apply_loss(
        self,
        value: torch.Tensor,
        next_value: torch.Tensor,
        reward: torch.Tensor,
        log_probabilities: torch.Tensor,
    ):
        # https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b#1384
        target_value = (reward + self.gamma * next_value).to(
            device=self.device, dtype=value.dtype
        )
        critic_loss = self.criterion(value, target_value)

        # Calculate the advantage
        advantage = target_value - value
        # https://medium.com/deeplearningmadeeasy/advantage-actor-critic-a2c-implementation-944e98616b#336b
        actor_loss = (-log_probabilities * advantage.detach()).sum()

        loss = actor_loss + (critic_loss * self.critic_loss_weight)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        shared_out = self.shared_network(x)
        advantage = self.advantage_network(shared_out)
        policy_out = self._call_policies(shared_out)
        policy_out = torch.nn.functional.log_softmax(policy_out, dim=-2)
        return policy_out, advantage

    def get_possible_steps(self):
        action_space = self.env.action_space(self.env_name)
        step_sizes = self._calculate_step_size(action_space)
        high = action_space.high
        low = action_space.low
        possible_steps = []
        for i in range(len(step_sizes)):
            possible_steps.append(
                np.arange(low[i], high[i] + step_sizes[i], step_sizes[i])
            )
        return possible_steps

    @property
    def in_features(self):
        return self.shared_network.in_features

    @property
    def out_features(self):
        # Advantage layer + actions from policy layers
        return 1 + len(self.policy_networks)

    def on_train(self):
        return self.update(self.batch_size)

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
        state, _, reward, new_state, _, log_probabilities = self.memory.sample(
            batch_size
        )
        state = torch.from_numpy(state).to(self.device).unsqueeze(1)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(1).unsqueeze(1)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(1)
        log_probabilities = torch.from_numpy(log_probabilities).to(self.device)

        _, value = self.forward(state)
        _, next_value = self.forward(new_state)

        return self.apply_loss(value, next_value, reward, log_probabilities)
