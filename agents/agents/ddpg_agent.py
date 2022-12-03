from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Collection, Iterable, TYPE_CHECKING, Union

import numpy as np
import torch

from ..agents import AbstractAgent
from agents import CriticNetwork, Memory

if TYPE_CHECKING:
    import pettingzoo as pz

    from agents import NeuralNetwork
    from agents import StepData

    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler


class DDPGAgent(AbstractAgent):
    @dataclass(kw_only=True)
    class Builder(AbstractAgent.Builder):
        actor: NeuralNetwork.Builder
        critic: CriticNetwork.Builder
        actor_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ]
        critic_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ]
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ]
        actor_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None
        critic_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None
        actor_optimizer_kwargs: dict = None
        actor_lr_scheduler_kwargs: dict = None
        critic_optimizer_kwargs: dict = None
        critic_lr_scheduler_kwargs: dict = None
        criterion_kwargs: dict = None
        device: str | torch.device = None
        memory: Memory | int = None
        batch_size: int = 1
        gamma: float = 0.99
        rho: float = 0.001
        should_explore: bool = True
        explore_distribution: torch.distributions.Distribution = None

        def build(self, env: pz.AECEnv) -> DDPGAgent:
            kwargs = self.__dict__.copy()
            kwargs["env"] = env
            kwargs["actor"] = self.actor.build()
            kwargs["critic"] = self.critic.build()
            return DDPGAgent(**kwargs)

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        actor: NeuralNetwork,
        critic: CriticNetwork,
        actor_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ],
        critic_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ],
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ],
        actor_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None,
        critic_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None,
        actor_optimizer_kwargs: dict = None,
        actor_lr_scheduler_kwargs: dict = None,
        critic_optimizer_kwargs: dict = None,
        critic_lr_scheduler_kwargs: dict = None,
        criterion_kwargs: dict = None,
        device: str | torch.device = None,
        memory: Memory | int = None,
        batch_size: int = 1,
        gamma: float = 0.99,
        rho: float = 0.001,
        should_explore: bool = True,
        explore_distribution: torch.distributions.Distribution = None,
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)
        if actor_lr_scheduler_kwargs is None and actor_lr_scheduler_kwargs is not None:
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )
        if (
            critic_lr_scheduler_kwargs is None
            and critic_lr_scheduler_kwargs is not None
        ):
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

        action_space = env.action_space(self.env_name).shape[0]
        assert (
            actor.out_features == action_space
        ), f"Actor output features ({actor.out_features}) must match action space."

        self.actor = actor
        self.critic = critic
        self.actor_optimizer = actor_optimizer_factory(
            self.actor.parameters(), **(actor_optimizer_kwargs or {})
        )
        self.critic_optimizer = critic_optimizer_factory(
            self.critic.parameters(), **(critic_optimizer_kwargs or {})
        )
        self.criterion = criterion_factory(**(criterion_kwargs or {}))
        self.actor_lr_scheduler = (
            actor_lr_scheduler_factory(
                self.actor_optimizer, **(actor_lr_scheduler_kwargs or {})
            )
            if actor_lr_scheduler_factory is not None
            else None
        )
        self.critic_lr_scheduler = (
            critic_lr_scheduler_factory(
                self.critic_optimizer, **(critic_lr_scheduler_kwargs or {})
            )
            if critic_lr_scheduler_factory is not None
            else None
        )

        self.target_actor = actor.clone()
        self.target_critic = critic.clone()

        if memory is None:
            memory = Memory(2048)
        elif isinstance(memory, int):
            memory = Memory(memory)
        self.memory = memory
        self.batch_size = batch_size
        self.gamma = gamma
        self.rho = rho
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)

        self.should_explore = should_explore
        if explore_distribution is None:
            explore_distribution = torch.distributions.Normal(
                torch.zeros(action_space), torch.ones(action_space)
            )
        self.explore_distribution = explore_distribution

    def __call__(self, obs) -> (torch.Tensor, Any):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # If obs is not in batch form, add a batch dimension
        while len(obs.shape) < 3:
            obs = obs.unsqueeze(-2)
        obs = obs.to(self.device)
        actions, *_ = torch.nn.Module.__call__(self, obs)
        # https://github.com/JL321/PolicyGradients-torch/blob/master/PolicyGradients
        # /DDPG.py#L80
        if self.should_explore:
            actions = actions + self.explore_distribution.sample().to(self.device) * 0.1
        actions = torch.clamp(actions, -1, 1)
        return actions, None

    def _update_targets(self):
        # https://github.com/JL321/PolicyGradients-torch/blob/master/PolicyGradients
        # /DDPG.py#L107
        for target_params, current_params in zip(
            self.target_actor.parameters(), self.actor.parameters()
        ):
            target_params.data *= 1 - self.rho
            target_params.data += self.rho * current_params.data
        for target_params, current_params in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_params.data *= 1 - self.rho
            target_params.data += self.rho * current_params.data

    def apply_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        new_state: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
    ):
        self.critic_optimizer.zero_grad()
        target_action = self.target_actor(new_state)
        target_action = torch.tanh(target_action)
        target_value = (
            reward
            + terminated * self.gamma * self.target_critic(new_state, target_action)
        ).to(dtype=action.dtype)
        value = self.critic(state, action)
        critic_loss = self.criterion(value, target_value.detach())
        critic_loss.backward()
        self.critic_optimizer.step()
        if self.critic_lr_scheduler is not None:
            self.critic_lr_scheduler.step()

        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic(state, self.forward(state)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        if self.actor_lr_scheduler is not None:
            self.actor_lr_scheduler.step()
        return critic_loss.item(), actor_loss.item()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        actor_out = self.actor(obs)
        out = torch.tanh(actor_out)
        return out

    @property
    def in_features(self):
        return self.actor.in_features

    @property
    def lr_scheduler(self):
        return {
            "actor": self.actor_lr_scheduler,
            "critic": self.critic_lr_scheduler,
        }

    @property
    def optimizer(self):
        return {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
        }

    @property
    def out_features(self):
        return self.actor.out_features

    def on_train(self):
        return self.update(self.batch_size)

    def post_step(self, data: StepData):
        self.memory.add(
            (
                data.state,
                data.action,
                data.next_state,
                data.terminated,
            ),
            data.reward,
        )

    def update(self, batch_size: int = 1):
        self.train()
        state, action, new_state, terminated, reward = self.memory.sample(
            batch_size, use_replacement_on_overflow=False
        )
        state = torch.from_numpy(state).to(self.device).unsqueeze(1)
        action = torch.from_numpy(action).to(self.device).unsqueeze(1)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(1).unsqueeze(1)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(1)
        terminated = (
            torch.from_numpy(terminated).to(self.device).unsqueeze(1).unsqueeze(1)
        )

        critic_loss, actor_loss = self.apply_loss(
            state, action, new_state, reward, terminated
        )

        self._update_targets()

        return {
            "critic_loss": critic_loss / batch_size,
            "actor_loss": actor_loss / batch_size,
        }

    def reset(self):
        self.actor.reset()
        self.critic.reset()
