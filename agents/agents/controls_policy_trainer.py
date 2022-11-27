from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Collection, Iterable, TYPE_CHECKING, Union

import numpy as np
import torch

from ..agents import ControlsAgent
from agents import CriticNetwork, Memory, StepData

if TYPE_CHECKING:
    import pettingzoo as pz

    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler


class ControlsPolicyTrainer(ControlsAgent):
    @dataclass(kw_only=True)
    class Builder(ControlsAgent.Builder):
        critic: CriticNetwork.Builder
        critic_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ]
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ]
        critic_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None
        critic_optimizer_kwargs: dict = None
        critic_lr_scheduler_kwargs: dict = None
        criterion_kwargs: dict = None
        device: str | torch.device = None
        memory: Memory | int = None
        batch_size: int = 1
        gamma: float = 0.99
        rho: float = 0.001
        should_explore: bool = True

        def build(self, env: pz.AECEnv) -> ControlsPolicyTrainer:
            kwargs = self.__dict__.copy()
            kwargs["env"] = env
            kwargs["critic"] = self.critic.build()
            return ControlsPolicyTrainer(**kwargs)

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        critic: CriticNetwork,
        critic_optimizer_factory: Callable[
            [Iterable[torch.Tensor], ...], torch.optim.Optimizer
        ],
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ],
        critic_lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None,
        critic_optimizer_kwargs: dict = None,
        critic_lr_scheduler_kwargs: dict = None,
        criterion_kwargs: dict = None,
        device: torch.device | str = None,
        memory: Memory | int = None,
        batch_size: int = 1,
        gamma: float = 0.99,
        rho: float = 0.001,
        should_explore: bool = True,
        name: str = "",
    ):
        super().__init__(env, env_name, name, device)

        if (
            critic_lr_scheduler_kwargs is None
            and critic_lr_scheduler_kwargs is not None
        ):
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

        self.critic = critic
        self.target_critic = critic.clone()
        self.critic_optimizer = critic_optimizer_factory(
            self.critic.parameters(), **(critic_optimizer_kwargs or {})
        )
        self.criterion = criterion_factory(**(criterion_kwargs or {}))
        self.critic_lr_scheduler = (
            critic_lr_scheduler_factory(
                self.critic_optimizer, **(critic_lr_scheduler_kwargs or {})
            )
            if critic_lr_scheduler_factory is not None
            else None
        )
        if memory is None:
            memory = Memory(2048)
        elif isinstance(memory, int):
            memory = Memory(memory)
        self.memory = memory
        self.batch_size = batch_size
        self.rho = rho
        self.gamma = gamma
        self.should_explore = should_explore
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.to(self.device)

    def __call__(self, obs):
        actions, info = super().__call__(obs)
        if self.should_explore:
            # Add some noise to the actions
            actions = actions + torch.normal(0, 0.1, actions.shape).to(self.device)
            # Normalize the actions
            magnitude = torch.linalg.norm(actions, axis=-1).to(self.device)
            actions = actions / magnitude[:, np.newaxis]
            # Randomly choose the new magnitude
            magnitude = torch.rand(magnitude.shape).to(self.device)
            # Scale the actions
            actions = actions * magnitude[:, np.newaxis]
        return actions, info

    def _update_targets(self):
        # https://github.com/JL321/PolicyGradients-torch/blob/master/PolicyGradients
        # /DDPG.py#L107
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
        target_action = self.forward(new_state)
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

        return critic_loss.item()

    @property
    def lr_scheduler(self):
        return {
            "critic": self.critic_lr_scheduler,
        }

    @property
    def optimizer(self):
        return {
            "critic": self.critic_optimizer,
        }

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

        critic_loss = self.apply_loss(state, action, new_state, reward, terminated)

        self._update_targets()

        return {
            "critic_loss": critic_loss / batch_size,
        }
