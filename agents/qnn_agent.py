from __future__ import annotations
from typing import Any, Callable, Iterable, TYPE_CHECKING, Union

import torch
import numpy as np
from torch import Tensor

from agents import AbstractAgent
from agents.memory import Memory

if TYPE_CHECKING:
    import pettingzoo as pz
    from agents.step_data import StepData
    from agents.neural_network import NeuralNetwork

    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler


class QNNAgent(AbstractAgent):
    __slots__ = (
        "batch_size",
        "criterion",
        "observation_parser",
        "optimizer",
        "lr_scheduler",
        "device",
        "enable_explore",
        "epsilon",
        "memory",
        "policy_models",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        env_name: str,
        policy_models: [NeuralNetwork],
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
        auto_select_device: bool = True,
        memory: Memory | int = None,
        batch_size: int = 1,
        observation_parser: Callable[[torch.Tensor], torch.Tensor] = lambda x: x,
        name: str = "",
    ):
        AbstractAgent.__init__(self, env, env_name, name=name)
        if lr_scheduler_factory is None and lr_scheduler_kwargs is not None:
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

        # Assert there are the same number of policy_models as actions
        assert (
            len(policy_models) == env.action_space(env.possible_agents[0]).shape[0]
        ), "There must be a policy_model for each action"

        self.policy_models = torch.nn.ModuleList(policy_models)

        self.optimizer = optimizer_factory(
            self.parameters(), **(optimizer_kwargs or {})
        )
        self.criterion = criterion_factory(**(criterion_kwargs or {}))
        self.lr_scheduler = (
            lr_scheduler_factory(self.optimizer, **(lr_scheduler_kwargs or {}))
            if lr_scheduler_factory
            else None
        )

        self.observation_parser = observation_parser

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
        self.epsilon = 0.1
        self.enable_explore = True

    def __call__(self, obs: np.ndarray | torch.Tensor) -> (np.ndarray, Any):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs)
        # If obs is not in batch form, add a batch dimension
        if len(obs.shape) < 2:
            obs = obs.unsqueeze(-2)
        if self.enable_explore and np.random.random() < self.epsilon:
            actions = self._get_random_actions(num_actions=obs.shape[0]).squeeze()
            return self._action_to_action_values(actions), actions
        self.eval()
        obs = obs.to(self.device)
        obs = self.observation_parser(obs)
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
        action_space = self.env.action_space(self.env_name)
        step_sizes = self._calculate_step_size(action_space)
        return (action * step_sizes + action_space.low).astype(
            action_space.dtype, copy=False
        )

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
                for pm in self.policy_models
            ],
            dtype=np.long,
        ).T

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
        new_targets = []
        reward = reward.detach()
        for v in self._call_policies(new_value):
            new_targets.append(reward + torch.amax(v, dim=-1, keepdim=True))

        return torch.stack(new_targets, dim=-1)

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
            o.gather(-1, a.unsqueeze(-1))
            for o, a in zip(old_targets, action)
        ]
        return torch.stack(old_targets, dim=-1)

    @property
    def in_features(self):
        return self.policy_models[0].in_features

    def on_train(self):
        return self.update(self.batch_size)

    @property
    def out_features(self):
        return self.policy_models[0].out_features

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
        state = torch.from_numpy(state).to(self.device).unsqueeze(1)
        reward = torch.from_numpy(reward).to(self.device).unsqueeze(1)
        new_state = torch.from_numpy(new_state).to(self.device).unsqueeze(1)

        value = state.detach()
        new_value = new_state.detach()
        old_policy_targets = self.get_old_policy_targets(value, action_index)
        new_policy_targets = self.get_new_policy_targets(reward, new_value)

        # Calculate loss
        loss = self.criterion(old_policy_targets, new_policy_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        return loss.item()
