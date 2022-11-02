from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, TYPE_CHECKING, Union

import pettingzoo as pz

import numpy as np

from custom_waterworld.Event import Event

if TYPE_CHECKING:
    from agents import AbstractAgent


class Runner:
    __slots__ = (
        "env",
        "agents",
        "_on_render_event",
        "_on_post_episode_event",
        "_on_finished_iterations_event",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        agents: Dict[str, AbstractAgent],
    ):
        if not isinstance(env, pz.utils.BaseWrapper):
            # Wrap the environment so we can use .unwrap without warnings
            env = pz.utils.BaseWrapper(env)

        # Ensure that each expected agent is in the agents dict
        # and that there are no extras
        env.reset()
        env_agents = set(env.agents)
        missing_agents = env_agents - set(agents.keys())
        if missing_agents:
            raise ValueError(
                f"Missing agents: {missing_agents}. Expected agents: {env_agents}"
            )
        extra_agents = set(agents.keys()) - env_agents
        if extra_agents:
            raise ValueError(
                f"Extra agents: {extra_agents}. Expected agents: {env_agents}"
            )
        self.env = env
        self.agents = agents
        self._on_render_event = Event(
            callback_type=Callable[[Runner, np.ndarray], None]
        )
        self._on_post_episode_event = Event(
            callback_type=Callable[[Runner, Dict[str, list[float]]], None]
        )
        self._on_finished_iterations_event = Event(
            callback_type=Callable[[Runner], None]
        )

    @property
    def on_render(self):
        return self._on_render_event

    @property
    def on_post_episode(self):
        return self._on_post_episode_event

    @property
    def on_finished_iterations(self):
        return self._on_finished_iterations_event

    def _render(self):
        out = self.env.render()
        self._on_render_event(self, out)

    def _on_post_episode(self, rewards: Dict[str, list[float]]):
        self._on_post_episode_event(self, rewards)

    def _on_finished_iterations(self):
        self._on_finished_iterations_event(self)

    def run_episode(self):
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        num_agents = env.num_agents
        self._render()

        for i, agent_name in enumerate(env.agent_iter(), start=1):
            obs, reward, terminated, truncated, info = env.last()
            rewards[agent_name].append(reward)
            agent = self.agents[agent_name]
            action = agent(agent_name, obs)
            action = None if terminated or truncated else action
            env.step(action)
            if i % num_agents == 0:
                self._render()
        self._on_post_episode(rewards)
        return rewards

    def run_iterations(self, iterations: int):
        for _ in range(iterations):
            rewards = self.run_episode()
