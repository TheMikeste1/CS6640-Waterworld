from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, TYPE_CHECKING, Union

import pettingzoo as pz


if TYPE_CHECKING:
    import numpy as np

    from agents import AbstractAgent


class Runner:
    __slots__ = (
        "env",
        "agents",
        "_on_render_callbacks",
        "_on_post_episode_callbacks",
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
        self._on_render_callbacks = set()
        self._on_post_episode_callbacks = set()

    def _render(self):
        out = self.env.render()
        for callback in self._on_render_callbacks:
            callback(self, out)

    def _on_post_episode(self, rewards: Dict[str, list[float]]):
        for callback in self._on_post_episode_callbacks:
            callback(self, rewards)

    def run_episode(self):
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        num_agents = env.num_agents

        render_out = env.render()
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

    def subscribe_render(
        self, callback: Callable[[Runner, Union[np.array, None]], None]
    ):
        self._on_render_callbacks.add(callback)

    def unsubscribe_render(
        self, callback: Callable[[Runner, Union[np.array, None]], None]
    ):
        self._on_render_callbacks.remove(callback)

    def subscribe_post_episode(
        self, callback: Callable[[Runner, Dict[str, list[float]]], None]
    ):
        self._on_post_episode_callbacks.add(callback)

    def unsubscribe_post_episode(
        self, callback: Callable[[Runner, Dict[str, list[float]]], None]
    ):
        self._on_post_episode_callbacks.remove(callback)
