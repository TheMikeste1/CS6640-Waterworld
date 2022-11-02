from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, TYPE_CHECKING

import pettingzoo as pz


if TYPE_CHECKING:
    import numpy as np

    from agents import AbstractAgent


class Runner:
    def __init__(
        self,
        env: pz.AECEnv,
        agents: Dict[str, AbstractAgent],
        on_render: Callable[[pz.utils.BaseWrapper, np.array], None] = lambda *_: None,
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
        self.on_render = on_render

    def run_episode(self):
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        render_out = env.render()
        self.on_render(env, render_out)
        for agent_name in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            rewards[agent_name].append(reward)
            agent = self.agents[agent_name]
            action = agent(agent_name, obs)
            action = None if terminated or truncated else action
            env.step(action)
            render_out = env.render()
            self.on_render(env, render_out)
