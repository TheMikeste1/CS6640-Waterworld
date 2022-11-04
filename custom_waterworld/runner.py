from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, TYPE_CHECKING

import numpy as np
import pettingzoo as pz
from tqdm import tqdm

from agents.step_data import StepData
from custom_waterworld.event import Event

if TYPE_CHECKING:
    from agents import AbstractAgent


# TODO: Add a way to run without training
class Runner:
    __slots__ = (
        "agents",
        "enable_tqdm",
        "env",
        "on_finished_iterations",
        "on_post_episode",
        "on_render",
        "on_step",
        "should_render_empty",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        agents: Dict[str, AbstractAgent],
        should_render_empty: bool = False,
        enable_tqdm: bool = True,
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
        self.on_finished_iterations = Event(callback_type=Callable[[Runner], None])
        self.on_post_episode = Event(
            callback_type=Callable[[Runner, Dict[str, list[float]]], None]
        )
        self.on_render = Event(callback_type=Callable[[Runner, np.ndarray], None])
        self.on_step = Event(callback_type=Callable[[Runner, str, StepData], None])

        self.should_render_empty = should_render_empty
        self.enable_tqdm = enable_tqdm

    def _on_finished_iterations(self):
        self.on_finished_iterations(self)

    def _on_post_episode(self, rewards: Dict[str, list[float]]):
        for agent in self.agents.values():
            agent.post_episode()
        self.on_post_episode(self, rewards)

    def _on_step(
        self, agent_name, state, next_state, action, reward, terminated, truncated, info
    ):
        step_data = StepData(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )
        self.agents[agent_name].post_step(step_data)
        self.on_step(self, agent_name, step_data)

    def _render(self):
        # If no one is listening, don't bother rendering
        # This should speed things up a bit
        if self.should_render_empty or len(self.on_render) > 0:
            out = self.env.render()
            self.on_render(self, out)

    def run_episode(self, train: bool = True):
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        num_agents = env.num_agents
        self._render()

        cached_data = dict()
        for i, agent_name in enumerate(env.agent_iter(), start=1):
            obs, reward, terminated, truncated, info = env.last()
            rewards[agent_name].append(reward)
            agent = self.agents[agent_name]
            action = agent(obs)
            # If the agent is dead or truncated the only allowed action is None
            env.step(None if terminated or truncated else action)
            # Cache the data for updates later
            cached_data[agent_name] = {
                "state": obs,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "info": info,
            }
            # Once all agents have taken a step, we can render and update
            if i % num_agents == 0:
                self._render()
                for name, data in cached_data.items():
                    # TODO: Might be able to optimize this
                    #  by updating using the call to env.last().
                    #  It's currently fairly expensive, so it would be nice to do.
                    next_state = env.observe(name)
                    self._on_step(agent_name=name, next_state=next_state, **data)
                cached_data.clear()
        self._on_post_episode(rewards)
        return rewards

    def run_iterations(self, iterations: int):
        rewards = None
        bar = range(iterations)
        if self.enable_tqdm:
            bar = tqdm(bar)
        for _ in bar:
            rewards = self.run_episode()
        print(rewards)
        self._on_finished_iterations()
