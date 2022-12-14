from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
import pettingzoo as pz
from tqdm.auto import tqdm

from agents import StepData
from custom_waterworld.event import Event

if TYPE_CHECKING:
    from agents import AbstractAgent

REWARDS_TYPE = Dict[str, list[float]]
POST_EPISODE_TYPE = Dict[str, Any]
POST_TRAIN_TYPE = Dict[str, float]


class Runner:
    __slots__ = (
        "agents",
        "enable_tqdm",
        "env",
        "on_finished_iterations",
        "on_post_episode",
        "on_post_step",
        "on_post_test_iterations",
        "on_post_train",
        "on_render",
        "should_render_empty",
        "tqdm_kwargs",
    )

    def __init__(
        self,
        env: pz.AECEnv,
        agents: Dict[str, AbstractAgent],
        should_render_empty: bool = False,
        enable_tqdm: bool = True,
        tqdm_kwargs: dict = None,
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
            callback_type=Callable[[Runner, int, REWARDS_TYPE, POST_EPISODE_TYPE], None]
        )
        self.on_render = Event(callback_type=Callable[[Runner, np.ndarray], None])
        self.on_post_step = Event(
            callback_type=Callable[[Runner, int, str, StepData], None]
        )
        self.on_post_train = Event(
            callback_type=Callable[[Runner, int, POST_TRAIN_TYPE], None]
        )
        self.on_post_test_iterations = Event(
            callback_type=Callable[[Runner, int, dict[str, list[list[float]]]], None]
        )

        self.should_render_empty = should_render_empty
        self.enable_tqdm = enable_tqdm
        self.tqdm_kwargs = tqdm_kwargs or {}

    def _on_finished_iterations(self):
        self.on_finished_iterations(self)

    def _on_post_episode(self, iteration: int, rewards: REWARDS_TYPE):
        agent_posts = dict()
        for agent_name, agent in self.agents.items():
            agent_posts[agent_name] = agent.post_episode()
        self.on_post_episode(self, iteration, rewards, agent_posts)

    def _post_step(
        self,
        training: bool,
        agent_name,
        state,
        next_state,
        action,
        reward,
        terminated,
        truncated,
        info,
        agent_info,
    ):
        step_data = StepData(
            state=state,
            next_state=next_state,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            agent_info=agent_info,
        )
        if training:
            agent = self.agents[agent_name]
            agent.train()
            agent.post_step(step_data)
            agent.eval()
        self.on_post_step(self, agent_name, step_data)

    def _post_train(self, iteration: int):
        agent_trains = dict()
        for agent_name, agent in self.agents.items():
            agent_trains[agent_name] = agent.on_train()
        self.on_post_train(self, iteration, agent_trains)

    def _post_test(self, i, rewards):
        self.on_post_test_iterations(self, i, rewards)

    def _render(self):
        # If no one is listening, don't bother rendering
        # This should speed things up a bit
        if self.should_render_empty or len(self.on_render) > 0:
            out = self.env.render()
            self.on_render(self, out)

    def run_episode(
        self, train: bool = True, explore: bool = True, with_dataframe: bool = False
    ) -> tuple[defaultdict[Any, list], pd.DataFrame | None] | defaultdict[Any, list]:
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        num_agents = env.num_agents
        self._render()

        df_out = None
        if with_dataframe:
            df_out = pd.DataFrame(columns=["agent", "i", "state", "action", "reward"])

        for agent in self.agents.values():
            agent.should_explore = explore
            agent.eval()

        cached_data = dict()
        for i, agent_name in enumerate(env.agent_iter(), start=1):
            obs, reward, terminated, truncated, info = env.last()
            rewards[agent_name].append(reward)
            agent = self.agents[agent_name]
            action, agent_info = agent(obs)
            action = action.detach().cpu().numpy()
            action = action.squeeze()  # Remove any extra dimensions

            if with_dataframe:
                df_out.loc[len(df_out)] = [agent_name, i, obs, action, reward]

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
                "agent_info": agent_info,
            }
            # Once all agents have taken a step, we can render and update
            if i % num_agents == 0:
                self._render()
                for name, data in cached_data.items():
                    # TODO: Might be able to optimize this
                    #  by updating using the call to env.last().
                    #  It's currently fairly expensive, so it would be nice to do.
                    next_state = env.observe(name)
                    self._post_step(
                        training=train, agent_name=name, next_state=next_state, **data
                    )
                cached_data.clear()
        if with_dataframe:
            return rewards, df_out
        return rewards

    def run_episode_with_dataframe(
        self,
        train: bool = True,
        explore: bool = True,
    ) -> (dict, pd.DataFrame):
        return self.run_episode(train=train,explore=explore, with_dataframe=True)

    def run_iterations(
        self,
        iterations: int,
        train: bool = True,
        test_iterations: int = 0,
        test_interval: int = 1,
    ):
        assert test_interval > 0
        assert test_iterations >= 0

        bar = range(iterations)
        if self.enable_tqdm:
            bar = tqdm(bar, **self.tqdm_kwargs)
        for i in bar:
            rewards = self.run_episode(train)
            self._on_post_episode(i, rewards)
            if train:
                self._post_train(i)
            for agent in self.agents.values():
                agent.reset()
            if test_iterations > 0 and i % test_interval == 0:
                rewards = {
                    agent_name: []
                    for agent_name in self.agents.keys()
                }
                for _ in range(test_iterations):
                    episode_rewards = self.run_episode(train=False, explore=False)
                    for agent_name, agent_rewards in episode_rewards.items():
                        rewards[agent_name].append(agent_rewards)
                self._post_test(i, rewards)
        self._on_finished_iterations()
