from __future__ import annotations

from collections import defaultdict
from typing import Callable, TYPE_CHECKING

import pettingzoo as pz

if TYPE_CHECKING:
    import numpy as np


class Runner:
    def __init__(
        self,
        env: pz.AECEnv,
        on_render: Callable[[np.array], None] = None,
    ):
        if not isinstance(env, pz.utils.BaseWrapper):
            # Wrap the environment so we can use .unwrap without warnings
            env = pz.utils.BaseWrapper(env)
        self.env = env

        if on_render is None:
            on_render = lambda x: None
        self.on_render = on_render

    def run_episode(self):
        env = self.env

        rewards = defaultdict(list)
        env.reset()
        render_out = env.render()
        self.on_render(render_out)
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            rewards[agent].append(reward)
            action = (
                None if terminated or truncated else env.action_space(agent).sample()
            )
            env.step(action)
            render_out = env.render()
            self.on_render(render_out)
