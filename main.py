import pettingzoo.sisl.waterworld_v4


# Link to environment base class:
# from pettingzoo.sisl.waterworld.waterworld_base import WaterworldBase


@property
def base_env(self) -> pettingzoo.sisl.waterworld_v4.env:
    return self.env.env.env


if __name__ == "__main__":
    kwargs = dict(render_mode="human", FPS=120, n_poisons=100)
    env = pettingzoo.sisl.waterworld_v4.env(**kwargs)
    setattr(env.__class__, "base_env", base_env)
    print(f"Running at {env.base_env.FPS} FPS")
    env.reset()
    env.render()

    for agent in env.agent_iter():
        obs, reward, done, truncated, info = env.last()
        action = None if env.terminations[agent] or env.truncations[agent] else env.action_space(agent).sample()
        env.step(action)
        env.render()
    env.close()
