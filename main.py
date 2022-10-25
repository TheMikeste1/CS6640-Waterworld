import custom_waterworld

if __name__ == "__main__":
    kwargs = dict(render_mode="human", FPS=120, n_poisons=100)
    env = custom_waterworld.waterworld.env(**kwargs)
    print(f"Running at {env.base_env.FPS} FPS")
    env.reset()
    env.render()

    for agent in env.agent_iter():
        obs, reward, done, truncated, info = env.last()
        action = (
            None
            if env.terminations[agent] or env.truncations[agent]
            else env.action_space(agent).sample()
        )
        env.step(action)
        env.render()
        terminations = sum(env.terminations.values())
        truncations = sum(env.truncations.values())
        print(f"Terminations: {terminations}, Truncations: {truncations}")

    env.close()
