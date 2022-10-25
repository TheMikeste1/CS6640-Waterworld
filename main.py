from collections import defaultdict

import pettingzoo as pz

import custom_waterworld


default_args = {
    "n_pursuers": 2,
    "n_evaders": 5,
    "n_poisons": 10,
    "n_obstacles": 1,
    "n_coop": 1,
    "n_sensors": 30,
    "sensor_range": 0.2,
    "radius": 0.015,
    "obstacle_radius": 0.1,
    "obstacle_coord": [(0.5, 0.5)],
    "pursuer_max_accel": 0.02,
    "pursuer_speed": 0.2,
    "evader_speed": 0.1,
    "poison_speed": 0.1,
    "poison_reward": -1.0,
    "food_reward": 10.0,
    "encounter_reward": 0.01,
    "thrust_penalty": -0.5,
    "local_ratio": 1.0,
    "speed_features": True,
    "max_cycles": 500,
    "render_mode": None,
    "FPS": 15,
}


def run_episode(env: pz.AECEnv):
    rewards = defaultdict(list)
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        rewards[agent].append(reward)
        action = None if terminated or truncated else env.action_space(agent).sample()
        env.step(action)
        env.render()
    pass


if __name__ == "__main__":
    kwargs = default_args.copy()
    kwargs.update(
        {"render_mode": None, "FPS": 120, "n_poisons": 10, "thrust_penalty": 0}
    )
    env = custom_waterworld.waterworld.env(**kwargs)
    print(f"Running at {env.base_env.FPS} FPS")

    run_episode(env)

    env.close()
