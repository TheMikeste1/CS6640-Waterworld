from collections import defaultdict

import pettingzoo as pz

import custom_waterworld
from custom_waterworld.waterworld_arguments import WaterworldArguments


def run_episode(env: pz.AECEnv):
    rewards = defaultdict(list)
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        rewards[agent].append(reward)
        action = None if terminated or truncated else env.action_space(agent).sample()
        env.step(action)
        env.render()


def main():
    args = WaterworldArguments(
        FPS=60,
        render_mode=WaterworldArguments.RenderMode.HUMAN,
    )
    env = custom_waterworld.waterworld.env(**args.to_dict())
    print(f"Running at {env.base_env.FPS} FPS")

    # noinspection PyBroadException
    try:
        run_episode(env)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
