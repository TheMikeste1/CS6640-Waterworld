from collections import defaultdict

import cv2
import pettingzoo as pz
from pettingzoo.sisl import waterworld_v4 as waterworld

from agents import RandomAgent
import custom_waterworld
from custom_waterworld import WaterworldArguments


def run_episode_with_video(env: pz.AECEnv, agents, memory):
    if env.unwrapped.env.render_mode != WaterworldArguments.RenderMode.RGB.value:
        raise ValueError("Video recording only works with RGB render mode")

    rewards = defaultdict(list)
    env.reset()
    out = env.render()
    fps = env.unwrapped.env.FPS
    vw = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, out.shape[:-1], True
    )
    vw.write(out)
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        rewards[agent].append(reward)
        action = None if terminated or truncated else env.action_space(agent).sample()
        env.step(action)
        out = env.render()
        vw.write(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    vw.release()


def run_iteration(
    env: pz.AECEnv, agents, memory, criterion, lr_scheduler, batch_size: int
):
    # Run an epoch
    # run_episode(env, agents, memory)

    # Perform updates
    error = {}
    for agent in agents:
        loss = 0
        for _ in range(64):
            memory_batch = memory.get_batch(batch_size)
            loss += agent.update(memory_batch)
        error[agent] = loss / 64
    lr_scheduler.step()


def main():
    args = WaterworldArguments(
        FPS=60,
        render_mode=WaterworldArguments.RenderMode.HUMAN,
        n_poisons=128,
        max_cycles=512,
    )
    env = waterworld.env(**args.to_dict())
    runner = custom_waterworld.Runner(
        env,
        agents={
            "pursuer_0": RandomAgent(env),
            "pursuer_1": RandomAgent(env),
        },
    )
    print(f"Running at {env.unwrapped.env.FPS} FPS")

    agents = []
    memory = None
    criterion = None
    lr_scheduler = None
    batch_size = 1

    # noinspection PyBroadException
    try:
        runner.run_episode()
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
