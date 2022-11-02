from collections import defaultdict

import cv2
import pettingzoo as pz
from pettingzoo.sisl import waterworld_v4 as waterworld

from agents import RandomAgent
import custom_waterworld
from custom_waterworld import WaterworldArguments


class VideoWriter:
    def __init__(self, fps: int, width: int, height: int, filename: str):
        self.vw = cv2.VideoWriter(
            filename, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True
        )

    def write(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.vw.write(frame)

    def close(self):
        self.vw.release()


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
        render_mode=WaterworldArguments.RenderMode.RGB,
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

    # width, height = env.unwrapped.env.pixel_scale, env.unwrapped.env.pixel_scale
    # vw = VideoWriter(env.unwrapped.env.FPS, width, height, "test.mp4")
    # runner.subscribe_render(lambda x, y: vw.write(y))
    # runner.subscribe_post_episode(lambda *_: vw.close())

    # noinspection PyBroadException
    try:
        runner.run_episode()
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()
        # vw.close()


if __name__ == "__main__":
    main()
