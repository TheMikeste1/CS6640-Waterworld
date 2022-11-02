from collections import defaultdict

import cv2
import pettingzoo as pz
import torch.nn
from pettingzoo.sisl import waterworld_v4 as waterworld

import custom_waterworld
from agents.nn_agent import NNAgent
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
        max_cycles=512,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]

    class SimpleNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(num_obs, num_obs**2),
                    torch.nn.Linear(num_obs**2, 2),
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    # Create agents
    network = SimpleNN()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    agent = NNAgent(
        env,
        model=network,
        optimizer=optimizer,
        loss=criterion,
        lr_scheduler=lr_scheduler,
    )

    runner = custom_waterworld.Runner(
        env,
        agents={
            "pursuer_0": agent,
            "pursuer_1": agent,
        },
    )
    if args.render_mode == WaterworldArguments.RenderMode.HUMAN:
        print(f"Running at {env.unwrapped.env.FPS} FPS")
    else:
        print(f"Running in the background")

    runner.on_step += lambda r, name, data: agent.memory.add(
        (data.state, data.action, data.reward, data.next_state, data.terminated)
    )
    runner.on_post_episode += lambda *_: agent.update(64)

    # width, height = env.unwrapped.env.pixel_scale, env.unwrapped.env.pixel_scale
    # vw = VideoWriter(env.unwrapped.env.FPS, width, height, "test.mp4")
    # runner.on_render += lambda x, y: vw.write(y)
    # runner.on_post_episode += lambda *_: vw.close()

    # noinspection PyBroadException
    try:
        runner.run_iterations(8)
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()
        # vw.close()


if __name__ == "__main__":
    main()
