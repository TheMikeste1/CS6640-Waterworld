from datetime import datetime
from functools import partial

import torch.nn
import torchinfo
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter

import custom_waterworld
from agents import NeuralNetwork, QNNAgent
from agents.DistanceNeuralNetwork import DistanceNeuralNetwork
from custom_waterworld import Runner, WaterworldArguments


def on_post_episode(writer: SummaryWriter, runner, it, rewards, agent_posts):
    for agent_name in agent_posts.keys():
        reward = rewards[agent_name]
        reward_sum = 0
        for i, r in enumerate(reward):
            writer.add_scalar(f"{agent_name}/reward_lifetime", r, i)
            reward_sum += r
        writer.add_scalar(f"{agent_name}/reward", reward_sum, it)


def on_post_train(writer: SummaryWriter, runner, it, agent_trains):
    for agent_name in agent_trains.keys():
        loss = agent_trains[agent_name]
        writer.add_scalar(f"{agent_name}/loss", loss, it)
        model = runner.agents[agent_name]


def record_episode(runner: Runner, video_name: str):
    # noinspection PyUnresolvedReferences
    env = runner.env.unwrapped.env
    # Run an episode to film
    previous_mode = env.render_mode
    env.render_mode = WaterworldArguments.RenderMode.RGB.value

    width, height = env.pixel_scale, env.pixel_scale
    if not video_name.endswith(".mp4"):
        video_name += ".mp4"
    visual_writer = custom_waterworld.VideoWriter(env.FPS, width, height, video_name)
    # visual_writer = custom_waterworld.GIFWriter(env.FPS, "test.gif")
    runner.on_render += lambda x, y: visual_writer.write(y)
    runner.on_post_episode += lambda *_: visual_writer.close()

    for agent in runner.agents.values():
        agent.enable_explore = False

    try:
        runner.run_episode(train=False)
    finally:
        visual_writer.close()
        env.render_mode = previous_mode


def train(runner: Runner, iterations: int, name_append: str = ""):
    # noinspection PyGlobalUndefined
    global on_post_episode, on_post_train

    # noinspection PyUnresolvedReferences
    env = runner.env.unwrapped.env

    tensorboard_writer = SummaryWriter()
    write_post_episode = partial(on_post_episode, tensorboard_writer)
    runner.on_post_episode += write_post_episode

    write_post_train = partial(on_post_train, tensorboard_writer)
    runner.on_post_train += write_post_train

    agent = next(iter(runner.agents.values()))

    if env.render_mode == WaterworldArguments.RenderMode.HUMAN.value:
        print(f"Running at {env.unwrapped.env.FPS} FPS on {agent.device}")
    else:
        print(f"Running in the background on {agent.device}")
    print("", end="", flush=True)

    agents = set(runner.agents.values())
    # Train
    try:
        runner.run_iterations(iterations)
    except KeyboardInterrupt:
        print("Run interrupted")
        for agent in agents:
            torch.save(
                agent.state_dict(), f"models/interrupted/{agent.name}_{name_append}.pt"
            )
        return
    finally:
        env.close()
        tensorboard_writer.close()
        runner.on_post_episode -= write_post_episode
        runner.on_post_train -= write_post_train

    for agent in agents:
        torch.save(
            agent.state_dict(), f"models/{agent.name}_{iterations}_{name_append}.pt"
        )


def main():
    ITERATIONS = 64
    args = WaterworldArguments(
        FPS=60,
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
        # n_evaders=20 * 3,
        # n_poisons=20 * 3,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]
    num_sensors = args.n_sensors
    # Create agents
    agent_name = "qnn_distance"
    policy_networks = [
        DistanceNeuralNetwork(
            layers=[
                # out_channels * num_sensors + 2 collision features + 3 speed layers
                torch.nn.Linear(32 * num_sensors + 2 + num_sensors * 3, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 3),
            ],
            distance_layers=[
                torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.ReLU(),
                torch.nn.Conv1d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                ),
            ],
            speed_features=True,
            num_sensors=num_sensors,
        )
        for _ in range(2)
    ]

    pursuer_0 = QNNAgent(
        env,
        "pursuer_0",
        name=agent_name,
        policy_models=policy_networks,
        batch_size=512,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        criterion_factory=torch.nn.SmoothL1Loss,
        criterion_kwargs={},
        lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
    )
    pursuer_0.enable_explore = False
    torchinfo.summary(
        pursuer_0, input_size=(1, num_obs), device=pursuer_0.device, depth=5
    )
    pursuer_0.enable_explore = True

    pursuer_1 = pursuer_0

    runner = Runner(
        env,
        agents={
            "pursuer_0": pursuer_0,
            "pursuer_1": pursuer_1,
        },
        should_render_empty=args.render_mode == WaterworldArguments.RenderMode.HUMAN,
    )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        train(runner, ITERATIONS, name_append=date_time)
    except KeyboardInterrupt:
        print("Run interrupted")
        return
    finally:
        env.close()

    try:
        record_episode(runner, video_name=f"videos/{agent_name}_{date_time}.mp4")
    except KeyboardInterrupt:
        print("Run interrupted")
    finally:
        env.close()


if __name__ == "__main__":
    main()
