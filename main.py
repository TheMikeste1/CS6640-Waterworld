import os.path
from datetime import datetime
from functools import partial

import torch.nn
import torchinfo
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter

import custom_waterworld
from agents import AbstractAgent, NeuralNetwork, QNNAgent
from agents.distance_neural_network import DistanceNeuralNetwork
from agents.do_nothing_agent import DoNothingAgent
from agents.human_agent import HumanAgent
from custom_waterworld import Runner, WaterworldArguments


def on_post_episode(writer: SummaryWriter, runner, it, rewards, agent_posts):
    for agent_name in agent_posts.keys():
        reward = rewards[agent_name]
        reward_sum = sum(reward)
        writer.add_scalar(f"{agent_name}/total_reward", reward_sum, it)


def on_post_train(writer: SummaryWriter, runner, it, agent_trains):
    for agent_name in agent_trains.keys():
        loss = agent_trains[agent_name]
        writer.add_scalar(f"{agent_name}/loss", loss, it)


def record_episode(runner: Runner, record_name: str):
    # noinspection PyUnresolvedReferences
    env = runner.env.unwrapped.env
    # Run an episode to film
    previous_mode = env.render_mode
    env.render_mode = WaterworldArguments.RenderMode.RGB.value

    width, height = env.pixel_scale, env.pixel_scale
    visual_writer = custom_waterworld.VideoWriter(env.FPS, width, height, record_name)
    # visual_writer = custom_waterworld.GIFWriter(env.FPS, "video_name")

    write_callback = lambda _, frame: visual_writer.write(frame)
    runner.on_render += write_callback

    try:
        runner.run_episode(train=False)
    finally:
        visual_writer.close()
        runner.on_render -= write_callback
        env.render_mode = previous_mode


def train(
    runner: Runner,
    iterations: int,
    name_prepend: str = "",
    verbose: bool = True,
    tensorboard_writer: SummaryWriter = None,
):
    # noinspection PyGlobalUndefined
    global on_post_episode, on_post_train

    # noinspection PyUnresolvedReferences
    env = runner.env.unwrapped.env

    agents = set(runner.agents.values())
    agent = next(iter(agents))

    if tensorboard_writer:
        write_post_episode = partial(on_post_episode, tensorboard_writer)
        runner.on_post_episode += write_post_episode

        write_post_train = partial(on_post_train, tensorboard_writer)
        runner.on_post_train += write_post_train

    if verbose:
        if env.render_mode == WaterworldArguments.RenderMode.HUMAN.value:
            print(f"Running at {env.FPS} FPS")
        else:
            print(f"Running in the background on {agent.device}")
        print("", end="", flush=True)

    if not os.path.exists("models"):
        os.mkdir("models")

    # Train
    try:
        runner.run_iterations(iterations)
    except KeyboardInterrupt:
        if verbose:
            print("Run interrupted")
        if not os.path.exists("models/interrupted"):
            os.mkdir("models/interrupted")
        for agent in agents:
            torch.save(
                agent.state_dict(),
                f"models/interrupted/{name_prepend}_{agent.name}_{agent.env_name}.pt",
            )
        return
    finally:
        env.close()
        tensorboard_writer.close()
        if tensorboard_writer:
            runner.on_post_episode -= write_post_episode
            runner.on_post_train -= write_post_train

    for agent in agents:
        torch.save(
            agent.state_dict(),
            f"models/{name_prepend}_{agent.name}_{agent.env_name}_{iterations}its.pt",
        )


def test_agent_effectiveness(agent: AbstractAgent, iterations: int, batch_size: int):
    device = agent.device
    for _ in range(iterations):
        test_input = torch.rand(
            size=(batch_size, agent.in_features), dtype=torch.float
        ).to(device)
        _, test_output = agent(test_input)
        test_output = torch.from_numpy(test_output).to(device).float()

        # policy to always take a specified action
        y = torch.ones(
            size=(batch_size, agent.env.action_space(agent.env_name).shape[0]),
            dtype=test_output.dtype,
        ).to(device)
        assert test_output.shape == y.shape

        # Choose a power we want to learn
        y *= 3

        # Calculate the loss
        agent.apply_loss(test_output, y)
    test_input = torch.rand(size=(1, agent.in_features), dtype=torch.float).to(device)
    out = agent(test_input)
    print(out)


def main():
    ITERATIONS = 128
    BATCH_SIZE = 4096
    agent_name = "qnn_distance"
    args = WaterworldArguments(
        # FPS=60
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
        # n_evaders=5 * 3,
        # n_poisons=10 * 3,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]
    num_sensors = args.n_sensors
    # Create agents
    policy_networks = [
        DistanceNeuralNetwork(
            layers=[
                # out_channels * num_sensors + 2 collision features + 3 speed layers
                torch.nn.Linear(64 * num_sensors + 2 + num_sensors * 3, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 256),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(256, 3),
            ],
            distance_layers=[
                torch.nn.BatchNorm1d(5),
                torch.nn.Conv1d(
                    in_channels=5,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.LeakyReLU(),
                torch.nn.Conv1d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                ),
                torch.nn.BatchNorm1d(64),
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
        batch_size=BATCH_SIZE,
        memory=BATCH_SIZE * 2,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.0001},
        criterion_factory=torch.nn.CrossEntropyLoss,
        criterion_kwargs={},
        lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
    )
    # test_agent_effectiveness(pursuer_0, 256, BATCH_SIZE)
    # exit(0)
    pursuer_0.enable_explore = False
    # torchinfo.summary(
    #     pursuer_0, input_size=(BATCH_SIZE, num_obs), device=pursuer_0.device, depth=5
    # )
    pursuer_0.enable_explore = True

    pursuer_1 = QNNAgent(
        env,
        "pursuer_1",
        name=agent_name,
        policy_models=policy_networks,
        batch_size=BATCH_SIZE,
        memory=BATCH_SIZE * 2,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.0001},
        criterion_factory=torch.nn.CrossEntropyLoss,
        criterion_kwargs={},
        lr_scheduler_factory=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.99},
    )

    agents = {
        "pursuer_0": pursuer_0,
        "pursuer_1": pursuer_1,
    }

    runner = Runner(
        env,
        agents=agents,
        should_render_empty=args.render_mode == WaterworldArguments.RenderMode.HUMAN,
    )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_writer = SummaryWriter(
        log_dir=f"runs/{date_time}_{agent_name}_{ITERATIONS}its"
    )
    for env_name, agent in agents.items():
        was_exploring = agent.enable_explore
        agent.enable_explore = False
        tensorboard_writer.add_graph(agent, torch.rand(size=(num_obs,)))
        agent.enable_explore = was_exploring
        tensorboard_writer.add_text(f"{env_name}/name", agent.name)
        tensorboard_writer.add_text(f"{env_name}/batch_size", str(agent.batch_size))
        tensorboard_writer.add_text(f"{env_name}/memory", str(len(agent.memory)))
        tensorboard_writer.add_text(f"{env_name}/optimizer", str(agent.optimizer))
        tensorboard_writer.add_text(f"{env_name}/criterion", str(agent.criterion))
        tensorboard_writer.add_text(f"{env_name}/lr_scheduler", str(agent.lr_scheduler))
    exit(0)
    try:
        train(runner, ITERATIONS, name_prepend=date_time)
        env.close()

        # Record an episode
        for agent in runner.agents.values():
            agent.enable_explore = False
        record_episode(
            runner,
            record_name=f"recordings/{date_time}_{agent_name}_{ITERATIONS}its",
        )
    except KeyboardInterrupt:
        print("Run interrupted")
        return
    finally:
        env.close()


if __name__ == "__main__":
    main()
