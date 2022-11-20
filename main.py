import os.path
from datetime import datetime
from functools import partial

import seaborn as sns
import torch.nn
import torchinfo
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import custom_waterworld
from agents import AbstractAgent, NeuralNetwork, QNNAgent
from agents.a2c import A2CAgent
from agents.distance_neural_network import DistanceNeuralNetwork
from agents.do_nothing_agent import DoNothingAgent
from agents.human_agent import HumanAgent
from assembled_agents import *
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
    except KeyboardInterrupt as e:
        if verbose:
            print("Run interrupted")
        if not os.path.exists("models/interrupted"):
            os.mkdir("models/interrupted")
        for agent in agents:
            torch.save(
                agent.state_dict(),
                f"models/interrupted/{name_prepend}_{agent.name}_{agent.env_name}.pt",
            )
        raise e from None
    finally:
        env.close()
        if tensorboard_writer:
            tensorboard_writer.close()
            runner.on_post_episode -= write_post_episode
            runner.on_post_train -= write_post_train

    for agent in agents:
        torch.save(
            agent.state_dict(),
            f"models/{name_prepend}_{agent.name}_{agent.env_name}_{iterations}its.pt",
        )


def test_agent_effectiveness(agent: AbstractAgent, iterations: int, batch_size: int):
    device = agent.device
    losses = []
    power = 3
    print(f"Testing {agent.name} on {agent.env_name} for {iterations} iterations")
    for _ in tqdm(range(iterations)):
        test_input = torch.rand(
            size=(batch_size, 1, agent.in_features), dtype=torch.float
        ).to(device)
        test_output, *_ = agent.forward(test_input)

        # policy to always take a specified action
        y = torch.ones(
            size=(
                batch_size,
                agent.env.action_space(agent.env_name).shape[0],
                agent.out_features,
            ),
            dtype=test_output.dtype,
        ).to(device)
        assert test_output.shape == y.shape

        # Choose a power we want to learn
        y *= power

        # Calculate the loss
        loss = agent.apply_loss(test_output, y)
        losses.append(loss)
    test_input = torch.rand(size=(1, 1, agent.in_features), dtype=torch.float).to(
        device
    )
    out, *_ = agent.forward(test_input)
    print(
        f"Policy out (should be near {1 / agent.out_features :.5f}): "
        f"{out.mean().item()}"
    )
    print(f"Average loss: {sum(losses) / len(losses)}")
    print(f"Max loss: {max(losses)}")
    print(f"Min loss: {min(losses)}")
    plot = sns.lineplot(x=range(len(losses)), y=losses)
    plot.figure.show()
    exit(0)


def main():
    ITERATIONS = 512
    BATCH_SIZE = 1024
    agent_name = "qnn_distance_with_logsoftmax"
    args = WaterworldArguments(
        # FPS=60
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
        # n_evaders=5 * 3,
        # n_poisons=10 * 3,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]
    num_actions = env.action_space(env.possible_agents[0]).shape[0]
    num_sensors = args.n_sensors
    # Create agents
    shared_network = generate_qnn_distance(num_sensors)
    advantage_network = NeuralNetwork(
        layers=[
            torch.nn.Linear(shared_network.out_features, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        ]
    )
    policy_networks = [
        NeuralNetwork(
            layers=[
                torch.nn.Linear(shared_network.out_features, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 5),
            ]
        )
        for _ in range(num_actions)
    ]

    pursuer_0 = A2CAgent(
        env,
        "pursuer_0",
        name=agent_name,
        shared_network=shared_network,
        advantage_network=advantage_network,
        policy_networks=policy_networks,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        criterion_factory=torch.nn.HuberLoss,
        criterion_kwargs={"reduction": "mean"},
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        batch_size=BATCH_SIZE,
        memory=BATCH_SIZE * 2,
        gamma=0.99,
    )

    # WARNING: This will exit the program
    # test_agent_effectiveness(pursuer_0, 512, BATCH_SIZE)

    pursuer_0.enable_explore = False
    torchinfo.summary(
        pursuer_0, input_size=(BATCH_SIZE, num_obs), device=pursuer_0.device, depth=5
    )
    pursuer_0.enable_explore = True

    pursuer_1 = A2CAgent(
        env,
        "pursuer_1",
        name=agent_name,
        shared_network=shared_network,
        advantage_network=advantage_network,
        policy_networks=policy_networks,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 1e-3},
        criterion_factory=torch.nn.HuberLoss,
        criterion_kwargs={"reduction": "mean"},
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        batch_size=BATCH_SIZE,
        memory=BATCH_SIZE * 2,
        gamma=0.99,
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
    tensorboard_writer: SummaryWriter | None = None
    tensorboard_writer = SummaryWriter(
        log_dir=f"runs/{date_time}_{agent_name}_{ITERATIONS}its"
    )
    for env_name, agent in agents.items():
        # was_exploring = agent.enable_explore
        # agent.enable_explore = False
        # tensorboard_writer.add_graph(agent, torch.rand(size=(num_obs,)))
        # agent.enable_explore = was_exploring
        agent_configs = (
            f"name: {agent.name},\n"
            f"batch_size: {agent.batch_size},\n"
            f"memory: {len(agent.memory)},\n"
            f"gamma: {agent.gamma},\n"
            f"optimizer: {agent.optimizer},\n"
            f"criterion: {agent.criterion},\n"
            f"lr_schedule: {agent.lr_scheduler},\n"
        )
        tensorboard_writer.add_text(f"{env_name}/config", agent_configs)

    try:
        train(
            runner,
            ITERATIONS,
            name_prepend=date_time,
            tensorboard_writer=tensorboard_writer,
        )
        env.close()

        # Record an episode
        for agent in runner.agents.values():
            agent.enable_explore = False
        print("Recording an episode. . .")
        record_episode(
            runner,
            record_name=f"recordings/{date_time}_{agent_name}_{ITERATIONS}its",
        )
    except KeyboardInterrupt:
        print("Run interrupted")
        return
    finally:
        env.close()
        if tensorboard_writer:
            tensorboard_writer.close()


if __name__ == "__main__":
    main()
