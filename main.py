import os.path
import re
from collections import OrderedDict
from datetime import datetime
from functools import partial

import seaborn as sns
import torch.nn
import torchinfo
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import custom_waterworld
from agents import (
    A2CAgent,
    AbstractAgent,
    ControlsPolicyTrainer,
    CriticNetwork,
    DDPGAgent,
    MemoryLSTM,
    ModuleBuilder, RewardPrioritizedMemory,
)
from assembled_agents import *
from custom_waterworld import Runner, WaterworldArguments
from custom_waterworld.runner import REWARDS_TYPE
from run_builders.ddpg_distance import generate_ddpg_distance


def on_post_episode(writer: SummaryWriter, runner, it, rewards, agent_posts):
    for agent_name in agent_posts.keys():
        reward = rewards[agent_name]
        reward_sum = sum(reward)
        writer.add_scalar(f"{agent_name}/total_reward", reward_sum, it)


def on_post_train(writer: SummaryWriter, runner, it, agent_trains):
    for agent_name in agent_trains.keys():
        loss = agent_trains[agent_name]
        for key, value in loss.items():
            writer.add_scalar(f"{agent_name}/{key}", value, it)


def on_post_test(
    writer: SummaryWriter,
    runner: Runner,
    it: int,
    episode_rewards: dict[str, list[list[float]]],
):
    for agent_name, rewards in episode_rewards.items():
        rewards = [sum(reward) for reward in rewards]
        writer.add_scalar(f"{agent_name}/test_total_reward", sum(rewards), it)


def record_episode(
    runner: Runner,
    record_name: str,
    record_as_gif: bool = False,
    with_dataframe: bool = False,
    dataframe_name: str = None,
    explore: bool = False,
):
    # noinspection PyUnresolvedReferences
    env = runner.env.unwrapped.env
    # Run an episode to film
    previous_mode = env.render_mode
    env.render_mode = WaterworldArguments.RenderMode.RGB.value

    width, height = env.pixel_scale, env.pixel_scale
    if record_as_gif:
        visual_writer = custom_waterworld.GIFWriter(env.FPS, record_name)
    else:
        visual_writer = custom_waterworld.VideoWriter(
            env.FPS, width, height, record_name
        )

    write_callback = lambda _, frame: visual_writer.write(frame)
    runner.on_render += write_callback

    kwargs = {
        "train": False,
        "explore": explore,
    }

    try:
        if with_dataframe:
            if dataframe_name is None:
                dataframe_name = re.split("[\\/]", record_name)[-1]
            _, df = runner.run_episode_with_dataframe(**kwargs)
            if not os.path.exists("dataframes"):
                os.mkdir("dataframes")
            df.to_csv(f"dataframes/{dataframe_name}.csv")
        else:
            runner.run_episode(kwargs)
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

        write_post_test = partial(on_post_test, tensorboard_writer)
        runner.on_post_test_iterations += write_post_test

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
        runner.run_iterations(iterations, test_iterations=2, test_interval=2)
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
            runner.on_post_test_iterations -= write_post_test

    for agent in agents:
        torch.save(
            agent.state_dict(),
            f"models/{name_prepend}_{agent.name}_{agent.env_name}_{iterations}its.pt",
        )


def agent_effectiveness_test(agent: AbstractAgent, iterations: int, batch_size: int):
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
    ITERATIONS = 512 + 256
    BATCH_SIZE = 2048
    run_name = "DDPG_Distance_CPT"
    args = WaterworldArguments(
        # FPS=60
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
        n_pursuers=2,
        # n_evaders=5 * 3,
        # n_poisons=10 * 3,
    )
    env = waterworld.env(**args.to_dict())

    num_obs = env.observation_space(env.possible_agents[0]).shape[0]
    num_actions = env.action_space(env.possible_agents[0]).shape[0]
    num_sensors = args.n_sensors
    speed_features = args.speed_features
    # Create agents
    agent_builder = DistanceNeuralNetwork.Builder(
        num_sensors=num_sensors,
        has_speed_features=speed_features,
        layers=[
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    # out_channels * num_sensors + 2 collision features + 3 speed layers
                    "in_features": 64 * num_sensors
                                   + 2
                                   + num_sensors * (3 if speed_features else 0),
                    "out_features": 256,
                },
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 256,
                    "out_features": 64,
                },
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 64,
                    "out_features": 2,
                },
            ),
        ],
        distance_layers=[
            ModuleBuilder(
                torch.nn.BatchNorm1d,
                kwargs={
                    # There are 5 distance features
                    "num_features": 5,
                },
            ),
            ModuleBuilder(
                torch.nn.Conv1d,
                kwargs=dict(
                    in_channels=5,
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                ),
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Conv1d,
                kwargs=dict(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                ),
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.BatchNorm1d,
                kwargs={
                    "num_features": 64,
                },
            ),
        ],
    )

    critic_builder = CriticNetwork.Builder(
        obs_layers=[
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": num_obs,
                    "out_features": 256,
                },
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 256,
                    "out_features": 64,
                },
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 64,
                    "out_features": 64,
                },
            ),
        ],
        action_layers=[
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 2,
                    "out_features": 64,
                },
            ),
        ],
        layers=[
            torch.nn.Tanh,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 64 * 2,
                    "out_features": 64,
                },
            ),
            torch.nn.LeakyReLU,
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 64,
                    "out_features": 1,
                },
            ),
        ],
    )

    # pursuer_0 = ControlsPolicyTrainer(
    #     env,
    #     env_name="pursuer_0",
    #     name="cpt_normal_memory",
    #     batch_size=512,
    #     memory=512 * 3,
    #     critic=critic_builder.build(),
    #     critic_optimizer_factory=torch.optim.Adam,
    #     critic_optimizer_kwargs={"lr": 0.001},
    #     critic_lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
    #     critic_lr_scheduler_kwargs={"gamma": 0.99},
    #     criterion_factory=torch.nn.MSELoss,
    # )

    pursuer_0 = DDPGAgent.Builder(
        env_name="pursuer_0",
        name="ddpg_distance_normal_memory_cpt",
        batch_size=512,
        memory=512 * 3,
        actor=agent_builder,
        actor_optimizer_factory=torch.optim.Adam,
        actor_optimizer_kwargs={"lr": 0.001},
        actor_lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        actor_lr_scheduler_kwargs={"gamma": 0.99},
        critic=critic_builder,
        critic_optimizer_factory=torch.optim.Adam,
        critic_optimizer_kwargs={"lr": 0.001},
        critic_lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        critic_lr_scheduler_kwargs={"gamma": 0.99},
        criterion_factory=torch.nn.MSELoss,
    ).build(env)

    cpt_params = torch.load(
        "models/keep/2022-12-03_18-51-03_cpt_normal_memory_pursuer_0_768its.pt"
    )
    critic_params = OrderedDict()
    target_critic_params = OrderedDict()
    for key, item in cpt_params.items():
        if key.startswith("critic"):
            key = key[len("critic."):]
            critic_params[key] = item
        elif key.startswith("target_critic"):
            key = key[len("target_critic."):]
            target_critic_params[key] = item

    pursuer_0.critic.load_state_dict(critic_params)
    pursuer_0.target_critic.load_state_dict(target_critic_params)

    # WARNING: This will exit the program
    # agent_effectiveness_test(pursuer_0, 512, BATCH_SIZE)

    pursuer_0.should_explore = False
    torchinfo.summary(
        pursuer_0, input_size=(BATCH_SIZE, num_obs), device=pursuer_0.device, depth=5
    )
    pursuer_0.should_explore = True
    pursuer_0.reset()

    pursuer_1 = DDPGAgent.Builder(
        env_name="pursuer_1",
        name="ddpg_distance_rpm_cpt",
        batch_size=512,
        memory=RewardPrioritizedMemory(512 * 3),
        actor=agent_builder,
        actor_optimizer_factory=torch.optim.Adam,
        actor_optimizer_kwargs={"lr": 0.001},
        actor_lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        actor_lr_scheduler_kwargs={"gamma": 0.99},
        critic=critic_builder,
        critic_optimizer_factory=torch.optim.Adam,
        critic_optimizer_kwargs={"lr": 0.001},
        critic_lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        critic_lr_scheduler_kwargs={"gamma": 0.99},
        criterion_factory=torch.nn.MSELoss,
    ).build(env)
    cpt_params = torch.load(
        "models/keep/2022-12-03_18-51-03_cpt_normal_memory_pursuer_0_768its.pt"
    )
    critic_params = OrderedDict()
    target_critic_params = OrderedDict()
    for key, item in cpt_params.items():
        if key.startswith("critic"):
            key = key[len("critic."):]
            critic_params[key] = item
        elif key.startswith("target_critic"):
            key = key[len("target_critic."):]
            target_critic_params[key] = item

    pursuer_1.critic.load_state_dict(critic_params)
    pursuer_1.target_critic.load_state_dict(target_critic_params)

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
        log_dir=f"runs/{date_time}_{run_name}_{ITERATIONS}its"
    )
    for env_name, agent in agents.items():
        agent_configs = f"name: {agent.name},\n"
        if hasattr(agent, "batch_size"):
            agent_configs += f"batch_size: {agent.batch_size},\n"
        if hasattr(agent, "memory"):
            agent_configs += f"memory: {len(agent.memory)},\n"
        if hasattr(agent, "gamma"):
            agent_configs += f"gamma: {agent.gamma},\n"
        if hasattr(agent, "optimizer"):
            agent_configs += f"optimizer: {agent.optimizer},\n"
        if hasattr(agent, "criterion"):
            agent_configs += f"criterion: {agent.criterion},\n"
        if hasattr(agent, "lr_scheduler"):
            agent_configs += f"lr_scheduler: {agent.lr_scheduler},\n"
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
            agent.should_explore = False
        print("Recording an episode. . .")
        record_episode(
            runner,
            record_name=f"recordings/{date_time}_{run_name}_{ITERATIONS}its",
            record_as_gif=False,
            with_dataframe=False,
            explore=False,
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
