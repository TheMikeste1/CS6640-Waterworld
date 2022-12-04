import torch

from agents import (
    CriticNetwork,
    DDPGAgent,
    DistanceNeuralNetwork,
    MemoryLSTM, ModuleBuilder,
    NeuralNetwork,
    QNNAgent,
    RewardPrioritizedMemory,
)
from custom_waterworld import WaterworldArguments
from run_arguments import RunArguments


def generate_ddpg_simple_lstm(
    num_episodes: int, env_args: WaterworldArguments, should_record: bool = False
) -> RunArguments:
    run_builder = (
        RunArguments.Builder()
        .set_num_episodes(num_episodes)
        .set_should_record(should_record)
        .set_environment_args(env_args)
        .set_run_name("ddpg_simple_lstm")
    )

    num_sensors = env_args.n_sensors
    speed_features = env_args.speed_features
    num_obs = num_sensors * (8 if speed_features else 5) + 2

    agent_builder = NeuralNetwork.Builder(
        [
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
                MemoryLSTM,
                kwargs={
                    "input_size": 64,
                    "hidden_size": 64,
                    "num_layers": 1,
                    "batch_first": True,
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
        ]
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

    normal_memory = DDPGAgent.Builder(
        env_name="pursuer_0",
        name="ddpg_simple_lstm_normal_memory",
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
    )
    run_builder.add_agent_builder(normal_memory)

    rp_memory = DDPGAgent.Builder(
        env_name="pursuer_1",
        name="ddpg_simple_lstm_rpm",
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
    )
    run_builder.add_agent_builder(rp_memory)
    return run_builder.build()
