import torch

from agents import (
    A2CAgent,
    CriticNetwork,
    DDPGAgent,
    DistanceNeuralNetwork,
    MemoryLSTM,
    ModuleBuilder,
    NeuralNetwork,
    QNNAgent,
    RewardPrioritizedMemory,
)
from custom_waterworld import WaterworldArguments
from run_arguments import RunArguments


def generate_a2c_simple_lstm(
    num_episodes: int, env_args: WaterworldArguments, should_record: bool = False
) -> RunArguments:
    run_builder = (
        RunArguments.Builder()
        .set_num_episodes(num_episodes)
        .set_should_record(should_record)
        .set_environment_args(env_args)
        .set_run_name("a2c_simple_lstm")
    )

    num_sensors = env_args.n_sensors
    speed_features = env_args.speed_features
    num_obs = num_sensors * (8 if speed_features else 5) + 2

    shared_builder = NeuralNetwork.Builder(
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
        ]
    )

    policy_builder = NeuralNetwork.Builder(
        [
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
                    "out_features": 5,
                },
            ),
            torch.nn.Tanh,
            ModuleBuilder(
                torch.nn.Threshold,
                kwargs={
                    "threshold": 0.0,
                    "value": 0.0,
                },
            ),
        ]
    )

    advantage_builder = NeuralNetwork.Builder(
        [
            ModuleBuilder(
                torch.nn.Linear,
                kwargs={
                    "in_features": 64,
                    "out_features": 1,
                },
            )
        ]
    )

    normal_memory = A2CAgent.Builder(
        env_name="pursuer_0",
        name="a2c_simple_normal_memory",
        batch_size=512,
        memory=512 * 3,
        shared_network=shared_builder,
        policy_networks=[policy_builder] * 2,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        advantage_network=advantage_builder,
        criterion_factory=torch.nn.MSELoss,
    )
    run_builder.add_agent_builder(normal_memory)

    rp_memory = A2CAgent.Builder(
        env_name="pursuer_1",
        name="a2c_simple_rpm",
        batch_size=512,
        memory=RewardPrioritizedMemory(512 * 3),
        shared_network=shared_builder,
        policy_networks=[policy_builder] * 2,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        advantage_network=advantage_builder,
        criterion_factory=torch.nn.MSELoss,
    )
    run_builder.add_agent_builder(rp_memory)
    return run_builder.build()
