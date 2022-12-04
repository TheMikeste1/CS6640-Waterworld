import torch

from agents import (
    DistanceNeuralNetwork,
    ModuleBuilder,
    NeuralNetwork,
    QNNAgent,
    RewardPrioritizedMemory,
)
from custom_waterworld import WaterworldArguments
from run_arguments import RunArguments


def generate_qnn_distance(
    num_episodes: int, env_args: WaterworldArguments, should_record: bool = False
) -> RunArguments:
    run_builder = (
        RunArguments.Builder()
        .set_num_episodes(num_episodes)
        .set_should_record(should_record)
        .set_environment_args(env_args)
        .set_run_name("qnn_distance")
    )

    num_sensors = env_args.n_sensors
    speed_features = env_args.speed_features
    network_builder = DistanceNeuralNetwork.Builder(
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

    qnn_normal_memory = QNNAgent.Builder(
        env_name="pursuer_0",
        name="qnn_distance_normal_memory",
        batch_size=512,
        memory=512 * 3,
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        criterion_factory=torch.nn.MSELoss,
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        policy_networks=[network_builder] * 2,
    )
    run_builder.add_agent_builder(qnn_normal_memory)

    qnn_rp_memory = QNNAgent.Builder(
        env_name="pursuer_1",
        name="qnn_distance_rpm",
        batch_size=512,
        memory=RewardPrioritizedMemory(512 * 3),
        optimizer_factory=torch.optim.Adam,
        optimizer_kwargs={"lr": 0.001},
        criterion_factory=torch.nn.MSELoss,
        lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
        lr_scheduler_kwargs={"gamma": 0.99},
        policy_networks=[network_builder] * 2,
    )
    run_builder.add_agent_builder(qnn_rp_memory)
    return run_builder.build()
