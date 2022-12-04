import torch

from agents import ModuleBuilder, NeuralNetwork, QNNAgent, RewardPrioritizedMemory
from custom_waterworld import WaterworldArguments
from run_arguments import RunArguments


def generate_qnn_simple(
    num_episodes: int, env_args: WaterworldArguments, should_record: bool = False
) -> RunArguments:
    run_builder = (
        RunArguments.Builder()
        .set_num_episodes(num_episodes)
        .set_should_record(should_record)
        .set_environment_args(env_args)
        .set_run_name("qnn_simple")
    )

    num_sensors = env_args.n_sensors
    num_obs = num_sensors * (8 if env_args.speed_features else 5) + 2
    network_builder = NeuralNetwork.Builder(
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

    qnn_normal_memory = QNNAgent.Builder(
        env_name="pursuer_0",
        name="qnn_simple_normal_memory",
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
        env_name="pursuer_0",
        name="qnn_simple_rpm",
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
