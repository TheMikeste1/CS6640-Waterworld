from datetime import datetime

import torch.optim
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from agents import A2CAgent, ModuleBuilder, NeuralNetwork
from agents.distance_neural_network import DistanceNeuralNetwork
from custom_waterworld import Runner, WaterworldArguments
from main import record_episode, train
from run_arguments import RunArguments


def run(args: tuple) -> None:
    run_args: RunArguments = args[0]
    run_id: int = args[1]
    env = waterworld.env(**run_args.environment_args.to_dict())
    agents = run_args.compile_agents(env)
    runner = Runner(
        env,
        agents,
        enable_tqdm=False,
        tqdm_kwargs={"desc": f"Run {run_id}", "position": run_id, "leave": False},
    )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_writer: SummaryWriter | None = None
    tensorboard_writer = SummaryWriter(
        log_dir=f"runs/{date_time}_{run_args.run_name}_{run_args.num_episodes}its"
    )
    for env_name, agent in agents.items():
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
            run_args.num_episodes,
            name_prepend=date_time,
            tensorboard_writer=tensorboard_writer,
            verbose=False,
        )
        env.close()

        # Record an episode
        if run_args.should_record:
            for agent in runner.agents.values():
                agent.enable_explore = False
            record_episode(
                runner,
                record_name=f"recordings/"
                f"{date_time}_{run_args.run_name}_{run_args.num_episodes}its",
            )
    finally:
        env.close()
        if tensorboard_writer:
            tensorboard_writer.close()


def main():
    NUM_PROCESSES = 4

    num_sensors = 30
    run_args = [
        RunArguments(
            run_name="a2c_distance_vs_linear_batch1024",
            num_episodes=1024,
            environment_args=WaterworldArguments(
                render_mode=WaterworldArguments.RenderMode.NONE,
                max_cycles=512,
            ),
            should_record=True,
            agent_builders=[
                A2CAgent.Builder(
                    env_name="pursuer_0",
                    name="a2c_distance",
                    shared_network=DistanceNeuralNetwork.Builder(
                        distance_layers=[
                            ModuleBuilder(
                                factory=torch.nn.BatchNorm1d,
                                kwargs={"num_features": 5},
                            ),
                            ModuleBuilder(
                                factory=torch.nn.Conv1d,
                                kwargs={
                                    "in_channels": 5,
                                    "out_channels": 32,
                                    "kernel_size": 3,
                                    "padding": 1,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Conv1d,
                                kwargs={
                                    "in_channels": 32,
                                    "out_channels": 64,
                                    "kernel_size": 3,
                                    "padding": 1,
                                },
                            ),
                            ModuleBuilder(
                                factory=torch.nn.BatchNorm1d,
                                kwargs={"num_features": 64},
                            ),
                        ],
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    # out_channels * num_sensors + 2 collision
                                    # features +
                                    # 3 speed layers
                                    "in_features": 64 * num_sensors
                                    + 2
                                    + num_sensors * 3,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 64,
                                },
                            ),
                        ],
                        num_sensors=num_sensors,
                    ),
                    advantage_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 1,
                                },
                            ),
                        ]
                    ),
                    policy_networks=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 5,
                                },
                            ),
                        ]
                    ),
                    batch_size=1024,
                    memory=1024 * 3,
                    optimizer_factory=torch.optim.Adam,
                    optimizer_kwargs={"lr": 0.003},
                    criterion_factory=torch.nn.MSELoss,
                    lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.99},
                ),
                A2CAgent.Builder(
                    env_name="pursuer_1",
                    name="a2c_linear",
                    shared_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    # https://pettingzoo.farama.org/environments/sisl/waterworld/#observation-space
                                    "in_features": 8 * num_sensors + 2,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 64,
                                },
                            ),
                        ],
                    ),
                    advantage_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 1,
                                },
                            ),
                        ]
                    ),
                    policy_networks=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 5,
                                },
                            ),
                        ]
                    ),
                    batch_size=1024,
                    memory=1024 * 3,
                    optimizer_factory=torch.optim.Adam,
                    optimizer_kwargs={"lr": 0.003},
                    criterion_factory=torch.nn.MSELoss,
                    lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.99},
                ),
            ],
        ),
        RunArguments(
            run_name="a2c_distance_vs_linear_batch512",
            num_episodes=1024,
            environment_args=WaterworldArguments(
                render_mode=WaterworldArguments.RenderMode.NONE,
                max_cycles=512,
            ),
            should_record=True,
            agent_builders=[
                A2CAgent.Builder(
                    env_name="pursuer_0",
                    name="a2c_distance",
                    shared_network=DistanceNeuralNetwork.Builder(
                        distance_layers=[
                            ModuleBuilder(
                                factory=torch.nn.BatchNorm1d,
                                kwargs={"num_features": 5},
                            ),
                            ModuleBuilder(
                                factory=torch.nn.Conv1d,
                                kwargs={
                                    "in_channels": 5,
                                    "out_channels": 32,
                                    "kernel_size": 3,
                                    "padding": 1,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Conv1d,
                                kwargs={
                                    "in_channels": 32,
                                    "out_channels": 64,
                                    "kernel_size": 3,
                                    "padding": 1,
                                },
                            ),
                            ModuleBuilder(
                                factory=torch.nn.BatchNorm1d,
                                kwargs={"num_features": 64},
                            ),
                        ],
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    # out_channels * num_sensors + 2 collision
                                    # features +
                                    # 3 speed layers
                                    "in_features": 64 * num_sensors
                                    + 2
                                    + num_sensors * 3,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 64,
                                },
                            ),
                        ],
                        num_sensors=num_sensors,
                    ),
                    advantage_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 1,
                                },
                            ),
                        ]
                    ),
                    policy_networks=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 5,
                                },
                            ),
                        ]
                    ),
                    batch_size=512,
                    memory=512 * 3,
                    optimizer_factory=torch.optim.Adam,
                    optimizer_kwargs={"lr": 0.003},
                    criterion_factory=torch.nn.MSELoss,
                    lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.99},
                ),
                A2CAgent.Builder(
                    env_name="pursuer_1",
                    name="a2c_linear",
                    shared_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    # https://pettingzoo.farama.org/environments/sisl/waterworld/#observation-space
                                    "in_features": 8 * num_sensors + 2,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 256,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 256,
                                    "out_features": 64,
                                },
                            ),
                        ],
                    ),
                    advantage_network=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 1,
                                },
                            ),
                        ]
                    ),
                    policy_networks=NeuralNetwork.Builder(
                        layers=[
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 64,
                                },
                            ),
                            torch.nn.LeakyReLU,
                            ModuleBuilder(
                                factory=torch.nn.Linear,
                                kwargs={
                                    "in_features": 64,
                                    "out_features": 5,
                                },
                            ),
                        ]
                    ),
                    batch_size=512,
                    memory=512 * 3,
                    optimizer_factory=torch.optim.Adam,
                    optimizer_kwargs={"lr": 0.003},
                    criterion_factory=torch.nn.MSELoss,
                    lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
                    lr_scheduler_kwargs={"gamma": 0.99},
                ),
            ],
        ),
    ]

    NUM_PROCESSES = min(NUM_PROCESSES, len(run_args))
    run_args = [(r, i + 1) for i, r in enumerate(run_args)]
    print(f"Starting {len(run_args)} runs with {NUM_PROCESSES} processes")
    process_map(run, run_args, max_workers=NUM_PROCESSES, position=0)
    print("Done")


if __name__ == "__main__":
    # Run the main function
    main()
