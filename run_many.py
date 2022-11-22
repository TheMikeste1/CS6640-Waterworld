import multiprocessing
from datetime import datetime

import torch.optim
from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from agents import A2CAgent, DDPGAgent, ModuleBuilder, NeuralNetwork
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
        enable_tqdm=True,
        tqdm_kwargs={"desc": f"Run {run_id}", "position": 1, "leave": True},
    )

    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tensorboard_writer: SummaryWriter | None = None
    tensorboard_writer = SummaryWriter(
        log_dir=f"runs/{date_time}/{run_args.run_name}_{run_args.num_episodes}its"
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
                            f"{date_time}/{run_args.run_name}_"
                            f"{run_args.num_episodes}its",
            )
    finally:
        env.close()
        if tensorboard_writer:
            tensorboard_writer.close()


def main():
    NUM_PROCESSES = 4

    run_args: [RunArguments] = []
    run_builder = RunArguments.Builder()
    run_builder.set_num_episodes(2048).set_should_record(False)
    run_builder.set_environment_args(
        WaterworldArguments(
            max_cycles=256,
        )
    )

    for num_sensors in range(1, 31):
        run_builder.set_run_name(f"waterworld_{num_sensors}_sensors")
        run_builder.environment_args.n_sensors = num_sensors
        agent_builder = A2CAgent.Builder(
            env_name="pursuer_0",
            name=f"A2C_{num_sensors}",
            shared_network=NeuralNetwork.Builder(
                layers=[
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 8 * num_sensors + 2,
                            "out_features": (8 * num_sensors + 2) * 2,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": (8 * num_sensors + 2) * 2,
                            "out_features": 512,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 512,
                            "out_features": 512,
                        }
                    ),
                ],
            ),
            advantage_network=NeuralNetwork.Builder(
                layers=[
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 512,
                            "out_features": 512,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 512,
                            "out_features": 256,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 256,
                            "out_features": 1,
                        },
                    ),
                ],
            ),
            policy_networks=NeuralNetwork.Builder(
                layers=[
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 512,
                            "out_features": 512,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 512,
                            "out_features": 256,
                        },
                    ),
                    torch.nn.ReLU,
                    ModuleBuilder(
                        torch.nn.Linear,
                        kwargs={
                            "in_features": 256,
                            "out_features": 8,
                        },
                    ),
                    torch.nn.Softmax,
                ],
            ),
            optimizer_factory=torch.optim.Adam,
            optimizer_kwargs={"lr": 0.0001},
            criterion_factory=torch.nn.HuberLoss,
            criterion_kwargs={"reduction": "mean"},
            lr_scheduler_factory=torch.optim.lr_scheduler.ExponentialLR,
            lr_scheduler_kwargs={"gamma": 0.999},
            gamma=0.99,
            batch_size=128,
            memory=2048,
        )
        run_builder.add_agent_builder(agent_builder)
        run_builder.add_agent_builder(agent_builder)
        args = run_builder.build()
        run_args.append(args)

        NUM_PROCESSES = min(NUM_PROCESSES, len(run_args))
        enumerated_args = [(r, i + 1) for i, r in enumerate(run_args)]
        multiprocessing.freeze_support()
        print(f"Starting {len(run_args)} runs with {NUM_PROCESSES} processes")
        process_map(run, enumerated_args, max_workers=NUM_PROCESSES, position=0)
        print("Done")

        if __name__ == "__main__":
        # Run the main function
            main()
