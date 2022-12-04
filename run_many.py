import multiprocessing
from datetime import datetime

from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from custom_waterworld import Runner, WaterworldArguments
from main import record_episode, train
from run_arguments import RunArguments
from run_builders.a2c_distance import generate_a2c_distance
from run_builders.a2c_distance_lstm import generate_a2c_distance_lstm
from run_builders.a2c_simple import generate_a2c_simple
from run_builders.a2c_simple_lstm import generate_a2c_simple_lstm
from run_builders.controls import generate_controls_run
from run_builders.ddpg_distance import generate_ddpg_distance
from run_builders.ddpg_distance_lstm import generate_ddpg_distance_lstm
from run_builders.ddpg_simple import generate_ddpg_simple
from run_builders.ddpg_simple_lstm import generate_ddpg_simple_lstm
from run_builders.qnn_distance import generate_qnn_distance
from run_builders.qnn_distance_lstm import generate_qnn_distance_lstm
from run_builders.qnn_simple import generate_qnn_simple
from run_builders.qnn_simple_lstm import generate_qnn_simple_lstm


def run(args: tuple) -> None:
    run_args: RunArguments = args[0]
    run_id: int = args[1]
    prepend: str = args[2]

    env = waterworld.env(**run_args.environment_args.to_dict())
    agents = run_args.compile_agents(env)
    runner = Runner(
        env,
        agents,
        enable_tqdm=True,
        tqdm_kwargs={"desc": f"Run {run_id}", "position": run_id, "leave": True},
    )

    tensorboard_writer: SummaryWriter | None = None
    tensorboard_writer = SummaryWriter(
        log_dir=f"runs/{prepend}/{run_args.run_name}_{run_args.num_episodes}its"
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
            run_args.num_episodes,
            name_prepend=prepend,
            tensorboard_writer=tensorboard_writer,
            verbose=False,
        )
        env.close()

        # Record an episode
        if run_args.should_record:
            record_episode(
                runner,
                record_name=f"recordings/"
                f"{prepend}/{run_args.run_name}_"
                f"{run_args.num_episodes}its",
                explore=False,
                with_dataframe=False,
                record_as_gif=False,
            )
    finally:
        env.close()
        if tensorboard_writer:
            tensorboard_writer.close()


def main():
    NUM_PROCESSES = 4


    num_episodes = 256
    should_record = False
    env_args = WaterworldArguments(
        render_mode=WaterworldArguments.RenderMode.NONE,
        max_cycles=512,
    )
    run_args: [RunArguments] = [
        generate_controls_run(num_episodes, env_args, should_record),

        generate_qnn_simple(num_episodes, env_args, should_record),
        generate_qnn_simple_lstm(num_episodes, env_args, should_record),
        generate_qnn_distance(num_episodes, env_args, should_record),
        generate_qnn_distance_lstm(num_episodes, env_args, should_record),

        generate_ddpg_simple(num_episodes, env_args, should_record),
        generate_ddpg_simple_lstm(num_episodes, env_args, should_record),
        generate_ddpg_distance(num_episodes, env_args, should_record),
        generate_ddpg_distance_lstm(num_episodes, env_args, should_record),

        generate_a2c_simple(num_episodes, env_args, should_record),
        generate_a2c_simple_lstm(num_episodes, env_args, should_record),
        generate_a2c_distance(num_episodes, env_args, should_record),
        generate_a2c_distance_lstm(num_episodes, env_args, should_record),
    ]

    now = datetime.now()
    prepend = now.strftime("%Y-%m-%d_%H-%M-%S")
    enumerated_args = [(r, i + 1, prepend) for i, r in enumerate(run_args)]
    multiprocessing.freeze_support()
    print(f"Starting {len(run_args)} runs with up to {NUM_PROCESSES} processes")
    process_map(run, enumerated_args, max_workers=NUM_PROCESSES, position=0)

    time_elapsed = datetime.now() - now
    print(f"Done after {time_elapsed}")


if __name__ == "__main__":
    # Run the main function
    main()
