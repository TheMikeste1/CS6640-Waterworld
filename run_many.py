from datetime import datetime

from pettingzoo.sisl import waterworld_v4 as waterworld
from torch.utils.tensorboard import SummaryWriter
from tqdm.contrib.concurrent import process_map

from custom_waterworld import Runner
from main import record_episode, train
from run_arguments import RunArguments


def run(run_args: RunArguments):
    env = waterworld.env(**run_args.environment_args.to_dict())
    agents = run_args.compile_agents(env)
    runner = Runner(env, agents)

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

    run_args = [i for i in range(NUM_PROCESSES**2)]

    print(f"Starting {len(run_args)} runs with {NUM_PROCESSES} processes")
    process_map(run, run_args, max_workers=NUM_PROCESSES)
    print("Done")


if __name__ == "__main__":
    # Run the main function
    main()
