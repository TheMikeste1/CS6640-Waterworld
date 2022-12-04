from agents import ControlsAgent
from custom_waterworld import WaterworldArguments
from run_arguments import RunArguments


def generate_controls_run(
    num_episodes: int, env_args: WaterworldArguments, should_record: bool = False
) -> RunArguments:
    run_builder = (
        RunArguments.Builder()
        .set_num_episodes(num_episodes)
        .set_should_record(should_record)
        .set_environment_args(env_args)
        .set_run_name("controls")
        .set_agent_builders(
            [
                ControlsAgent.Builder(
                    env_name=f"pursuer_{i}",
                    name="controls",
                )
                for i in range(env_args.n_pursuers)
            ]
        )
    )

    return run_builder.build()
