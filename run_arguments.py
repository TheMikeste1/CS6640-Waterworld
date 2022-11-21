from typing import Dict, TYPE_CHECKING

from dataclasses import dataclass


if TYPE_CHECKING:
    from agents import AgentBuilder
    from custom_waterworld import WaterworldArguments


@dataclass(frozen=True, kw_only=True, slots=True)
class RunArguments:
    agent_builders: Dict[str, AgentBuilder]
    environment_args: WaterworldArguments
    num_episodes: int
    run_name: str
    should_record: bool
