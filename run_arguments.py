from __future__ import annotations

from typing import List, TYPE_CHECKING

from dataclasses import dataclass


if TYPE_CHECKING:
    from agents import AgentBuilder
    from custom_waterworld import WaterworldArguments


@dataclass(frozen=True, kw_only=True, slots=True)
class RunArguments:
    agent_builders: List[AgentBuilder]
    environment_args: WaterworldArguments
    num_episodes: int
    run_name: str
    should_record: bool

    def compile_agents(self, env):
        return {builder.env_name: builder.build(env) for builder in self.agent_builders}
