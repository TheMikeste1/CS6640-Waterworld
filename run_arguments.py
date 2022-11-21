from __future__ import annotations

from typing import Iterable, List, Sized, TYPE_CHECKING

from dataclasses import dataclass


if TYPE_CHECKING:
    from agents import AgentBuilder
    from custom_waterworld import WaterworldArguments


@dataclass(frozen=True, kw_only=True, slots=True)
class RunArguments:
    agent_builders: List[AgentBuilder] | AgentBuilder
    environment_args: WaterworldArguments
    num_episodes: int
    run_name: str
    should_record: bool

    def __post_init__(self):
        if isinstance(self.agent_builders, Sized):
            assert len(self.agent_builders) == self.environment_args.n_pursuers, (
                f"Expected {self.environment_args.n_pursuers} agent builders, "
                f"got {len(self.agent_builders)}"
            )

    def compile_agents(self, env):
        if isinstance(self.agent_builders, Iterable):
            return {
                builder.env_name: builder.build(env) for builder in self.agent_builders
            }
        agent = self.agent_builders.build(env)
        return {f"pursuer_{i}": agent for i in range(self.environment_args.n_pursuers)}
