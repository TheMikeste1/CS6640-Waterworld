from __future__ import annotations

from typing import Iterable, List, Sized, TYPE_CHECKING

from dataclasses import dataclass


if TYPE_CHECKING:
    from agents import AgentBuilder
    from custom_waterworld import WaterworldArguments


@dataclass(frozen=True, kw_only=True, slots=True)
class RunArguments:
    @dataclass
    class Builder:
        agent_builders: List[AgentBuilder] | AgentBuilder
        environment_args: WaterworldArguments
        num_episodes: int
        run_name: str
        should_record: bool

        def add_agent_builder(
            self, agent_builder: AgentBuilder
        ) -> RunArguments.Builder:
            if isinstance(self.agent_builders, list):
                self.agent_builders.append(agent_builder)
            else:
                self.agent_builders = [self.agent_builders, agent_builder]
            return self

        def set_agent_builders(
            self, agent_builders: Iterable[AgentBuilder]
        ) -> RunArguments.Builder:
            self.agent_builders = list(agent_builders)
            return self

        def set_environment_args(
            self, environment_args: WaterworldArguments
        ) -> RunArguments.Builder:
            self.environment_args = environment_args
            return self

        def set_num_episodes(self, num_episodes: int) -> RunArguments.Builder:
            self.num_episodes = num_episodes
            return self

        def set_run_name(self, run_name: str) -> RunArguments.Builder:
            self.run_name = run_name
            return self

        def set_should_record(self, should_record: bool) -> RunArguments.Builder:
            self.should_record = should_record
            return self

        def build(self) -> RunArguments:
            return RunArguments(
                agent_builders=self.agent_builders,
                environment_args=self.environment_args,
                num_episodes=self.num_episodes,
                run_name=self.run_name,
                should_record=self.should_record,
            )

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
