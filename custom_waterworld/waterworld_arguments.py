from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Collection, Tuple


@dataclass(kw_only=True, frozen=True, slots=True)
class WaterworldArguments:
    class RenderMode(Enum):
        """Render modes for WaterWorld."""

        RGB = "rgb_array"
        HUMAN = "human"
        NONE = None

    n_pursuers: int = 2
    n_evaders: int = 5
    n_poisons: int = 10
    n_coop: int = 1
    n_sensors: int = 30
    sensor_range: float = 0.2
    radius: float = 0.015
    obstacle_radius: float = 0.1
    obstacle_coord: Collection[Tuple[float, float]] | None = field(
        default_factory=lambda: [(0.5, 0.5)]
    )
    pursuer_max_accel: float = 0.02
    pursuer_speed: float = 0.2
    evader_speed: float = 0.1
    poison_speed: float = 0.1
    poison_reward: float = -1.0
    food_reward: float = 10.0
    encounter_reward: float = 0.01
    thrust_penalty: float = -0.5
    local_ratio: float = 1.0
    speed_features: bool = True
    max_cycles: int = 500
    render_mode: WaterworldArguments.RenderMode = RenderMode.NONE
    FPS: int = 15

    """
    Number of randomly placed obstacles in the environment.
    Only used when `obstacle_coord` is `None`.
    """
    num_random_obstacles: int = 0

    @property
    def n_obstacles(self) -> int:
        return (
            len(self.obstacle_coord)
            if self.obstacle_coord is not None
            else self.num_random_obstacles
        )

    def to_dict(self) -> dict:
        # noinspection PyUnresolvedReferences
        out = {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
        del out["num_random_obstacles"]
        out["render_mode"] = self.render_mode.value
        out["n_obstacles"] = self.n_obstacles
        return out
