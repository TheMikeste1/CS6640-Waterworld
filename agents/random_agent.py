from __future__ import annotations
from typing import TYPE_CHECKING

from agents import AbstractAgent

if TYPE_CHECKING:
    import numpy as np


class RandomAgent(AbstractAgent):
    def __call__(self, name, obs) -> np.ndarray:
        return self.env.action_space(name).sample()
