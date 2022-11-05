from __future__ import annotations
from typing import Any, TYPE_CHECKING

from agents import AbstractAgent

if TYPE_CHECKING:
    import numpy as np


class RandomAgent(AbstractAgent):
    def __call__(self, obs) -> (np.ndarray, Any):
        return self.env.action_space(self.name).sample(), None
