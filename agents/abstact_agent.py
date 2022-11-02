from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pettingzoo as pz


class AbstractAgent(ABC):
    __slots__ = ("env",)

    def __init__(self, env: pz.AECEnv):
        self.env = env

    @abstractmethod
    def __call__(self, name, obs) -> np.ndarray:
        pass
