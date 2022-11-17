from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class StepData:
    state: np.ndarray
    next_state: np.ndarray
    action: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Any
    agent_info: Any
