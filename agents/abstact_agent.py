from abc import ABC, abstractmethod

import pettingzoo as pz


class AbstractAgent(ABC):
    def __init__(self, env: pz.AECEnv):
        self.env = env

    @abstractmethod
    def __call__(self, name, obs):
        pass
