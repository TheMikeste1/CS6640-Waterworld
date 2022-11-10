from typing import Any

import numpy as np
import pygame

from agents import AbstractAgent


class HumanAgent(AbstractAgent):
    def __call__(self, obs) -> (np.ndarray, Any):
        command = [0] * self.out_features
        for event in pygame.event.get():
            match event.key:
                case pygame.K_UP:
                    command[0] += -1
                case pygame.K_DOWN:
                    command[0] += 1
                case pygame.K_LEFT:
                    command[2] += -1
                case pygame.K_RIGHT:
                    command[3] += 1
        pygame.event.clear()

        return np.array(command), None

    @property
    def in_features(self):
        return self.env.observation_spaces[self.name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.name).n

