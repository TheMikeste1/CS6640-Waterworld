from typing import Any

import numpy as np
import pygame

from agents import AbstractAgent


class HumanAgent(AbstractAgent):
    def __init__(self, env, env_name, name):
        super().__init__(env, env_name, name)
        self.up_pressed = False
        self.down_pressed = False
        self.left_pressed = False
        self.right_pressed = False

    def __call__(self, obs) -> (np.ndarray, Any):
        self.gather_inputs()
        command = [
            self.right_pressed - self.left_pressed,
            self.up_pressed - self.down_pressed,
        ]

        return np.array(command, dtype=np.float32), None

    def gather_inputs(self):
        for event in pygame.event.get():
            if event.type != pygame.KEYDOWN and event.type != pygame.KEYUP:
                continue

            set_on = event.type == pygame.KEYDOWN
            match event.key:
                case pygame.K_UP:
                    self.up_pressed = set_on
                case pygame.K_DOWN:
                    self.down_pressed = set_on
                case pygame.K_LEFT:
                    self.left_pressed = set_on
                case pygame.K_RIGHT:
                    self.right_pressed = set_on
        pygame.event.clear()

    @property
    def in_features(self):
        return self.env.observation_spaces[self.env_name].shape[0]

    @property
    def out_features(self):
        return self.env.action_space(self.env_name).shape[0]
