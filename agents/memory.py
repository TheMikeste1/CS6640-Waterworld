import random
from collections import deque

import numpy as np


class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self.buffer)

    @property
    def max_size(self):
        return self.buffer.maxlen

    @max_size.setter
    def max_size(self, value):
        new_buffer = deque(maxlen=value)
        new_buffer.extend(self.buffer)
        self.buffer = new_buffer

    def add(self, experience: tuple):
        self.buffer.append(experience)

    def sample(self, batch_size, zip_by_column=True) -> tuple[np.ndarray, ...]:
        """
        Returns some random experiences from the memory.

        :param batch_size: How many experiences to return.
        :param zip_by_column: If we should return the experiences by column.
        For example, assume we have 2 experiences consisting of a float and a string.
        If we zip by column, we will return ((float, float), (string, string)).
        If we don't zip by column, we will return each experience in its original form:
        ((float, string), (float, string)).

        :return:
        """
        choices = random.sample(self.buffer, k=batch_size)
        if zip_by_column:
            choices = zip(*choices)
        return tuple(map(np.array, choices))
