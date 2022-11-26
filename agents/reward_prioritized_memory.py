import random
import numpy as np

from agents import Memory


class RewardPrioritizedMemory(Memory):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.mean = 0.0
        self.std_dev = 0.0
        self.instances_seen = 0

    def add(self, experience: tuple, reward: float):
        self.buffer.append((*experience, reward))
        self.instances_seen += 1
        # Update mean and standard deviation
        old_mean = self.mean
        self.mean = (
            self.mean * (self.instances_seen - 1) + reward
        ) / self.instances_seen

        """
        std_dev = sqrt(sum((x - mean)^2) / n)
        """
        # Get the old numerator for the standard deviation
        old_std_sum = self.std_dev ** 2 * (self.instances_seen - 1)
        # Adjust the numerator for the new mean
        adjusted_std_sum = old_std_sum + (reward - old_mean) * (reward - self.mean)
        # Update the standard deviation
        self.std_dev = np.sqrt(adjusted_std_sum / self.instances_seen)

    def sample(
        self, batch_size, zip_by_column=True, use_replacement_on_overflow: bool = True
    ) -> tuple[np.ndarray, ...]:
        if use_replacement_on_overflow and batch_size <= len(self.buffer):
            choices = random.sample(self.buffer, k=batch_size)
        else:
            # We'll use mean and standard deviation to weight the rewards
            weights = [0.0] * len(self.buffer)
            for i, memory in enumerate(self.buffer):
                z_score = (memory[-1] - self.mean) / self.std_dev
                # We want the most extreme rewards to be sampled more often, so we'll
                # use the absolute value of the z-score as the weight.
                weights[i] = abs(z_score)

            choices = random.choices(
                self.buffer,
                k=min(len(self.buffer), batch_size),
                weights=weights,
            )
        if zip_by_column:
            choices = zip(*choices)
        return tuple(map(np.array, choices))
