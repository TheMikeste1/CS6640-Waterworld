from agents import AbstractAgent


class RandomAgent(AbstractAgent):
    def __call__(self, name, obs):
        return self.env.action_space(name).sample()
