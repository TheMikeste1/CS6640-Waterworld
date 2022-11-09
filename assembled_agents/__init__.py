import torch

from agents import NeuralNetwork, QNNAgent


def generate_simple_linear_256_64_3(num_obs: int):
    return NeuralNetwork(
        layers=[
            torch.nn.Linear(num_obs, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3),
        ],
    )
