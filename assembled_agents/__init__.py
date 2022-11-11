import torch

from agents import NeuralNetwork, QNNAgent
from agents.distance_neural_network import DistanceNeuralNetwork


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


def generate_qnn_distance(num_sensors: int):
    return DistanceNeuralNetwork(
        layers=[
            # out_channels * num_sensors + 2 collision features + 3 speed layers
            torch.nn.Linear(64 * num_sensors + 2 + num_sensors * 3, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3),
        ],
        distance_layers=[
            torch.nn.BatchNorm1d(5),
            torch.nn.Conv1d(
                in_channels=5,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.BatchNorm1d(64),
        ],
        speed_features=True,
        num_sensors=num_sensors,
    )
