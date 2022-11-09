import torch

from agents import NeuralNetwork


class DistanceNeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        layers: [torch.nn.Module],
        poison_layers: [torch.nn.Module],
        pursuer_layers: [torch.nn.Module],
        food_layers: [torch.nn.Module],
        obstacle_layers: [torch.nn.Module],
        barrier_layers: [torch.nn.Module],
        speed_features: bool,
        num_sensors: int = None,
    ):
        """
        If num_sensors is None, the model will try to calculate the number of layers
        itself.
        """
        super().__init__(layers)
        self.poison_layers = NeuralNetwork(layers=poison_layers)
        self.pursuer_layers = NeuralNetwork(layers=pursuer_layers)
        self.food_layers = NeuralNetwork(layers=food_layers)
        self.obstacle_layers = NeuralNetwork(layers=obstacle_layers)
        self.barrier_layers = NeuralNetwork(layers=barrier_layers)
        # All incoming layers should all have the same number of inputs
        assert (
            self.poison_layers.in_features == self.pursuer_layers.in_features
            and self.poison_layers.in_features == self.food_layers.in_features
        ), (
            "Poison, Pursuer, Food, Barrier, and Obstacle layers should all have"
            "the same number of inputs: one for each input sensor."
        )
        self.num_sensors = (
            num_sensors if num_sensors else self.poison_layers.in_features
        )
        self.speed_features = speed_features

        # The number of inputs to layers must be the sum of outputs
        # for all distance layers, plus all speed features, plus 2 for collisions
        total_layer_outputs = (
            self.poison_layers.out_features
            + self.pursuer_layers.out_features
            + self.food_layers.out_features
            + self.obstacle_layers.out_features
            + self.barrier_layers.out_features
            + (self.num_sensors * 3 if self.speed_features else 0)
            + 2
        )
        assert super(DistanceNeuralNetwork, self).in_features == total_layer_outputs, (
            "The number of inputs to the layers must be the sum of outputs for all"
            "distance layers, plus all speed features, plus 2 for collisions. "
            "See https://pettingzoo.farama.org/environments/sisl/waterworld"
            "/#observation-space"
        )

    @property
    def in_features(self) -> int:
        return self.num_sensors * (5 + (3 if self.speed_features else 0)) + 2

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        current_slice = 0
        # Obstacle features
        obstacle_distance_obs = obs[
            ..., current_slice : current_slice + self.num_sensors
        ]
        current_slice += self.num_sensors
        # Barrier features
        barrier_distance_obs = obs[
            ..., current_slice : current_slice + self.num_sensors
        ]
        current_slice += self.num_sensors
        # Food features
        food_distance_obs = obs[..., current_slice : current_slice + self.num_sensors]
        current_slice += self.num_sensors
        food_speed_obs = []
        if self.speed_features:
            food_speed_obs = obs[..., current_slice : current_slice + self.num_sensors]
            current_slice += self.num_sensors
        # Poison features
        poison_distance_obs = obs[..., current_slice : current_slice + self.num_sensors]
        current_slice += self.num_sensors
        poison_speed_obs = []
        if self.speed_features:
            poison_speed_obs = obs[
                ..., current_slice : current_slice + self.num_sensors
            ]
            current_slice += self.num_sensors
        # Pursuer features
        pursuer_distance_obs = obs[
            ..., current_slice : current_slice + self.num_sensors
        ]
        current_slice += self.num_sensors
        pursuer_speed_obs = []
        if self.speed_features:
            pursuer_speed_obs = obs[
                ..., current_slice : current_slice + self.num_sensors
            ]
            current_slice += self.num_sensors
        # Collision features
        collision_obs = obs[..., current_slice : current_slice + 2]
        current_slice += 2
        assert (
            current_slice == obs.shape[-1]
        ), f"Too many observations! Found {obs.shape[-1]} but expected {current_slice}"

        # Run observations through each layer
        obstacle_distance_obs = self.obstacle_layers(obstacle_distance_obs)
        barrier_distance_obs = self.barrier_layers(barrier_distance_obs)
        food_distance_obs = self.food_layers(food_distance_obs)
        poison_distance_obs = self.poison_layers(poison_distance_obs)
        pursuer_distance_obs = self.pursuer_layers(pursuer_distance_obs)
        # Concatenate all distance observations
        distance_obs = torch.cat(
            [
                obstacle_distance_obs,
                barrier_distance_obs,
                food_distance_obs,
                poison_distance_obs,
                pursuer_distance_obs,
            ],
            dim=-1,
        )
        # Concatenate all speed observations
        speed_obs = torch.cat(
            [
                food_speed_obs,
                poison_speed_obs,
                pursuer_speed_obs,
            ],
            dim=-1,
        )
        # Concatenate all observations
        obs = torch.cat(
            [
                distance_obs,
                speed_obs,
                collision_obs,
            ],
            dim=-1,
        )
        # Run observations through the main layers
        return super(DistanceNeuralNetwork, self).forward(obs)
