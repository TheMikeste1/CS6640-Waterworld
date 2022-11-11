import torch

from agents import NeuralNetwork


class DistanceNeuralNetwork(NeuralNetwork):
    def __init__(
        self,
        layers: [torch.nn.Module],
        distance_layers: [torch.nn.Module],
        speed_features: bool,
        num_sensors: int,
    ):
        """
        If num_sensors is None, the model will try to calculate the number of layers
        itself.
        """
        super().__init__(layers)
        self.distance_layers = NeuralNetwork(layers=distance_layers)
        # Distance layers should have 5 incoming channels--one for each
        # distance feature type:
        #   obstacle, barrier, food, poison, pursuer
        assert self.distance_layers.in_features == 5, (
            f"Distance layers should have 5 incoming channels,"
            f"but found {self.distance_layers.in_features}"
        )

        self.num_sensors = num_sensors
        self.speed_features = speed_features

        # The number of inputs to layers must be the sum of outputs
        # for all 5 distance channels, plus all speed features, plus 2 for collisions
        total_layer_outputs = (
            self.distance_layers.out_features * num_sensors
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
        distance_obs, speed_obs, collision_obs = self.slice_features(obs)

        # Process distance features
        distance_obs = self.distance_layers(distance_obs)
        # Flatten distance features for use with other features
        distance_obs = distance_obs.flatten(start_dim=-2)
        distance_obs = distance_obs.unsqueeze(dim=-2)

        # Flatten speed features for use with other features
        speed_obs = speed_obs.flatten(start_dim=-2)
        speed_obs = speed_obs.unsqueeze(dim=-2)

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

    def slice_features(
        self, obs: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
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

        # Stack all distance features
        distance_obs = torch.cat(
            [
                obstacle_distance_obs,
                barrier_distance_obs,
                food_distance_obs,
                poison_distance_obs,
                pursuer_distance_obs,
            ],
            dim=-2,
        )
        # Stack all speed observations
        speed_obs = torch.cat(
            [
                food_speed_obs,
                poison_speed_obs,
                pursuer_speed_obs,
            ],
            dim=-2,
        )

        return distance_obs, speed_obs, collision_obs
