from __future__ import annotations
from typing import Callable, Iterable, TYPE_CHECKING, Union

import torch

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences,PyProtectedMember
    AbstractLRScheduler = torch.optim.lr_scheduler._LRScheduler
    from torch import Tensor


class NeuralNetwork(torch.nn.Module):
    __slots__ = (
        "criterion",
        "layers",
        "lr_scheduler",
        "optimizer",
    )

    def __init__(
        self,
        layers: [torch.nn.Module],
        optimizer_factory: Callable[[Iterable[Tensor], ...], torch.optim.Optimizer],
        criterion_factory: Union[
            Callable[[...], torch.nn.Module], Callable[[...], torch.Tensor]
        ],
        lr_scheduler_factory: Callable[
            [torch.optim.Optimizer, ...], AbstractLRScheduler
        ] = None,
        optimizer_kwargs: dict = None,
        criterion_kwargs: dict = None,
        lr_scheduler_kwargs: dict = None,
    ):
        super().__init__()
        if lr_scheduler_factory is None and lr_scheduler_kwargs is not None:
            raise ValueError(
                "lr_scheduler_kwargs cannot be specified without lr_scheduler_factory"
            )

        self.layers = torch.nn.Sequential(*layers)
        self.optimizer = optimizer_factory(
            self.parameters(), **(optimizer_kwargs or {})
        )
        self.criterion = criterion_factory(**(criterion_kwargs or {}))
        self.lr_scheduler = (
            lr_scheduler_factory(self.optimizer, **(lr_scheduler_kwargs or {}))
            if lr_scheduler_factory
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x

    @property
    def in_features(self):
        return self.layers[0].in_features

    @property
    def out_features(self):
        return self.layers[-1].out_features

    def step(self, old_target: torch.Tensor, new_target: torch.Tensor):
        self.optimizer.zero_grad()
        loss = self.criterion(old_target, new_target)
        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return loss.item()
