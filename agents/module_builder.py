from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(kw_only=True)
class ModuleBuilder:
    factory: Callable[[...], torch.nn.Module]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self) -> torch.nn.Module:
        return self.build()

    def build(self) -> torch.nn.Module:
        return self.factory(**self.kwargs)
