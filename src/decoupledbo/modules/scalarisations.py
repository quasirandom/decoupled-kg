from typing import Callable

import torch

TScalarisation = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def scalarise_linear(points: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(points * weights, dim=-1)
