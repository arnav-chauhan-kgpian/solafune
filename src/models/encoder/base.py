"""Encoder ABC."""
from __future__ import annotations

from abc import abstractmethod
from typing import List

from torch import Tensor, nn


class Encoder(nn.Module):
    """4-stage encoder producing feature maps at strides [4, 8, 16, 32]."""

    feature_channels: List[int]
    feature_strides: List[int] = [4, 8, 16, 32]

    @abstractmethod
    def forward(self, x: Tensor) -> List[Tensor]:
        ...
