"""Decoder ABC."""
from __future__ import annotations

from abc import abstractmethod
from typing import List

from torch import Tensor, nn


class Decoder(nn.Module):
    """Takes a list of encoder feature maps and returns a single output map."""

    out_channels: int

    @abstractmethod
    def forward(self, features: List[Tensor]) -> Tensor:
        ...
