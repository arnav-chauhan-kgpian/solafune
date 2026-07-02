"""Temporal module ABC."""
from __future__ import annotations

from abc import abstractmethod
from torch import Tensor, nn


class TemporalModule(nn.Module):
    """Base class. Reports its output channel count via `out_channels`."""

    in_channels: int
    out_channels: int
    n_frames: int
    n_diff_frames: int

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        ...
