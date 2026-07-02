"""Passthrough temporal module — channel-stack format is left untouched.

The Dataset already emits `(B, (n_frames + n_diff_frames) * C_per_frame, H, W)`.
This module is a no-op.
"""
from __future__ import annotations

from torch import Tensor

from .base import TemporalModule


class PassthroughTemporal(TemporalModule):
    def __init__(self, in_channels_per_frame: int, n_frames: int,
                 n_diff_frames: int = 0, **_):
        super().__init__()
        self.in_channels = in_channels_per_frame
        self.n_frames = n_frames
        self.n_diff_frames = n_diff_frames
        self.out_channels = in_channels_per_frame * (n_frames + n_diff_frames)

    def forward(self, x: Tensor) -> Tensor:
        return x
