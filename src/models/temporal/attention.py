"""Learned temporal attention: weight each frame by a learned scalar per pixel."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .base import TemporalModule


class TemporalAttention(TemporalModule):
    """Softmax attention over frames.

    Each temporal position (frames + diffs) produces a scalar attention map
    per pixel; the softmax-weighted sum across time becomes the output. This
    is a cheap self-attention variant tailored to T=3+diffs.
    """

    def __init__(
        self,
        in_channels_per_frame: int,
        n_frames: int,
        n_diff_frames: int = 0,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels_per_frame
        self.n_frames = n_frames
        self.n_diff_frames = n_diff_frames
        self.hidden_dim = hidden_dim
        self.t = n_frames + n_diff_frames
        # score net: 1x1 conv from C_per_frame -> 1 per time step
        self.score = nn.Conv2d(in_channels_per_frame, 1, kernel_size=1)
        # project each frame to hidden_dim
        self.value = nn.Conv2d(in_channels_per_frame, hidden_dim, kernel_size=1)
        self.out_channels = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        b, cin, h, w = x.shape
        c_per = self.in_channels
        x_stack = x.view(b, self.t, c_per, h, w)
        # scores: (B, T, 1, H, W)
        s = self.score(x_stack.view(b * self.t, c_per, h, w)).view(b, self.t, 1, h, w)
        attn = F.softmax(s, dim=1)
        v = self.value(x_stack.view(b * self.t, c_per, h, w)).view(b, self.t, self.hidden_dim, h, w)
        # weighted sum
        return (attn * v).sum(dim=1)
