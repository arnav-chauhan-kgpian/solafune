"""ConvLSTM temporal module (single-layer, batch-first)."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.utils.checkpoint as ckpt

from .base import TemporalModule


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels + hidden_dim, 4 * hidden_dim,
                              kernel_size, padding=pad)

    def forward(self, x: Tensor, h: Tensor, c: Tensor):
        z = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(z, 4, dim=1)
        i = torch.sigmoid(i); f = torch.sigmoid(f); o = torch.sigmoid(o); g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMTemporal(TemporalModule):
    def __init__(
        self,
        in_channels_per_frame: int,
        n_frames: int,
        n_diff_frames: int = 0,
        hidden_dim: int = 128,
        kernel_size: int = 3,
        include_diffs_in_input: bool = True,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels_per_frame
        self.n_frames = n_frames
        self.n_diff_frames = n_diff_frames
        self.hidden_dim = hidden_dim
        self.include_diffs_in_input = include_diffs_in_input
        self.use_ckpt = use_gradient_checkpointing

        # If diffs are used, each temporal step's input has 2*C channels
        # (raw frame + diff-of-that-step). We concatenate diff with the
        # following frame when available; for t=0 we duplicate frame_0 as diff.
        input_c = in_channels_per_frame * (2 if include_diffs_in_input else 1)
        self.cell = ConvLSTMCell(input_c, hidden_dim, kernel_size)
        self.out_channels = hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        b, cin, h, w = x.shape
        c_per = self.in_channels
        # split back into frames + diffs blocks
        x_stack = x.view(b, self.n_frames + self.n_diff_frames, c_per, h, w)
        frames = x_stack[:, : self.n_frames]                     # (B, T, C, H, W)
        diffs = x_stack[:, self.n_frames :]                       # (B, T-1, C, H, W)

        # initial states
        hidden = frames.new_zeros(b, self.hidden_dim, h, w)
        cell = frames.new_zeros(b, self.hidden_dim, h, w)

        for t in range(self.n_frames):
            frame = frames[:, t]
            if self.include_diffs_in_input:
                if t == 0:
                    diff = torch.zeros_like(frame)
                else:
                    diff = diffs[:, t - 1] if t - 1 < diffs.shape[1] else torch.zeros_like(frame)
                inp = torch.cat([frame, diff], dim=1)
            else:
                inp = frame
            if self.use_ckpt and self.training:
                hidden, cell = ckpt.checkpoint(self.cell, inp, hidden, cell, use_reentrant=False)
            else:
                hidden, cell = self.cell(inp, hidden, cell)
        return hidden
