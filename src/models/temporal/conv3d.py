"""3D-Conv temporal stem.

Reshapes the channel-stacked input from `(B, T*C, H, W)` back to
`(B, C, T, H, W)`, applies a Conv3d that mixes across T, and reduces the
temporal axis (concat / mean / last) before returning a 4D tensor.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from ..layers import build_activation, build_norm
from .base import TemporalModule


class Conv3DStem(TemporalModule):
    def __init__(
        self,
        in_channels_per_frame: int,
        n_frames: int,
        n_diff_frames: int = 0,
        out_channels_per_step: int = 64,
        kernel_size=(3, 3, 3),
        norm: str = "batch",
        activation: str = "gelu",
        temporal_reduce: str = "concat",   # "concat" | "mean" | "last"
        include_diffs_in_conv: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels_per_frame
        self.n_frames = n_frames
        self.n_diff_frames = n_diff_frames
        self.temporal_reduce = temporal_reduce
        self.include_diffs_in_conv = include_diffs_in_conv

        # Total time steps processed by Conv3D:
        # frames + optionally diffs. Diffs are folded into T dim.
        self.t_total = n_frames + (n_diff_frames if include_diffs_in_conv else 0)

        pad = tuple(k // 2 for k in kernel_size)
        # Conv3d input shape (B, C, T, H, W). We produce out_channels_per_step per T slot.
        self.conv = nn.Conv3d(
            in_channels=in_channels_per_frame,
            out_channels=out_channels_per_step,
            kernel_size=kernel_size, padding=pad, bias=False,
        )
        self.n = _norm3d(norm, out_channels_per_step)
        self.act = build_activation(activation)

        # If diffs are not fed through Conv3D, they are appended to the
        # final 2D output as raw channels.
        residual_c = 0 if include_diffs_in_conv else in_channels_per_frame * n_diff_frames

        if temporal_reduce == "concat":
            self.out_channels = out_channels_per_step * self.t_total + residual_c
        elif temporal_reduce in ("mean", "last"):
            self.out_channels = out_channels_per_step + residual_c
        else:
            raise ValueError(f"unknown temporal_reduce: {temporal_reduce!r}")

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, (n_frames + n_diff_frames) * C_per_frame, H, W)
        b, cin, h, w = x.shape
        expected = (self.n_frames + self.n_diff_frames) * self.in_channels
        if cin != expected:
            raise ValueError(
                f"Conv3DStem: expected {expected} input channels, got {cin}"
            )
        # split channels into frames + diffs blocks
        c_per = self.in_channels
        # frames: (B, n_frames, C, H, W); diffs: (B, n_diff, C, H, W)
        x_stack = x.view(b, self.n_frames + self.n_diff_frames, c_per, h, w)
        frames = x_stack[:, : self.n_frames]
        diffs = x_stack[:, self.n_frames :]
        if self.include_diffs_in_conv:
            conv_in = x_stack  # all T
        else:
            conv_in = frames
        # (B, T, C, H, W) -> (B, C, T, H, W)
        conv_in = conv_in.permute(0, 2, 1, 3, 4).contiguous()
        y = self.act(self.n(self.conv(conv_in)))  # (B, C_out, T, H, W)
        if self.temporal_reduce == "concat":
            b2, c2, t2, h2, w2 = y.shape
            y2 = y.permute(0, 2, 1, 3, 4).reshape(b2, t2 * c2, h2, w2)
        elif self.temporal_reduce == "mean":
            y2 = y.mean(dim=2)
        else:  # last
            y2 = y[:, :, -1]

        if not self.include_diffs_in_conv and self.n_diff_frames > 0:
            b3, dt, dc, dh, dw = diffs.shape
            diffs_flat = diffs.reshape(b3, dt * dc, dh, dw)
            y2 = torch.cat([y2, diffs_flat], dim=1)
        return y2


def _norm3d(norm: str, c: int) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm3d(c)
    if norm == "group":
        g = min(8, c)
        while c % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, c)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown 3D norm: {norm!r}")
