"""FPN-style decoder."""
from __future__ import annotations

from typing import List, Sequence

from torch import Tensor, nn
import torch.nn.functional as F

from ..layers import build_activation, build_norm
from .base import Decoder


class FPNDecoder(Decoder):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        fpn_channels: int = 256,
        out_channels: int = 64,
        norm: str = "batch",
        activation: str = "relu",
        output_size=(41, 41),
    ):
        super().__init__()
        assert len(encoder_channels) == 4
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_channels, 1, bias=False) for c in encoder_channels
        ])
        self.smooth = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, 1, 1, bias=False),
                build_norm(norm, fpn_channels),
                build_activation(activation),
            ) for _ in range(4)
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * fpn_channels, out_channels, 3, 1, 1, bias=False),
            build_norm(norm, out_channels),
            build_activation(activation),
        )
        self.output_size = tuple(output_size)
        self.out_channels = out_channels

    def forward(self, features: List[Tensor]) -> Tensor:
        p = [None] * 4
        p[3] = self.laterals[3](features[3])
        for i in (2, 1, 0):
            up = F.interpolate(p[i + 1], size=features[i].shape[-2:],
                               mode="bilinear", align_corners=False)
            p[i] = self.laterals[i](features[i]) + up
        # smooth
        p = [s(x) for s, x in zip(self.smooth, p)]
        # upsample all to finest resolution, concat, fuse
        target_hw = p[0].shape[-2:]
        p_up = [p[0]] + [F.interpolate(pi, size=target_hw, mode="bilinear",
                                        align_corners=False) for pi in p[1:]]
        import torch
        y = self.fuse(torch.cat(p_up, dim=1))
        if y.shape[-2:] != self.output_size:
            y = F.adaptive_avg_pool2d(y, self.output_size)
        return y
