"""U-Net decoder with skip connections."""
from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ..layers import build_activation, build_norm
from .base import Decoder


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 norm: str = "batch", activation: str = "relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, 1, 1, bias=False)
        self.n1 = build_norm(norm, out_channels)
        self.act = build_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.n2 = build_norm(norm, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.act(self.n1(self.conv1(x)))
        x = self.act(self.n2(self.conv2(x)))
        return x


class UNetDecoder(Decoder):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        norm: str = "batch",
        activation: str = "relu",
        output_size=(41, 41),
    ):
        super().__init__()
        assert len(encoder_channels) == 4 and len(decoder_channels) == 4
        ec = list(encoder_channels)
        dc = list(decoder_channels)
        self.output_size = tuple(output_size)
        # decoder stages: bottleneck -> up1 -> up2 -> up3 -> up4 (to encoder_stride/2 level)
        self.up1 = UpBlock(ec[3], ec[2], dc[0], norm, activation)
        self.up2 = UpBlock(dc[0], ec[1], dc[1], norm, activation)
        self.up3 = UpBlock(dc[1], ec[0], dc[2], norm, activation)
        # last up: no skip; upsample to output_size
        self.final = nn.Sequential(
            nn.Conv2d(dc[2], dc[3], 3, 1, 1, bias=False),
            build_norm(norm, dc[3]),
            build_activation(activation),
        )
        self.out_channels = dc[3]

    def forward(self, features: List[Tensor]) -> Tensor:
        f1, f2, f3, f4 = features
        x = self.up1(f4, f3)
        x = self.up2(x, f2)
        x = self.up3(x, f1)
        x = self.final(x)
        if x.shape[-2:] != self.output_size:
            x = F.adaptive_avg_pool2d(x, self.output_size)
        return x
