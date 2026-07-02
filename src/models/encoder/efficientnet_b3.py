"""EfficientNet-B3-style encoder built from MBConv blocks.

Follows the EfficientNet-B3 depth+width scaling but simplified: we produce
4 feature maps at strides [4, 8, 16, 32] matching the Encoder contract.
No pretrained weights (competition-legal).
"""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from ..layers import MBConv, build_activation, build_norm
from .base import Encoder


class EfficientNetB3Encoder(Encoder):
    feature_channels: List[int] = [32, 48, 136, 384]

    def __init__(
        self,
        in_channels: int,
        norm: str = "batch",
        activation: str = "silu",
        drop_path_rate: float = 0.2,
    ):
        super().__init__()
        # Stem: stride-2 3x3 conv (input downsample from H to H/2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 40, 3, 2, 1, bias=False),
            build_norm(norm, 40),
            build_activation(activation),
        )
        # Stage config: (out_channels, depth, stride, kernel, expand_ratio)
        # Strides at each stage: [1, 2, 2, 2, 2, 2, 1] — we group into 4 outputs
        # by cumulative stride to hit [4, 8, 16, 32].
        stage_specs = [
            # after stem H is /2. We need /4, /8, /16, /32.
            # stage1: 40→32 stride 2 (total /4), depth=2, k=3, e=1
            (32, 2, 2, 3, 1),
            # stage2: 32→48 stride 2 (total /8), depth=3, k=3, e=6
            (48, 3, 2, 3, 6),
            # stage3: 48→136 stride 2 (total /16), depth=5, k=5, e=6
            (136, 5, 2, 5, 6),
            # stage4: 136→384 stride 2 (total /32), depth=6, k=3, e=6
            (384, 6, 2, 3, 6),
        ]
        total_blocks = sum(s[1] for s in stage_specs)
        dpr = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()

        stages = []
        c_in = 40
        b = 0
        for (c_out, depth, stride, k, e) in stage_specs:
            blocks = []
            for j in range(depth):
                blocks.append(MBConv(
                    in_channels=c_in if j == 0 else c_out,
                    out_channels=c_out,
                    stride=stride if j == 0 else 1,
                    kernel_size=k,
                    expand_ratio=e,
                    norm=norm, activation=activation,
                    drop_path=dpr[b],
                ))
                b += 1
            c_in = c_out
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs
