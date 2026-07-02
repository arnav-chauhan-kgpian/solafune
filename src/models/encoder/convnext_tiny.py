"""ConvNeXt-Tiny-style encoder (from scratch, no pretrained weights)."""
from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from ..layers import ConvNeXtBlock, LayerNorm2d
from .base import Encoder


class ConvNeXtTinyEncoder(Encoder):
    feature_channels: List[int] = [96, 192, 384, 768]

    def __init__(
        self,
        in_channels: int,
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        depths = (3, 3, 9, 3)
        # patchify stem: 4x4 conv stride 4 → stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.feature_channels[0], 4, 4),
            LayerNorm2d(self.feature_channels[0]),
        )
        downsamplers = []
        stages = []
        total = sum(depths)
        dpr = torch.linspace(0.0, drop_path_rate, total).tolist()
        cur = 0
        for i, (depth, c) in enumerate(zip(depths, self.feature_channels)):
            if i == 0:
                downsamplers.append(nn.Identity())
            else:
                downsamplers.append(nn.Sequential(
                    LayerNorm2d(self.feature_channels[i - 1]),
                    nn.Conv2d(self.feature_channels[i - 1], c, 2, 2),
                ))
            blocks = []
            for _ in range(depth):
                blocks.append(ConvNeXtBlock(c, drop_path=dpr[cur],
                                            layer_scale_init=layer_scale_init))
                cur += 1
            stages.append(nn.Sequential(*blocks))
        self.downsamplers = nn.ModuleList(downsamplers)
        self.stages = nn.ModuleList(stages)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for ds, stage in zip(self.downsamplers, self.stages):
            x = ds(x)
            x = stage(x)
            outs.append(x)
        return outs
