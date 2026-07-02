"""ResNet-34-style encoder (no pretrained weights — competition-legal)."""
from __future__ import annotations

from typing import List

from torch import Tensor, nn

from ..layers import BasicBlock, build_activation, build_norm
from .base import Encoder


class ResNet34Encoder(Encoder):
    feature_channels: List[int] = [64, 128, 256, 512]

    def __init__(
        self,
        in_channels: int,
        norm: str = "batch",
        activation: str = "relu",
        attention: str = "none",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        # stem: stride 2 + maxpool → total stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            build_norm(norm, 64),
            build_activation(activation),
            nn.MaxPool2d(3, 2, 1),
        )
        # depth per stage: 3, 4, 6, 3
        depths = (3, 4, 6, 3)
        dpr = [x.item() for x in _linspace(drop_path_rate, sum(depths))]
        c_in = 64
        stages = []
        cur = 0
        for i, (depth, c_out) in enumerate(zip(depths, self.feature_channels)):
            stride = 1 if i == 0 else 2
            blocks = []
            for j in range(depth):
                blocks.append(BasicBlock(
                    in_channels=c_in if j == 0 else c_out,
                    out_channels=c_out,
                    stride=stride if j == 0 else 1,
                    norm=norm, activation=activation,
                    attention=attention, drop_path=dpr[cur],
                ))
                cur += 1
            c_in = c_out
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if getattr(m, "weight", None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)              # stride 4
        outs = []
        for stage in self.stages:
            x = stage(x)
            outs.append(x)
        return outs


def _linspace(end: float, n: int):
    import torch
    if n <= 1:
        return torch.tensor([0.0])
    return torch.linspace(0, end, n)
