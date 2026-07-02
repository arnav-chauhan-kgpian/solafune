"""Neural network primitive layers.

Includes:
    * norm helpers (BatchNorm / GroupNorm / LayerNorm2d) selectable by name
    * activation helpers (ReLU / GELU / SiLU) selectable by name
    * DropPath (stochastic depth)
    * Squeeze-and-Excitation (SE)
    * CBAM
    * ResNet basic block
    * ConvNeXt block
    * EfficientNet-style MBConv (depthwise + SE + point-wise)
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Norm and activation selectors
# ---------------------------------------------------------------------------
def build_norm(norm: str, num_channels: int, groups: int = 8) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm == "group":
        g = min(groups, num_channels)
        while num_channels % g != 0 and g > 1:
            g -= 1
        return nn.GroupNorm(g, num_channels)
    if norm == "layer":
        return LayerNorm2d(num_channels)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {norm!r}")


def build_activation(act: str) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError(f"unknown activation: {act!r}")


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm applied over the channel dim of a 4D (B, C, H, W) tensor."""

    def forward(self, x: Tensor) -> Tensor:
        # (B, C, H, W) -> (B, H, W, C) -> normalize C -> back
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# DropPath (stochastic depth)
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """Per-sample stochastic depth (drops the residual branch)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask


# ---------------------------------------------------------------------------
# SE and CBAM attention
# ---------------------------------------------------------------------------
class SqueezeExcitation(nn.Module):
    """Standard SE block: (B, C, H, W) -> channel-wise gating."""

    def __init__(self, channels: int, reduction: int = 8, activation: str = "silu"):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = build_activation(activation)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.fc2(self.act(self.fc1(s)))
        return x * torch.sigmoid(s)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        hidden = max(channels // reduction, 4)
        # channel attention: mean+max pool → shared MLP → sigmoid
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        # spatial attention: mean+max channel pool → conv → sigmoid
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                      padding=spatial_kernel // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx)).view(b, c, 1, 1)
        x = x * ca
        avg2 = x.mean(dim=1, keepdim=True)
        mx2 = x.max(dim=1, keepdim=True).values
        sa = torch.sigmoid(self.spatial_conv(torch.cat([avg2, mx2], dim=1)))
        return x * sa


def build_attention(kind: str, channels: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "se":
        return SqueezeExcitation(channels)
    if kind == "cbam":
        return CBAM(channels)
    raise ValueError(f"unknown attention: {kind!r}")


# ---------------------------------------------------------------------------
# ResNet-basic block
# ---------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: str = "batch",
        activation: str = "relu",
        attention: str = "none",
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.n1 = build_norm(norm, out_channels)
        self.act = build_activation(activation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.n2 = build_norm(norm, out_channels)
        self.attn = build_attention(attention, out_channels)
        self.drop_path = DropPath(drop_path)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                build_norm(norm, out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)
        out = self.n1(self.conv1(x))
        out = self.act(out)
        out = self.n2(self.conv2(out))
        out = self.attn(out)
        out = self.drop_path(out) + identity
        return self.act(out)


# ---------------------------------------------------------------------------
# ConvNeXt block
# ---------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    """Standard ConvNeXt block: depthwise conv → LN → pointwise → GELU → pointwise."""

    def __init__(
        self,
        channels: int,
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, 4 * channels, 1)
        self.act = build_activation(activation)
        self.pwconv2 = nn.Conv2d(4 * channels, channels, 1)
        self.gamma = nn.Parameter(
            layer_scale_init * torch.ones(1, channels, 1, 1), requires_grad=True
        ) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = x * self.gamma
        return identity + self.drop_path(x)


# ---------------------------------------------------------------------------
# EfficientNet-style MBConv
# ---------------------------------------------------------------------------
class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 4,
        stride: int = 1,
        kernel_size: int = 3,
        norm: str = "batch",
        activation: str = "silu",
        se_reduction: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, hidden, 1, bias=False),
                build_norm(norm, hidden),
                build_activation(activation),
            ]
        layers += [
            nn.Conv2d(hidden, hidden, kernel_size, stride, kernel_size // 2,
                      groups=hidden, bias=False),
            build_norm(norm, hidden),
            build_activation(activation),
            SqueezeExcitation(hidden, reduction=se_reduction, activation=activation),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
            build_norm(norm, out_channels),
        ]
        self.block = nn.Sequential(*layers)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        y = self.block(x)
        if self.use_residual:
            y = self.drop_path(y) + x
        return y
