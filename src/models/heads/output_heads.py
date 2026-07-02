"""Output heads producing (mean, rain_logit[, log_var])."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn


@dataclass
class HeadsConfig:
    in_channels: int
    probabilistic: bool = False
    logvar_min: float = -10.0
    logvar_max: float = 5.0
    hidden_channels: int = 0    # 0 = single 1x1 conv per head; >0 = 3x3 conv + 1x1


class OutputHeads(nn.Module):
    def __init__(self, cfg: HeadsConfig):
        super().__init__()
        self.cfg = cfg
        cin = cfg.in_channels
        if cfg.hidden_channels > 0:
            self.mean_head = nn.Sequential(
                nn.Conv2d(cin, cfg.hidden_channels, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(cfg.hidden_channels, 1, 1),
            )
            self.rain_head = nn.Sequential(
                nn.Conv2d(cin, cfg.hidden_channels, 3, 1, 1), nn.ReLU(inplace=True),
                nn.Conv2d(cfg.hidden_channels, 1, 1),
            )
            if cfg.probabilistic:
                self.logvar_head = nn.Sequential(
                    nn.Conv2d(cin, cfg.hidden_channels, 3, 1, 1), nn.ReLU(inplace=True),
                    nn.Conv2d(cfg.hidden_channels, 1, 1),
                )
        else:
            self.mean_head = nn.Conv2d(cin, 1, 1)
            self.rain_head = nn.Conv2d(cin, 1, 1)
            if cfg.probabilistic:
                self.logvar_head = nn.Conv2d(cin, 1, 1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = {
            "mean": self.mean_head(x).squeeze(1),
            "rain_logit": self.rain_head(x).squeeze(1),
        }
        if self.cfg.probabilistic:
            lv = self.logvar_head(x).squeeze(1)
            out["log_var"] = lv.clamp(self.cfg.logvar_min, self.cfg.logvar_max)
        return out


def build_heads(in_channels: int, probabilistic: bool = False,
                hidden_channels: int = 0, **kwargs) -> OutputHeads:
    cfg = HeadsConfig(
        in_channels=in_channels,
        probabilistic=probabilistic,
        hidden_channels=hidden_channels,
        **{k: v for k, v in kwargs.items()
           if k in HeadsConfig.__dataclass_fields__},
    )
    return OutputHeads(cfg)
