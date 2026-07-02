"""Top-level precipitation nowcasting model.

Composition:
    Temporal → Encoder → Decoder → (aux fusion) → Heads

The model is agnostic to satellite identity — the padded-channel input plus
an auxiliary satellite one-hot lets a single set of weights handle all three
sensors. A learned satellite embedding is added at the decoder output.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

import torch
from torch import Tensor, nn

from ..constants import SATELLITES
from .decoder import build_decoder, Decoder
from .encoder import build_encoder, Encoder
from .heads import build_heads, OutputHeads
from .temporal import build_temporal, TemporalModule


@dataclass
class NowcasterConfig:
    # channel wiring
    in_channels_per_frame: int          # C_max after padding, e.g. 10 for ir_only
    n_frames: int = 3
    n_diff_frames: int = 2              # T-1 diff channels when include_diff_frames
    # module selectors
    encoder: str = "resnet34"
    temporal: str = "none"
    decoder: str = "unet"
    # head config
    probabilistic: bool = False
    head_hidden_channels: int = 0
    # aux fusion
    aux_dim: int = 6                    # sat_onehot(3) + zenith + hour_sin + hour_cos
    aux_embed_dim: int = 64
    # per-module kwargs (passed through)
    encoder_kwargs: Mapping[str, Any] = field(default_factory=dict)
    temporal_kwargs: Mapping[str, Any] = field(default_factory=dict)
    decoder_kwargs: Mapping[str, Any] = field(default_factory=dict)
    # output size
    output_size: tuple = (41, 41)


class PrecipitationNowcaster(nn.Module):
    def __init__(self, cfg: NowcasterConfig):
        super().__init__()
        self.cfg = cfg

        self.temporal: TemporalModule = build_temporal(
            cfg.temporal,
            in_channels_per_frame=cfg.in_channels_per_frame,
            n_frames=cfg.n_frames,
            n_diff_frames=cfg.n_diff_frames,
            **dict(cfg.temporal_kwargs),
        )
        self.encoder: Encoder = build_encoder(
            cfg.encoder, in_channels=self.temporal.out_channels,
            **dict(cfg.encoder_kwargs),
        )
        # ensure decoder gets output_size from cfg
        dkw = dict(cfg.decoder_kwargs)
        dkw.setdefault("output_size", cfg.output_size)
        self.decoder: Decoder = build_decoder(
            cfg.decoder, encoder_channels=self.encoder.feature_channels, **dkw,
        )

        # Aux embedding — mapped to decoder output channels and broadcast-added
        self.aux_embed = nn.Sequential(
            nn.Linear(cfg.aux_dim, cfg.aux_embed_dim),
            nn.GELU(),
            nn.Linear(cfg.aux_embed_dim, self.decoder.out_channels),
        )

        self.heads: OutputHeads = build_heads(
            in_channels=self.decoder.out_channels,
            probabilistic=cfg.probabilistic,
            hidden_channels=cfg.head_hidden_channels,
        )

    @torch.jit.ignore
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, sat: Tensor, aux: Tensor) -> Dict[str, Tensor]:
        """
        Args:
            sat: (B, C_in, H, W) float32/16
            aux: (B, aux_dim) float
        Returns:
            dict with keys `mean`, `rain_logit`, and optionally `log_var`,
            each of shape (B, output_H, output_W).
        """
        x = self.temporal(sat)
        feats = self.encoder(x)
        d = self.decoder(feats)                                     # (B, C_dec, 41, 41)
        aux_e = self.aux_embed(aux)                                  # (B, C_dec)
        d = d + aux_e[:, :, None, None]
        return self.heads(d)
