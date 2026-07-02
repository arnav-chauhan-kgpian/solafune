"""Model factory. Reads a plain dict/OmegaConf-like config, returns a nn.Module."""
from __future__ import annotations

from typing import Any, Mapping

from .nowcaster import NowcasterConfig, PrecipitationNowcaster


def _to_dict(x) -> dict:
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "items"):
        return {k: v for k, v in x.items()}
    return dict(x)


def build_model(cfg: Mapping[str, Any]) -> PrecipitationNowcaster:
    """Instantiate the model from a config dict.

    Expected keys:
        in_channels_per_frame, n_frames, n_diff_frames,
        encoder, temporal, decoder,
        encoder_kwargs, temporal_kwargs, decoder_kwargs,
        probabilistic, head_hidden_channels,
        aux_dim, aux_embed_dim, output_size
    """
    c = _to_dict(cfg)
    ncfg = NowcasterConfig(
        in_channels_per_frame=int(c["in_channels_per_frame"]),
        n_frames=int(c.get("n_frames", 3)),
        n_diff_frames=int(c.get("n_diff_frames", 2)),
        encoder=str(c.get("encoder", "resnet34")),
        temporal=str(c.get("temporal", "none")),
        decoder=str(c.get("decoder", "unet")),
        probabilistic=bool(c.get("probabilistic", False)),
        head_hidden_channels=int(c.get("head_hidden_channels", 0)),
        aux_dim=int(c.get("aux_dim", 6)),
        aux_embed_dim=int(c.get("aux_embed_dim", 64)),
        encoder_kwargs=_to_dict(c.get("encoder_kwargs", {})),
        temporal_kwargs=_to_dict(c.get("temporal_kwargs", {})),
        decoder_kwargs=_to_dict(c.get("decoder_kwargs", {})),
        output_size=tuple(c.get("output_size", (41, 41))),
    )
    return PrecipitationNowcaster(ncfg)
