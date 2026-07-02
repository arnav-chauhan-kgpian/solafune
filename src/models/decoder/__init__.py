"""Decoder implementations."""
from .base import Decoder
from .unet import UNetDecoder
from .fpn import FPNDecoder


DECODERS = {
    "unet": UNetDecoder,
    "fpn": FPNDecoder,
}


def build_decoder(name: str, encoder_channels, **kwargs) -> Decoder:
    if name not in DECODERS:
        raise ValueError(f"unknown decoder: {name!r}. Available: {list(DECODERS)}")
    return DECODERS[name](encoder_channels=encoder_channels, **kwargs)


__all__ = ["Decoder", "UNetDecoder", "FPNDecoder", "build_decoder", "DECODERS"]
