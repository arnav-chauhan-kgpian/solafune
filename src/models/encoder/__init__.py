"""Encoder implementations. Each exposes a common interface:

    encoder = build_encoder(name, in_channels, **cfg)
    features = encoder(x)   # returns 4 tensors at strides [4, 8, 16, 32]
    encoder.feature_channels   # List[int] channel counts at those strides
"""
from .base import Encoder
from .resnet34 import ResNet34Encoder
from .efficientnet_b3 import EfficientNetB3Encoder
from .convnext_tiny import ConvNeXtTinyEncoder


ENCODERS = {
    "resnet34": ResNet34Encoder,
    "efficientnet_b3": EfficientNetB3Encoder,
    "convnext_tiny": ConvNeXtTinyEncoder,
}


def build_encoder(name: str, in_channels: int, **kwargs) -> Encoder:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder: {name!r}. Available: {list(ENCODERS)}")
    return ENCODERS[name](in_channels=in_channels, **kwargs)


__all__ = ["Encoder", "ResNet34Encoder", "EfficientNetB3Encoder",
           "ConvNeXtTinyEncoder", "build_encoder", "ENCODERS"]
