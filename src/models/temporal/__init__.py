"""Temporal fusion modules.

Every temporal module takes a 4D tensor (B, C_in, H, W) where the temporal
axis is flattened into the channel dim by the Dataset, and outputs a 4D
tensor (B, C_out, H, W). The module reports both `in_channels` and
`out_channels`.
"""
from .base import TemporalModule
from .none import PassthroughTemporal
from .conv3d import Conv3DStem
from .convlstm import ConvLSTMTemporal
from .attention import TemporalAttention


TEMPORAL_MODULES = {
    "none": PassthroughTemporal,
    "conv3d": Conv3DStem,
    "convlstm": ConvLSTMTemporal,
    "attention": TemporalAttention,
}


def build_temporal(name: str, in_channels_per_frame: int, n_frames: int,
                   n_diff_frames: int = 0, **kwargs) -> TemporalModule:
    if name not in TEMPORAL_MODULES:
        raise ValueError(f"unknown temporal module: {name!r}. "
                         f"Available: {list(TEMPORAL_MODULES)}")
    return TEMPORAL_MODULES[name](
        in_channels_per_frame=in_channels_per_frame,
        n_frames=n_frames,
        n_diff_frames=n_diff_frames,
        **kwargs,
    )


__all__ = ["TemporalModule", "PassthroughTemporal", "Conv3DStem",
           "ConvLSTMTemporal", "TemporalAttention",
           "build_temporal", "TEMPORAL_MODULES"]
