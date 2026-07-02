"""Unit tests for model modules."""
from __future__ import annotations

import pytest
import torch

torch.manual_seed(0)

from src.constants import max_active_channels
from src.models import build_model
from src.models.decoder import build_decoder
from src.models.encoder import build_encoder
from src.models.layers import (
    BasicBlock, CBAM, ConvNeXtBlock, DropPath, LayerNorm2d, MBConv,
    SqueezeExcitation, build_activation, build_norm,
)
from src.models.temporal import build_temporal


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------
def test_build_norm_activation():
    n = build_norm("group", 8)
    a = build_activation("gelu")
    x = torch.randn(2, 8, 4, 4)
    y = a(n(x))
    assert y.shape == x.shape


def test_layernorm2d():
    ln = LayerNorm2d(4)
    y = ln(torch.randn(2, 4, 3, 3))
    assert y.shape == (2, 4, 3, 3)


def test_drop_path_train_eval():
    dp = DropPath(0.5)
    x = torch.ones(4, 3, 2, 2)
    dp.train(); y = dp(x)
    assert y.shape == x.shape
    dp.eval(); y = dp(x)
    assert torch.equal(y, x)


def test_se_and_cbam_shape():
    x = torch.randn(2, 16, 8, 8)
    assert SqueezeExcitation(16)(x).shape == x.shape
    assert CBAM(16)(x).shape == x.shape


def test_basicblock_and_mbconv_shape():
    x = torch.randn(2, 16, 8, 8)
    b = BasicBlock(16, 32, stride=2)
    assert b(x).shape == (2, 32, 4, 4)
    m = MBConv(16, 32, stride=2)
    assert m(x).shape == (2, 32, 4, 4)


def test_convnext_block_shape():
    x = torch.randn(2, 96, 16, 16)
    y = ConvNeXtBlock(96, drop_path=0.1)(x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,expected", [
    ("resnet34", [64, 128, 256, 512]),
    ("efficientnet_b3", [32, 48, 136, 384]),
    ("convnext_tiny", [96, 192, 384, 768]),
])
def test_encoder_shapes(name, expected):
    enc = build_encoder(name, in_channels=20)
    x = torch.randn(1, 20, 96, 96)
    feats = enc(x)
    assert len(feats) == 4
    for f, c in zip(feats, expected):
        assert f.shape[1] == c
    assert enc.feature_channels == expected


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["none", "conv3d", "convlstm", "attention"])
def test_temporal_shapes(name):
    C = 10; T = 3; D = 2
    tm = build_temporal(name, in_channels_per_frame=C, n_frames=T, n_diff_frames=D)
    x = torch.randn(2, (T + D) * C, 32, 32)
    y = tm(x)
    assert y.shape[0] == 2
    assert y.shape[-2:] == (32, 32)
    assert y.shape[1] == tm.out_channels


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["unet", "fpn"])
def test_decoder_shapes(name):
    channels = [64, 128, 256, 512]
    dec = build_decoder(name, encoder_channels=channels, output_size=(41, 41))
    feats = [
        torch.randn(1, 64, 24, 24),
        torch.randn(1, 128, 12, 12),
        torch.randn(1, 256, 6, 6),
        torch.randn(1, 512, 3, 3),
    ]
    y = dec(feats)
    assert y.shape == (1, dec.out_channels, 41, 41)


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
def _model_cfg(**over):
    base = {
        "in_channels_per_frame": max_active_channels("ir_only"),
        "n_frames": 3, "n_diff_frames": 2,
        "encoder": "resnet34", "temporal": "none", "decoder": "unet",
        "probabilistic": False,
        "encoder_kwargs": {"norm": "group"},
        "decoder_kwargs": {"norm": "group",
                           "decoder_channels": [128, 64, 32, 16]},
    }
    base.update(over)
    return base


def test_model_forward_and_shape():
    m = build_model(_model_cfg())
    b = 2; c = m.temporal.in_channels * 5
    sat = torch.randn(b, c, 64, 64)
    aux = torch.randn(b, 6)
    y = m(sat, aux)
    assert y["mean"].shape == (b, 41, 41)
    assert y["rain_logit"].shape == (b, 41, 41)
    assert "log_var" not in y


def test_model_probabilistic():
    m = build_model(_model_cfg(probabilistic=True))
    b = 1
    c = m.temporal.in_channels * 5
    sat = torch.randn(b, c, 64, 64)
    aux = torch.randn(b, 6)
    y = m(sat, aux)
    assert "log_var" in y
    assert y["log_var"].min() >= -10.0 and y["log_var"].max() <= 5.0


def test_model_backward():
    m = build_model(_model_cfg())
    b = 2
    c = m.temporal.in_channels * 5
    sat = torch.randn(b, c, 64, 64, requires_grad=True)
    aux = torch.randn(b, 6)
    y = m(sat, aux)
    loss = y["mean"].mean() + y["rain_logit"].mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
