"""Unit tests for EMA."""
from __future__ import annotations

import copy
import torch
from torch import nn

from src.training.ema import ExponentialMovingAverage


def test_ema_updates_and_restores():
    model = nn.Sequential(nn.Linear(4, 4))
    ema = ExponentialMovingAverage(model, decay=0.5)
    orig = {n: p.detach().clone() for n, p in model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.0)
    ema.update(model)
    # shadow should now be halfway between orig and zero
    for n, p in model.named_parameters():
        expected = orig[n] * 0.5
        assert torch.allclose(ema.shadow[n], expected, atol=1e-6)

    # apply context: model params become shadow, then revert
    saved = {n: p.detach().clone() for n, p in model.named_parameters()}
    with ema.apply(model):
        for n, p in model.named_parameters():
            assert torch.allclose(p, ema.shadow[n].to(p.dtype), atol=1e-5)
    for n, p in model.named_parameters():
        assert torch.allclose(p, saved[n])


def test_ema_state_dict_roundtrip():
    m = nn.Linear(3, 3)
    e = ExponentialMovingAverage(m, decay=0.9)
    s = e.state_dict()
    e2 = ExponentialMovingAverage(m, decay=0.9)
    e2.load_state_dict(s)
    for k in e.shadow:
        assert torch.allclose(e.shadow[k], e2.shadow[k])
