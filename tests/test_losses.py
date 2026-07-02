"""Unit tests for loss functions."""
from __future__ import annotations

import pytest
import torch

from src.training.losses import (
    CompositeLoss, LossConfig, bce_loss, build_loss, dice_loss,
    gaussian_nll_loss, gradient_loss, huber_loss, mae_loss, mse_loss,
    rain_weight, smooth_l1_loss, ssim_loss,
)


def _sample():
    torch.manual_seed(0)
    pred = torch.randn(2, 41, 41, requires_grad=True)
    target = torch.randn(2, 41, 41)
    return pred, target


@pytest.mark.parametrize("fn", [mse_loss, mae_loss])
def test_regression_losses_backward(fn):
    p, t = _sample()
    l = fn(p, t)
    assert l.ndim == 0 and torch.isfinite(l)
    l.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_smooth_l1_huber():
    p, t = _sample()
    assert torch.isfinite(smooth_l1_loss(p, t))
    assert torch.isfinite(huber_loss(p, t, delta=0.5))


def test_bce_and_focal():
    logit = torch.randn(2, 41, 41, requires_grad=True)
    mask = (torch.rand(2, 41, 41) > 0.7).float()
    l1 = bce_loss(logit, mask)
    l2 = bce_loss(logit, mask, focal_gamma=2.0)
    assert torch.isfinite(l1) and torch.isfinite(l2)
    l2.backward()
    assert torch.isfinite(logit.grad).all()


def test_nll_and_gradient_and_ssim():
    p, t = _sample()
    lv = torch.zeros_like(p, requires_grad=True)
    assert torch.isfinite(gaussian_nll_loss(p.detach(), lv, t))
    assert torch.isfinite(gradient_loss(p, t))
    assert torch.isfinite(ssim_loss(p, t))


def test_dice():
    rp = torch.rand(2, 41, 41)
    rm = (torch.rand(2, 41, 41) > 0.5).float()
    assert torch.isfinite(dice_loss(rp, rm))


def test_rain_weight_positive():
    y = torch.tensor([[0.0, 1.0], [2.0, 5.0]])
    w = rain_weight(y, scale=3.0)
    assert torch.all(w >= 1.0)


def test_composite_loss_mask_and_finite():
    torch.manual_seed(0)
    pred = {
        "mean": torch.randn(2, 41, 41, requires_grad=True),
        "rain_logit": torch.randn(2, 41, 41, requires_grad=True),
        "log_var": torch.zeros(2, 41, 41, requires_grad=True),
    }
    batch = {
        "gpm_log1p": torch.rand(2, 41, 41) * 2.0,
        "gpm_raw": torch.rand(2, 41, 41) * 5.0,
        "rain_mask": (torch.rand(2, 41, 41) > 0.7).float(),
        "has_data": torch.tensor([1.0, 0.0]),
    }
    cfg = LossConfig(mse_weight=1.0, bce_weight=0.5, nll_weight=0.3,
                     gradient_weight=0.01, ssim_weight=0.05,
                     rain_weighted=True, mask_missing_frames=True)
    loss_fn = CompositeLoss(cfg)
    losses = loss_fn(pred, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert torch.isfinite(pred["mean"].grad).all()


def test_composite_all_zero_weights():
    """All weights zero → total is a scalar zero tensor."""
    pred = {
        "mean": torch.randn(1, 41, 41, requires_grad=True),
        "rain_logit": torch.randn(1, 41, 41, requires_grad=True),
    }
    batch = {
        "gpm_log1p": torch.zeros(1, 41, 41),
        "gpm_raw": torch.zeros(1, 41, 41),
        "rain_mask": torch.zeros(1, 41, 41),
        "has_data": torch.tensor([1.0]),
    }
    loss = CompositeLoss(LossConfig(mse_weight=0, bce_weight=0))(pred, batch)
    assert torch.equal(loss["total"], torch.zeros(()))


def test_build_loss_ignores_unknown_keys():
    fn = build_loss({"mse_weight": 1.0, "bce_weight": 0.5, "unknown_key": 42})
    assert isinstance(fn, CompositeLoss)
