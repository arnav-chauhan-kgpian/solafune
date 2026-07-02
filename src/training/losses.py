"""Loss functions for precipitation nowcasting.

All losses operate on model predictions (dict with "mean", "rain_logit",
optionally "log_var") and a batch dict (with "gpm_log1p", "gpm_raw",
"rain_mask", "has_data"). Individual losses return a scalar tensor and are
composed via `CompositeLoss` which applies masking and weighting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _masked_mean(x: Tensor, valid: Optional[Tensor]) -> Tensor:
    """x: (B, ...) per-sample loss values. valid: (B,) 0/1 or None."""
    if valid is None:
        return x.mean()
    # per-sample reduce to (B,)
    dims = tuple(range(1, x.ndim))
    per_sample = x.mean(dim=dims) if dims else x
    denom = valid.sum().clamp_min(1.0)
    return (per_sample * valid).sum() / denom


def _pair(pred: Tensor, target: Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs target {target.shape}")


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------
def mse_loss(pred: Tensor, target: Tensor,
             weight: Optional[Tensor] = None,
             valid: Optional[Tensor] = None) -> Tensor:
    _pair(pred, target)
    err = (pred - target) ** 2
    if weight is not None:
        err = err * weight
    return _masked_mean(err, valid)


def mae_loss(pred: Tensor, target: Tensor,
             weight: Optional[Tensor] = None,
             valid: Optional[Tensor] = None) -> Tensor:
    _pair(pred, target)
    err = (pred - target).abs()
    if weight is not None:
        err = err * weight
    return _masked_mean(err, valid)


def smooth_l1_loss(pred: Tensor, target: Tensor,
                   beta: float = 1.0,
                   valid: Optional[Tensor] = None) -> Tensor:
    _pair(pred, target)
    err = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return _masked_mean(err, valid)


def huber_loss(pred: Tensor, target: Tensor,
               delta: float = 1.0,
               valid: Optional[Tensor] = None) -> Tensor:
    _pair(pred, target)
    err = F.huber_loss(pred, target, delta=delta, reduction="none")
    return _masked_mean(err, valid)


def rain_weight(target: Tensor, scale: float = 3.0) -> Tensor:
    """Per-pixel weight w = 1 + scale * target (target is expected in log1p space)."""
    return 1.0 + scale * target.clamp_min(0.0)


def bce_loss(rain_logit: Tensor, rain_mask: Tensor,
             valid: Optional[Tensor] = None,
             focal_gamma: float = 0.0) -> Tensor:
    _pair(rain_logit, rain_mask)
    bce = F.binary_cross_entropy_with_logits(rain_logit, rain_mask, reduction="none")
    if focal_gamma > 0:
        p = torch.sigmoid(rain_logit)
        pt = torch.where(rain_mask == 1, p, 1 - p)
        bce = bce * (1 - pt).pow(focal_gamma)
    return _masked_mean(bce, valid)


def gaussian_nll_loss(mean: Tensor, log_var: Tensor, target: Tensor,
                       valid: Optional[Tensor] = None) -> Tensor:
    _pair(mean, target)
    _pair(log_var, target)
    nll = 0.5 * (log_var + (mean - target) ** 2 / log_var.exp())
    return _masked_mean(nll, valid)


def gradient_loss(pred: Tensor, target: Tensor,
                  valid: Optional[Tensor] = None) -> Tensor:
    """L1 loss between Sobel gradients of pred and target."""
    _pair(pred, target)
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    kx = pred.new_tensor([[[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]])
    ky = pred.new_tensor([[[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]])
    gx_p = F.conv2d(pred, kx, padding=1)
    gy_p = F.conv2d(pred, ky, padding=1)
    gx_t = F.conv2d(target, kx, padding=1)
    gy_t = F.conv2d(target, ky, padding=1)
    err = (gx_p - gx_t).abs() + (gy_p - gy_t).abs()
    if squeeze:
        err = err.squeeze(1)
    return _masked_mean(err, valid)


def ssim_loss(pred: Tensor, target: Tensor,
              window_size: int = 7, C1: float = 0.01 ** 2, C2: float = 0.03 ** 2,
              valid: Optional[Tensor] = None) -> Tensor:
    """1 - SSIM. Operates on (B, H, W) or (B, 1, H, W) tensors, single channel."""
    _pair(pred, target)
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        squeeze = True
    else:
        squeeze = False
    k = torch.ones(1, 1, window_size, window_size, device=pred.device,
                   dtype=pred.dtype) / (window_size * window_size)
    mu_p = F.conv2d(pred, k, padding=window_size // 2)
    mu_t = F.conv2d(target, k, padding=window_size // 2)
    mu_p2 = mu_p * mu_p; mu_t2 = mu_t * mu_t; mu_pt = mu_p * mu_t
    sig_p = F.conv2d(pred * pred, k, padding=window_size // 2) - mu_p2
    sig_t = F.conv2d(target * target, k, padding=window_size // 2) - mu_t2
    sig_pt = F.conv2d(pred * target, k, padding=window_size // 2) - mu_pt
    num = (2 * mu_pt + C1) * (2 * sig_pt + C2)
    den = (mu_p2 + mu_t2 + C1) * (sig_p + sig_t + C2)
    ssim = num / den.clamp_min(1e-8)
    ssim = ssim.clamp(-1.0, 1.0)
    loss = 1.0 - ssim
    if squeeze:
        loss = loss.squeeze(1)
    return _masked_mean(loss, valid)


def dice_loss(rain_prob: Tensor, rain_mask: Tensor,
              valid: Optional[Tensor] = None,
              smooth: float = 1.0) -> Tensor:
    _pair(rain_prob, rain_mask)
    dims = tuple(range(1, rain_prob.ndim))
    inter = (rain_prob * rain_mask).sum(dim=dims)
    total = rain_prob.sum(dim=dims) + rain_mask.sum(dim=dims)
    d = 1.0 - (2 * inter + smooth) / (total + smooth)
    if valid is not None:
        return (d * valid).sum() / valid.sum().clamp_min(1.0)
    return d.mean()


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------
@dataclass
class LossConfig:
    # regression on log1p target
    mse_weight: float = 1.0
    mae_weight: float = 0.0
    smooth_l1_weight: float = 0.0
    huber_weight: float = 0.0
    huber_delta: float = 1.0
    # rain-weighting
    rain_weighted: bool = True
    rain_weight_scale: float = 3.0
    # classification
    bce_weight: float = 0.5
    focal_gamma: float = 0.0
    # probabilistic
    nll_weight: float = 0.0
    # structural
    gradient_weight: float = 0.0
    ssim_weight: float = 0.0
    dice_weight: float = 0.0
    # masking
    mask_missing_frames: bool = True


class CompositeLoss(nn.Module):
    """Composes the individual losses per config. Returns:

        {"total": scalar, "mse": ..., "bce": ..., ...}
    """

    def __init__(self, cfg: LossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, pred: Dict[str, Tensor], batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        cfg = self.cfg
        y = batch["gpm_log1p"]
        mu = pred["mean"]
        rain_mask = batch["rain_mask"]
        rain_logit = pred["rain_logit"]
        valid = batch.get("has_data") if cfg.mask_missing_frames else None
        weight = rain_weight(y, cfg.rain_weight_scale) if cfg.rain_weighted else None

        out: Dict[str, Tensor] = {}
        total = mu.new_zeros(())
        if cfg.mse_weight > 0:
            out["mse"] = mse_loss(mu, y, weight=weight, valid=valid)
            total = total + cfg.mse_weight * out["mse"]
        if cfg.mae_weight > 0:
            out["mae"] = mae_loss(mu, y, weight=weight, valid=valid)
            total = total + cfg.mae_weight * out["mae"]
        if cfg.smooth_l1_weight > 0:
            out["smooth_l1"] = smooth_l1_loss(mu, y, valid=valid)
            total = total + cfg.smooth_l1_weight * out["smooth_l1"]
        if cfg.huber_weight > 0:
            out["huber"] = huber_loss(mu, y, delta=cfg.huber_delta, valid=valid)
            total = total + cfg.huber_weight * out["huber"]
        if cfg.bce_weight > 0:
            out["bce"] = bce_loss(rain_logit, rain_mask, valid=valid,
                                   focal_gamma=cfg.focal_gamma)
            total = total + cfg.bce_weight * out["bce"]
        if cfg.nll_weight > 0 and "log_var" in pred:
            out["nll"] = gaussian_nll_loss(mu, pred["log_var"], y, valid=valid)
            total = total + cfg.nll_weight * out["nll"]
        if cfg.gradient_weight > 0:
            out["gradient"] = gradient_loss(mu, y, valid=valid)
            total = total + cfg.gradient_weight * out["gradient"]
        if cfg.ssim_weight > 0:
            out["ssim"] = ssim_loss(mu, y, valid=valid)
            total = total + cfg.ssim_weight * out["ssim"]
        if cfg.dice_weight > 0:
            out["dice"] = dice_loss(torch.sigmoid(rain_logit), rain_mask, valid=valid)
            total = total + cfg.dice_weight * out["dice"]

        out["total"] = total
        return out


def build_loss(cfg_dict: Mapping) -> CompositeLoss:
    d = dict(cfg_dict)
    known = LossConfig.__dataclass_fields__
    filtered = {k: v for k, v in d.items() if k in known}
    return CompositeLoss(LossConfig(**filtered))
