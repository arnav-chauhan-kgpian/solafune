"""Exponential moving average of model parameters."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict

import torch
from torch import nn


class ExponentialMovingAverage:
    """Maintains an EMA of a model's parameters (and buffers, optionally).

    Usage::

        ema = ExponentialMovingAverage(model, decay=0.9999)
        # every optimizer step:
        ema.update(model)
        # at validation:
        with ema.apply(model):
            validate(model)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 include_buffers: bool = False):
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.include_buffers = include_buffers
        self.shadow: Dict[str, torch.Tensor] = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
        } if include_buffers else {
            n: p.detach().clone().float() for n, p in model.named_parameters()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        if self.include_buffers:
            src = model.state_dict()
            for k, v in src.items():
                if v.dtype.is_floating_point:
                    self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
                else:
                    self.shadow[k].copy_(v)
        else:
            for n, p in model.named_parameters():
                if p.dtype.is_floating_point:
                    self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        for k in self.shadow:
            if k in state:
                self.shadow[k] = state[k].clone().float()

    @contextmanager
    def apply(self, model: nn.Module):
        """Temporarily swap model params with EMA weights."""
        backup: Dict[str, torch.Tensor] = {}
        if self.include_buffers:
            src_iter = list(model.state_dict().items())
            for k, v in src_iter:
                if k in self.shadow:
                    backup[k] = v.detach().clone()
                    v.copy_(self.shadow[k].to(v.dtype).to(v.device))
        else:
            for n, p in model.named_parameters():
                if n in self.shadow:
                    backup[n] = p.detach().clone()
                    p.data.copy_(self.shadow[n].to(p.dtype).to(p.device))
        try:
            yield
        finally:
            if self.include_buffers:
                sd = model.state_dict()
                for k, v in backup.items():
                    sd[k].copy_(v.to(sd[k].dtype).to(sd[k].device))
            else:
                params = dict(model.named_parameters())
                for n, v in backup.items():
                    params[n].data.copy_(v.to(params[n].dtype).to(params[n].device))
