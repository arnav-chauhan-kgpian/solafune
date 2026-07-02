"""LR scheduler factories with warmup support."""
from __future__ import annotations

from typing import Any, Mapping

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LambdaLR, OneCycleLR, ReduceLROnPlateau, SequentialLR,
)


def build_optimizer(model: torch.nn.Module, cfg: Mapping[str, Any]) -> Optimizer:
    name = str(cfg.get("name", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))
        eps = float(cfg.get("eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd,
                                  betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                               weight_decay=wd, nesterov=bool(cfg.get("nesterov", True)))
    raise ValueError(f"unknown optimizer: {name!r}")


def build_scheduler(optimizer: Optimizer, cfg: Mapping[str, Any],
                    steps_per_epoch: int, epochs: int):
    """Return (scheduler, step_each_batch: bool).

    * step_each_batch=True → call scheduler.step() every optimizer step
    * step_each_batch=False → call once per epoch after validate()
    """
    name = str(cfg.get("name", "onecycle")).lower()
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    max_lr = float(cfg.get("max_lr", 1e-3))
    total_steps = int(steps_per_epoch * epochs)

    if name == "onecycle":
        sched = OneCycleLR(
            optimizer, max_lr=max_lr,
            total_steps=max(total_steps, 1),
            pct_start=float(cfg.get("pct_start", 0.3)),
            div_factor=float(cfg.get("div_factor", 25.0)),
            final_div_factor=float(cfg.get("final_div_factor", 1e4)),
            anneal_strategy=str(cfg.get("anneal_strategy", "cos")),
        )
        return sched, True
    if name == "cosine":
        base = CosineAnnealingLR(optimizer, T_max=max(epochs, 1),
                                  eta_min=float(cfg.get("eta_min", 1e-6)))
        if warmup_epochs > 0:
            warm = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
            sched = SequentialLR(optimizer, schedulers=[warm, base],
                                 milestones=[warmup_epochs])
        else:
            sched = base
        return sched, False
    if name == "plateau":
        sched = ReduceLROnPlateau(
            optimizer, mode=str(cfg.get("mode", "min")),
            factor=float(cfg.get("factor", 0.5)),
            patience=int(cfg.get("patience", 3)),
            min_lr=float(cfg.get("min_lr", 1e-6)),
            threshold=float(cfg.get("threshold", 1e-4)),
        )
        return sched, False
    if name == "constant":
        return LambdaLR(optimizer, lr_lambda=lambda e: 1.0), False
    raise ValueError(f"unknown scheduler: {name!r}")
