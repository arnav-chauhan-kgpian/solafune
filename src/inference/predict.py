"""Batched inference with optional TTA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


@dataclass
class PredictionConfig:
    amp: bool = True
    tta: bool = True
    rain_mask_threshold: float = 0.15
    ensemble_paths: Optional[List[str]] = None


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    cfg: PredictionConfig,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    preds: List[np.ndarray] = []
    uids: List[str] = []
    for batch in loader:
        sat = batch["sat"].to(device, non_blocking=True)
        aux = batch["aux"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
            out = model(sat, aux)
            mm = torch.expm1(out["mean"].float().clamp_min(0.0))
            rp = torch.sigmoid(out["rain_logit"].float())
            mm = mm * (rp > cfg.rain_mask_threshold).float()
            if cfg.tta:
                sat_f = torch.flip(sat, dims=[-1])
                out_f = model(sat_f, aux)
                mm_f = torch.expm1(out_f["mean"].float().clamp_min(0.0))
                rp_f = torch.sigmoid(out_f["rain_logit"].float())
                mm_f = mm_f * (rp_f > cfg.rain_mask_threshold).float()
                mm = 0.5 * (mm + torch.flip(mm_f, dims=[-1]))
        preds.append(mm.detach().cpu().numpy())
        uids.extend(batch.get("unique_id", []))
    result = np.concatenate(preds, axis=0)
    return result
