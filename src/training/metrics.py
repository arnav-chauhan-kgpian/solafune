"""Metrics for precipitation nowcasting.

`MetricAccumulator` stores per-batch tensors (on CPU) and computes all
metrics at epoch end. Supports per-satellite and per-location breakdowns.

Metrics (mm/h space):
    RMSE, MAE, MSE, Bias
    Pearson r, Spearman rho, R^2
    CSI, POD, FAR, ETS (Equitable Threat Score)
    Rain/no-rain accuracy (F1), heavy-rain accuracy (>10 mm/h)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Vector metrics on flat 1D arrays
# ---------------------------------------------------------------------------
def _rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _bias(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(pred - target))


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    p = pred - pred.mean()
    t = target - target.mean()
    denom = float(np.sqrt((p ** 2).sum() * (t ** 2).sum()))
    if denom < 1e-12:
        return 0.0
    return float((p * t).sum() / denom)


def _spearman(pred: np.ndarray, target: np.ndarray, max_n: int = 200_000) -> float:
    if pred.size > max_n:
        idx = np.random.default_rng(0).choice(pred.size, size=max_n, replace=False)
        pred = pred[idx]; target = target[idx]
    rp = _rankdata(pred)
    rt = _rankdata(target)
    return _pearson(rp, rt)


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(x.size, dtype=np.float64)
    return ranks


# ---------------------------------------------------------------------------
# Categorical / rain-event metrics
# ---------------------------------------------------------------------------
def _confusion(pred_bin: np.ndarray, true_bin: np.ndarray) -> Dict[str, int]:
    tp = int(np.sum(pred_bin & true_bin))
    fp = int(np.sum(pred_bin & ~true_bin))
    fn = int(np.sum(~pred_bin & true_bin))
    tn = int(np.sum(~pred_bin & ~true_bin))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _csi(cm: Dict[str, int]) -> float:
    d = cm["tp"] + cm["fp"] + cm["fn"]
    return cm["tp"] / d if d else 0.0


def _pod(cm: Dict[str, int]) -> float:
    d = cm["tp"] + cm["fn"]
    return cm["tp"] / d if d else 0.0


def _far(cm: Dict[str, int]) -> float:
    d = cm["tp"] + cm["fp"]
    return cm["fp"] / d if d else 0.0


def _f1(cm: Dict[str, int]) -> float:
    p = cm["tp"] / (cm["tp"] + cm["fp"]) if (cm["tp"] + cm["fp"]) else 0.0
    r = cm["tp"] / (cm["tp"] + cm["fn"]) if (cm["tp"] + cm["fn"]) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _ets(cm: Dict[str, int]) -> float:
    tp = cm["tp"]; fp = cm["fp"]; fn = cm["fn"]; tn = cm["tn"]
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    hits_random = (tp + fp) * (tp + fn) / n
    denom = tp + fp + fn - hits_random
    if abs(denom) < 1e-12:
        return 0.0
    return (tp - hits_random) / denom


# ---------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------
@dataclass
class MetricAccumulator:
    rain_threshold: float = 0.1
    heavy_threshold: float = 10.0
    _preds: List[np.ndarray] = field(default_factory=list)
    _targets: List[np.ndarray] = field(default_factory=list)
    _sat_ids: List[np.ndarray] = field(default_factory=list)
    _loc_ids: List[np.ndarray] = field(default_factory=list)

    def reset(self) -> None:
        self._preds.clear()
        self._targets.clear()
        self._sat_ids.clear()
        self._loc_ids.clear()

    def update(
        self,
        pred_mm: Tensor,       # (B, H, W) float, in mm/h
        target_mm: Tensor,     # (B, H, W) float, in mm/h
        sat_id: Tensor,        # (B,) int
        location_id: Tensor,   # (B,) int
    ) -> None:
        b = pred_mm.shape[0]
        assert target_mm.shape == pred_mm.shape
        self._preds.append(pred_mm.detach().cpu().numpy().reshape(b, -1))
        self._targets.append(target_mm.detach().cpu().numpy().reshape(b, -1))
        self._sat_ids.append(sat_id.detach().cpu().numpy().reshape(b))
        self._loc_ids.append(location_id.detach().cpu().numpy().reshape(b))

    def _all(self):
        return (
            np.concatenate(self._preds, axis=0) if self._preds else np.zeros((0, 1), dtype=np.float32),
            np.concatenate(self._targets, axis=0) if self._targets else np.zeros((0, 1), dtype=np.float32),
            np.concatenate(self._sat_ids, axis=0) if self._sat_ids else np.zeros((0,), dtype=np.int64),
            np.concatenate(self._loc_ids, axis=0) if self._loc_ids else np.zeros((0,), dtype=np.int64),
        )

    def _compute_scalar(self, pred_flat: np.ndarray, target_flat: np.ndarray) -> Dict[str, float]:
        rt = float(self.rain_threshold)
        ht = float(self.heavy_threshold)
        m: Dict[str, float] = {
            "rmse": _rmse(pred_flat, target_flat),
            "mae": _mae(pred_flat, target_flat),
            "mse": _mse(pred_flat, target_flat),
            "bias": _bias(pred_flat, target_flat),
            "r2": _r2(pred_flat, target_flat),
            "pearson": _pearson(pred_flat, target_flat),
            "spearman": _spearman(pred_flat, target_flat),
        }
        pred_bin = pred_flat > rt
        true_bin = target_flat > rt
        cm = _confusion(pred_bin, true_bin)
        m["csi"] = _csi(cm)
        m["pod"] = _pod(cm)
        m["far"] = _far(cm)
        m["ets"] = _ets(cm)
        m["rain_f1"] = _f1(cm)
        # rain/no-rain accuracy
        m["rain_accuracy"] = float(np.mean(pred_bin == true_bin))
        # heavy
        pred_h = pred_flat > ht
        true_h = target_flat > ht
        cm_h = _confusion(pred_h, true_h)
        m["heavy_f1"] = _f1(cm_h)
        m["heavy_rmse"] = _rmse(pred_flat[true_h], target_flat[true_h]) if true_h.any() else 0.0
        return m

    def compute(self) -> Dict[str, float]:
        pred, target, sat, loc = self._all()
        if pred.size == 0:
            return {}
        out: Dict[str, float] = {}
        flat_p = pred.reshape(-1); flat_t = target.reshape(-1)
        out.update(self._compute_scalar(flat_p, flat_t))
        # per-satellite
        for sid in np.unique(sat):
            m = sat == sid
            sub_p = pred[m].reshape(-1); sub_t = target[m].reshape(-1)
            if sub_p.size == 0:
                continue
            for k, v in self._compute_scalar(sub_p, sub_t).items():
                out[f"sat{int(sid)}/{k}"] = v
        # per-location
        for lid in np.unique(loc):
            m = loc == lid
            sub_p = pred[m].reshape(-1); sub_t = target[m].reshape(-1)
            if sub_p.size == 0:
                continue
            for k, v in self._compute_scalar(sub_p, sub_t).items():
                out[f"loc{int(lid)}/{k}"] = v
        return out


def running_rmse_from_batch(pred_mm: Tensor, target_mm: Tensor) -> float:
    """Quick per-batch RMSE for progress logging."""
    return float(torch.sqrt(((pred_mm - target_mm) ** 2).mean()).item())
