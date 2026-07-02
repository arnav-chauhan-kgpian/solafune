"""Unit tests for metrics."""
from __future__ import annotations

import numpy as np
import torch

from src.training.metrics import MetricAccumulator, running_rmse_from_batch


def test_running_rmse_from_batch():
    p = torch.zeros(2, 4, 4)
    t = torch.ones(2, 4, 4)
    assert abs(running_rmse_from_batch(p, t) - 1.0) < 1e-6


def test_metric_accumulator_end_to_end():
    acc = MetricAccumulator(rain_threshold=0.1, heavy_threshold=5.0)
    pred = torch.tensor([
        [[0.0, 1.0], [3.0, 8.0]],
        [[0.0, 0.0], [1.0, 6.0]],
    ], dtype=torch.float32)
    target = torch.tensor([
        [[0.0, 0.5], [3.0, 8.0]],
        [[0.0, 0.1], [2.0, 6.0]],
    ], dtype=torch.float32)
    sat = torch.tensor([0, 1])
    loc = torch.tensor([0, 0])
    acc.update(pred, target, sat, loc)
    m = acc.compute()
    for k in ("rmse", "mae", "mse", "bias", "pearson", "spearman", "r2",
             "csi", "pod", "far", "ets", "rain_f1", "rain_accuracy",
             "heavy_f1", "heavy_rmse"):
        assert k in m
        assert np.isfinite(m[k])
    # per-satellite keys
    assert any(k.startswith("sat0/") for k in m)
    assert any(k.startswith("sat1/") for k in m)
    assert any(k.startswith("loc0/") for k in m)


def test_metric_accumulator_reset():
    acc = MetricAccumulator()
    acc.update(torch.zeros(1, 2, 2), torch.zeros(1, 2, 2), torch.zeros(1, dtype=torch.long),
                torch.zeros(1, dtype=torch.long))
    acc.reset()
    m = acc.compute()
    assert m == {}


def test_metric_perfect_prediction():
    acc = MetricAccumulator()
    x = torch.rand(2, 4, 4) * 10
    acc.update(x, x, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    m = acc.compute()
    assert m["rmse"] < 1e-4
    assert m["mae"] < 1e-4
    assert abs(m["bias"]) < 1e-4
