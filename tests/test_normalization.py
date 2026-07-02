"""Tests for src/data/normalization.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import NUM_BANDS_TOTAL, SATELLITES
from src.data.normalization import (
    NormStats,
    compute_norm_stats,
    load_norm_stats,
    normalize_frame,
    save_norm_stats,
)
from src.paths import sat_tif_path
from src.utils import parse_frame_list


def _collect_paths(root, csv):
    df = pd.read_csv(csv)
    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        sat = row["satellite_target"]
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[sat].append(sat_tif_path(root, sat, f))
    return paths


def test_compute_norm_stats_end_to_end(synthetic_workspace, tmp_path):
    paths = _collect_paths(synthetic_workspace["root"], synthetic_workspace["csv"])
    stats = compute_norm_stats(paths, max_files_per_satellite=100, pixel_stride=1)
    for s in SATELLITES:
        bs = stats.per_satellite[s]
        assert len(bs.mean) == NUM_BANDS_TOTAL
        assert len(bs.std) == NUM_BANDS_TOTAL
        # synthetic data is uniform random in [0, 255] → mean ~127, std ~74
        arr_mean = np.asarray(bs.mean)
        arr_std = np.asarray(bs.std)
        assert (arr_mean > 50).all() and (arr_mean < 200).all()
        assert (arr_std > 20).all() and (arr_std < 200).all()


def test_save_and_load_roundtrip(tmp_path):
    stats = NormStats(per_satellite={
        s: type("BS", (), {"mean": [0.0]*NUM_BANDS_TOTAL, "std": [1.0]*NUM_BANDS_TOTAL, "n_pixels": 10})()
        for s in SATELLITES
    })
    # use real dataclass via factory
    from src.data.normalization import BandStats
    stats = NormStats(per_satellite={s: BandStats(mean=[float(i)]*NUM_BANDS_TOTAL, std=[1.0]*NUM_BANDS_TOTAL, n_pixels=10) for i, s in enumerate(SATELLITES)})
    path = tmp_path / "n.json"
    save_norm_stats(path, stats)
    back = load_norm_stats(path)
    for s in SATELLITES:
        assert back.per_satellite[s].mean == stats.per_satellite[s].mean
        assert back.per_satellite[s].std == stats.per_satellite[s].std


def test_normalize_frame_shapes():
    from src.data.normalization import BandStats
    mean = np.arange(4, dtype=np.float32)
    std = np.ones(4, dtype=np.float32)
    x3 = np.ones((4, 5, 5), dtype=np.uint8) * 3
    n3 = normalize_frame(x3, mean, std)
    assert n3.shape == (4, 5, 5)
    assert n3.dtype == np.float32
    # channel 0 = (3 - 0)/1 = 3
    assert n3[0, 0, 0] == pytest.approx(3.0)
    x4 = np.ones((2, 4, 5, 5), dtype=np.uint8) * 3
    n4 = normalize_frame(x4, mean, std)
    assert n4.shape == (2, 4, 5, 5)
