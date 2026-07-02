"""Tests for the SolafuneDataset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.constants import FRAMES_PER_SAMPLE, GPM_SIZE, SATELLITES, band_indices_for, max_active_channels
from src.data.cache import get_backend
from src.data.dataset import DatasetConfig, SolafuneDataset, split_indices_by_location
from src.data.normalization import compute_norm_stats, save_norm_stats
from src.data.preprocessing import build_cache, build_cache_spec
from src.paths import sat_tif_path
from src.utils import parse_frame_list


def _prep_cache(root: Path, csv: Path, cache_root: Path, band_mode: str = "ir_only"):
    pytest.importorskip("zarr")
    df = pd.read_csv(csv)
    spec, _ = build_cache_spec(df, cache_root, band_mode)
    backend = get_backend("zarr")(spec, compressor="lz4")
    build_cache(
        csv_path=csv, data_root=root, cache_root=cache_root,
        backend=backend, band_mode=band_mode, load_gpm=True,
    )
    backend.close()
    return spec


def _prep_norm(root: Path, csv: Path, out: Path):
    df = pd.read_csv(csv)
    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        s = row["satellite_target"]
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[s].append(sat_tif_path(root, s, f))
    stats = compute_norm_stats(paths, max_files_per_satellite=100, pixel_stride=1)
    save_norm_stats(out, stats)


def test_dataset_shapes_and_dtypes(synthetic_workspace, tmp_path):
    cache_dir = tmp_path / "cache"
    _prep_cache(synthetic_workspace["root"], synthetic_workspace["csv"], cache_dir)
    norm_path = tmp_path / "norm.json"
    _prep_norm(synthetic_workspace["root"], synthetic_workspace["csv"], norm_path)

    cfg = DatasetConfig(
        cache_dir=cache_dir,
        csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path,
        image_size=32,
        bands="ir_only",
        include_diff_frames=True,
    )
    ds = SolafuneDataset(cfg)
    assert len(ds) == synthetic_workspace["n_rows"]

    sample = ds[0]
    # channel width = max_active_c * (T + T-1)
    c_max = max_active_channels("ir_only")
    expected_c = c_max * (FRAMES_PER_SAMPLE + FRAMES_PER_SAMPLE - 1)
    assert sample["sat"].shape == (expected_c, 32, 32)
    assert sample["sat"].dtype == torch.float32
    assert sample["gpm_log1p"].shape == GPM_SIZE
    assert sample["gpm_raw"].shape == GPM_SIZE
    assert sample["rain_mask"].shape == GPM_SIZE
    assert sample["aux"].shape == (6,)
    assert isinstance(sample["unique_id"], str)


@pytest.mark.parametrize("image_size", [16, 32, 48])
def test_dataset_resize_configurable(synthetic_workspace, tmp_path, image_size):
    cache_dir = tmp_path / "cache_resize"
    _prep_cache(synthetic_workspace["root"], synthetic_workspace["csv"], cache_dir)
    norm_path = tmp_path / "norm.json"
    _prep_norm(synthetic_workspace["root"], synthetic_workspace["csv"], norm_path)
    cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path, image_size=image_size, bands="ir_only",
    )
    ds = SolafuneDataset(cfg)
    sample = ds[0]
    assert sample["sat"].shape[-2:] == (image_size, image_size)


def test_dataset_no_diffs(synthetic_workspace, tmp_path):
    cache_dir = tmp_path / "cache_nd"
    _prep_cache(synthetic_workspace["root"], synthetic_workspace["csv"], cache_dir)
    norm_path = tmp_path / "n.json"
    _prep_norm(synthetic_workspace["root"], synthetic_workspace["csv"], norm_path)
    cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path, image_size=16, bands="ir_only",
        include_diff_frames=False,
    )
    ds = SolafuneDataset(cfg)
    sample = ds[0]
    c_max = max_active_channels("ir_only")
    assert sample["sat"].shape == (c_max * FRAMES_PER_SAMPLE, 16, 16)


def test_missing_frame_strategy(synthetic_workspace_missing_frames, tmp_path):
    ws = synthetic_workspace_missing_frames
    cache_dir = tmp_path / "cache_m"
    _prep_cache(ws["root"], ws["csv"], cache_dir)
    # norm stats: only Himawari present
    df = pd.read_csv(ws["csv"])
    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[row["satellite_target"]].append(sat_tif_path(ws["root"], row["satellite_target"], f))
    stats = compute_norm_stats(paths, max_files_per_satellite=50, pixel_stride=1)
    norm_path = tmp_path / "n.json"
    save_norm_stats(norm_path, stats)

    cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=ws["csv"],
        norm_stats_path=norm_path, image_size=16, bands="ir_only",
        missing_frame_strategy="repeat_last",
    )
    ds = SolafuneDataset(cfg)
    # sample index 0: 3 frames present → has_data = 1
    s0 = ds[0]
    assert float(s0["has_data"]) == 1.0
    # sample index 3: 0 frames present → has_data = 0
    s3 = ds[3]
    assert float(s3["has_data"]) == 0.0


def test_split_indices_by_location(synthetic_workspace):
    df = pd.read_csv(synthetic_workspace["csv"])
    val_locs = [df["name_location"].iloc[0]]
    tr, va = split_indices_by_location(df, val_locs)
    assert len(va) == 1
    assert len(tr) == len(df) - 1
    assert set(tr) & set(va) == set()
