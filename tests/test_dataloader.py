"""Tests for the DataLoader factory."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.constants import SATELLITES
from src.data.cache import get_backend
from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
from src.data.dataset import DatasetConfig, SolafuneDataset
from src.data.normalization import compute_norm_stats, save_norm_stats
from src.data.preprocessing import build_cache, build_cache_spec
from src.paths import sat_tif_path
from src.utils import parse_frame_list


def _prep(root, csv, tmp_path):
    pytest.importorskip("zarr")
    cache_root = tmp_path / "cache"
    df = pd.read_csv(csv)
    spec, _ = build_cache_spec(df, cache_root, "ir_only")
    backend = get_backend("zarr")(spec, compressor="lz4")
    build_cache(csv, root, cache_root, backend, "ir_only", load_gpm=True)
    backend.close()
    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[row["satellite_target"]].append(sat_tif_path(root, row["satellite_target"], f))
    stats = compute_norm_stats(paths, max_files_per_satellite=50, pixel_stride=1)
    norm_path = tmp_path / "n.json"
    save_norm_stats(norm_path, stats)
    return cache_root, norm_path


def test_dataloader_single_worker(synthetic_workspace, tmp_path):
    cache_root, norm_path = _prep(
        synthetic_workspace["root"], synthetic_workspace["csv"], tmp_path
    )
    ds_cfg = DatasetConfig(
        cache_dir=cache_root, csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path, image_size=16, bands="ir_only",
    )
    ds = SolafuneDataset(ds_cfg)
    dl_cfg = DataLoaderConfig(batch_size=3, num_workers=0, pin_memory=False,
                              drop_last=False, persistent_workers=False)
    dl = build_dataloader(ds, dl_cfg, base_seed=0)
    batch = next(iter(dl))
    assert batch["sat"].shape[0] == 3
    assert batch["gpm_log1p"].shape[0] == 3
    assert isinstance(batch["unique_id"], list)
    assert len(batch["unique_id"]) == 3


def test_dataloader_multi_worker(synthetic_workspace, tmp_path):
    cache_root, norm_path = _prep(
        synthetic_workspace["root"], synthetic_workspace["csv"], tmp_path
    )
    ds_cfg = DatasetConfig(
        cache_dir=cache_root, csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path, image_size=16, bands="ir_only",
    )
    ds = SolafuneDataset(ds_cfg)
    dl_cfg = DataLoaderConfig(batch_size=3, num_workers=2, pin_memory=False,
                              persistent_workers=False, drop_last=False, prefetch_factor=2)
    dl = build_dataloader(ds, dl_cfg, base_seed=0)
    it = iter(dl)
    batch = next(it)
    assert batch["sat"].shape[0] == 3


def test_precip_sampler(synthetic_workspace, tmp_path):
    cache_root, norm_path = _prep(
        synthetic_workspace["root"], synthetic_workspace["csv"], tmp_path
    )
    ds_cfg = DatasetConfig(
        cache_dir=cache_root, csv_path=synthetic_workspace["csv"],
        norm_stats_path=norm_path, image_size=16, bands="ir_only",
    )
    ds = SolafuneDataset(ds_cfg)
    sampler = build_sampler(ds, "precip_stratified", precip_weight_scale=3.0)
    indices = list(sampler)
    assert len(indices) == len(ds)
    assert all(0 <= i < len(ds) for i in indices)
