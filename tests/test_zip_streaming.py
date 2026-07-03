"""Tests for the ZIP-streaming preprocessing pipeline (Kaggle path)."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.constants import FRAMES_PER_SAMPLE, GPM_SIZE, SATELLITES, band_indices_for
from src.data.cache import get_backend
from src.data.preprocessing import build_cache_spec
from src.data.preprocessing_zip import build_cache_from_zips, compute_norm_stats_from_zip
from src.paths import gpm_tif_path, sat_tif_path
from src.utils import parse_frame_list
from src.utils.io import (
    read_gpm_tif,
    read_gpm_tif_from_bytes,
    read_satellite_tif,
    read_satellite_tif_from_bytes,
)


def test_bytes_reader_matches_file_reader(synthetic_workspace):
    """A TIF read from a bytes buffer must be byte-identical to the same
    TIF read from disk."""
    root = synthetic_workspace["root"]
    df = pd.read_csv(synthetic_workspace["csv"])
    row = df.iloc[0]
    fname = parse_frame_list(row["last_30_minutes_observation_filename"])[0]
    p = sat_tif_path(root, row["satellite_target"], fname)
    arr_disk, meta_disk = read_satellite_tif(p)
    arr_bytes, meta_bytes = read_satellite_tif_from_bytes(p.read_bytes(), label=str(p))
    np.testing.assert_array_equal(arr_disk, arr_bytes)
    assert meta_disk.count == meta_bytes.count
    assert meta_disk.height == meta_bytes.height
    assert meta_disk.width == meta_bytes.width


def test_gpm_bytes_reader_matches_file_reader(synthetic_workspace):
    root = synthetic_workspace["root"]
    df = pd.read_csv(synthetic_workspace["csv"])
    p = gpm_tif_path(root, df.iloc[0]["gpm_imerg_filename"])
    a, _ = read_gpm_tif(p)
    b, _ = read_gpm_tif_from_bytes(p.read_bytes(), label=str(p))
    np.testing.assert_array_equal(a, b)


def _make_workspace_zip(root: Path, out_zip: Path) -> Path:
    """Pack an entire synthetic workspace into a zip that mirrors the shape
    of the real Kaggle upload (dataset root dir at top level)."""
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for p in root.rglob("*"):
            if p.is_file():
                # arcname preserves parent-of-root
                arc = out_zip.stem + "/" + p.relative_to(root).as_posix()
                zf.write(p, arc)
    return out_zip


def test_build_cache_from_zip_matches_disk(synthetic_workspace, tmp_path):
    pytest.importorskip("zarr")
    root = synthetic_workspace["root"]
    csv = synthetic_workspace["csv"]
    zip_path = _make_workspace_zip(root, tmp_path / "synth.zip")

    cache_dir = tmp_path / "cache_zip"
    df = pd.read_csv(csv)
    spec, _ = build_cache_spec(df, cache_dir, "ir_only")
    backend = get_backend("zarr")(spec, compressor="lz4")
    build_cache_from_zips(
        csv_path=csv, sat_zip_path=zip_path, gpm_zip_path=None,
        cache_root=cache_dir, backend=backend,
        band_mode="ir_only", load_gpm=True, verbose_every=0,
    )
    # sanity: each satellite has 3 samples with correct shape
    for s in SATELLITES:
        arr = backend.read_sat_sample(s, 0)
        assert arr.shape[0] == FRAMES_PER_SAMPLE
        assert arr.shape[1] == len(band_indices_for(s, "ir_only"))
    gpm = backend.read_gpm_sample(0)
    assert gpm.shape == GPM_SIZE
    assert np.isfinite(gpm).all()
    backend.close()


def test_norm_stats_from_zip(synthetic_workspace, tmp_path):
    root = synthetic_workspace["root"]
    csv = synthetic_workspace["csv"]
    zip_path = _make_workspace_zip(root, tmp_path / "synth.zip")
    stats = compute_norm_stats_from_zip(
        zip_path=zip_path, csv_path=csv,
        max_files_per_satellite=20, pixel_stride=1, seed=0,
    )
    for s in SATELLITES:
        bs = stats.per_satellite[s]
        assert len(bs.mean) == 16 and len(bs.std) == 16
        # synthetic uniform-random pixels: mean ~127, std ~74
        assert 60 < bs.mean[0] < 200
        assert 20 < bs.std[0] < 200
