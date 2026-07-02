"""Tests for cache backends and preprocessing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.constants import FRAMES_PER_SAMPLE, GPM_SIZE, NATIVE_SIZES, SATELLITES, band_indices_for
from src.data.cache import BACKENDS, get_backend
from src.data.cache.base import CacheSpec
from src.data.cache.benchmark import run_benchmark
from src.data.preprocessing import build_cache, build_cache_spec, verify_cache


@pytest.mark.parametrize("backend_name", ["zarr", "memmap"])
def test_backend_roundtrip(tmp_path, backend_name):
    if backend_name == "zarr":
        pytest.importorskip("zarr")
    n_him = 3
    spec = CacheSpec(
        root=tmp_path / backend_name,
        n_total=n_him,
        per_sat_counts={"himawari": n_him, "goes": 0, "meteosat": 0},
        per_sat_shapes={
            "himawari": (FRAMES_PER_SAMPLE, 4, 8, 8),
            "goes": (FRAMES_PER_SAMPLE, 4, 1, 1),
            "meteosat": (FRAMES_PER_SAMPLE, 4, 1, 1),
        },
    )
    cls = get_backend(backend_name)
    if backend_name == "zarr":
        b = cls(spec, compressor="lz4", chunk_size=2)
    else:
        b = cls(spec)
    b.create()
    rng = np.random.default_rng(0)
    for i in range(n_him):
        sample = rng.integers(0, 255, size=(FRAMES_PER_SAMPLE, 4, 8, 8), dtype=np.uint8)
        b.write_sat_sample("himawari", i, sample)
        gpm = rng.random(GPM_SIZE, dtype=np.float32).astype(np.float32)
        b.write_gpm_sample(i, gpm)
        b.write_valid_mask(i, np.array([1, 1, 1], dtype=np.uint8))
    b.flush()
    for i in range(n_him):
        s = b.read_sat_sample("himawari", i)
        assert s.dtype == np.uint8
        assert s.shape == (FRAMES_PER_SAMPLE, 4, 8, 8)
        assert (b.read_valid_mask(i) == np.array([1, 1, 1], dtype=np.uint8)).all()
        g = b.read_gpm_sample(i)
        assert g.dtype == np.float32
        assert g.shape == GPM_SIZE
    b.close()


def test_build_cache_end_to_end(synthetic_workspace, tmp_path):
    pytest.importorskip("zarr")
    cache_root = tmp_path / "cache"
    df = pd.read_csv(synthetic_workspace["csv"])
    spec, index = build_cache_spec(df, cache_root, "ir_only")
    assert spec.n_total == len(df)
    assert sum(spec.per_sat_counts.values()) == len(df)
    # each satellite: 3 rows
    for s in SATELLITES:
        assert spec.per_sat_counts[s] == 3
        assert spec.per_sat_shapes[s][1] == len(band_indices_for(s, "ir_only"))

    backend_cls = get_backend("zarr")
    backend = backend_cls(spec, compressor="lz4")
    build_cache(
        csv_path=synthetic_workspace["csv"],
        data_root=synthetic_workspace["root"],
        cache_root=cache_root,
        backend=backend,
        band_mode="ir_only",
        load_gpm=True,
    )
    counts = verify_cache(backend)
    assert counts["gpm"] == len(df)
    for s in SATELLITES:
        assert counts[s] == 3
    # read a sample and check dtype/shape
    for s in SATELLITES:
        arr = backend.read_sat_sample(s, 0)
        assert arr.dtype == np.uint8
        assert arr.shape == (FRAMES_PER_SAMPLE, len(band_indices_for(s, "ir_only")), *NATIVE_SIZES[s])
    backend.close()


def test_build_cache_missing_frames(synthetic_workspace_missing_frames, tmp_path):
    pytest.importorskip("zarr")
    ws = synthetic_workspace_missing_frames
    cache_root = tmp_path / "cache_missing"
    df = pd.read_csv(ws["csv"])
    spec, _ = build_cache_spec(df, cache_root, "ir_only")
    backend_cls = get_backend("zarr")
    backend = backend_cls(spec, compressor="lz4")
    build_cache(
        csv_path=ws["csv"],
        data_root=ws["root"],
        cache_root=cache_root,
        backend=backend,
        band_mode="ir_only",
        load_gpm=True,
    )
    # row 0: 3 frames present -> mask (1,1,1)
    assert (backend.read_valid_mask(0) == np.array([1, 1, 1], dtype=np.uint8)).all()
    # row 3: 0 frames -> mask (0,0,0)
    assert (backend.read_valid_mask(3) == np.array([0, 0, 0], dtype=np.uint8)).all()
    backend.close()


def test_benchmark_runs(tmp_path):
    pytest.importorskip("zarr")
    result = run_benchmark(
        output_path=tmp_path / "backend.json",
        n_samples=32,
        channels=4,
        hw=16,
        seed=0,
    )
    assert result.recommended in ("zarr", "memmap")
    assert (tmp_path / "backend.json").exists()
