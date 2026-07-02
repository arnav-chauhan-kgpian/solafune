"""Shared pytest fixtures.

To avoid test coupling to the real 40k-sample dataset, most tests build a
small synthetic on-disk dataset in a temporary directory that mirrors the
Solafune workspace layout.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.constants import (  # noqa: E402
    FRAMES_PER_SAMPLE,
    GPM_SIZE,
    GPM_SUBDIR,
    NATIVE_SIZES,
    NUM_BANDS_TOTAL,
    SATELLITES,
    SAT_SUBDIRS,
)


def _rasterio_available() -> bool:
    try:
        import rasterio  # noqa: F401
        return True
    except Exception:
        return False


HAVE_RASTERIO = _rasterio_available()


def _write_sat_tif(path: Path, satellite: str, seed: int) -> None:
    import rasterio
    from rasterio.transform import Affine
    h, w = NATIVE_SIZES[satellite]
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 255, size=(NUM_BANDS_TOTAL, h, w), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": h, "width": w,
        "count": NUM_BANDS_TOTAL,
        "dtype": "uint8",
        "transform": Affine.identity(),
    }
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(data)


def _write_gpm_tif(path: Path, seed: int) -> None:
    import rasterio
    from rasterio.transform import Affine
    rng = np.random.default_rng(seed)
    # 80% zeros, some rain values
    values = rng.random(size=GPM_SIZE, dtype=np.float32)
    mask = values > 0.8
    out = np.zeros(GPM_SIZE, dtype=np.float32)
    out[mask] = values[mask] * 20.0  # up to ~20 mm/h
    profile = {
        "driver": "GTiff",
        "height": GPM_SIZE[0], "width": GPM_SIZE[1],
        "count": 1,
        "dtype": "float32",
        "transform": Affine.identity(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(path), "w", **profile) as dst:
        dst.write(out, 1)


@pytest.fixture(scope="session")
def synthetic_workspace(tmp_path_factory):
    """Build a synthetic mini-workspace with 3 samples per satellite = 9 rows."""
    if not HAVE_RASTERIO:
        pytest.skip("rasterio not available")

    root = tmp_path_factory.mktemp("workspace")
    csv_rows: List[Dict] = []
    seed = 0
    for sat_idx, sat in enumerate(SATELLITES):
        for i in range(3):
            location = f"loc_{sat}_{i}"
            frame_names: List[str] = []
            for t in range(FRAMES_PER_SAMPLE):
                fname = f"synth_{sat}_{i}_{t}.tif"
                _write_sat_tif(root / SAT_SUBDIRS[sat] / fname, sat, seed)
                frame_names.append(fname)
                seed += 1
            gpm_name = f"synth_{sat}_{i}_gpm.tif"
            _write_gpm_tif(root / GPM_SUBDIR / gpm_name, seed)
            seed += 1
            csv_rows.append({
                "unique_id": f"{sat_idx:02d}-{i:04d}",
                "name_location": location,
                "satellite_target": sat,
                "datetime": f"2024-01-0{i+1} 12:00:00",
                "last_30_minutes_observation_filename": str(frame_names),
                "gpm_imerg_filename": gpm_name,
            })
    df = pd.DataFrame(csv_rows)
    csv_path = root / "train_dataset.csv"
    df.to_csv(csv_path, index=False)
    return {"root": root, "csv": csv_path, "n_rows": len(df)}


@pytest.fixture(scope="session")
def synthetic_workspace_missing_frames(tmp_path_factory):
    """A tiny workspace where some rows have <3 frames to test padding."""
    if not HAVE_RASTERIO:
        pytest.skip("rasterio not available")

    root = tmp_path_factory.mktemp("workspace_missing")
    rows: List[Dict] = []
    seed = 10000
    sat = "himawari"
    for i, n_frames in enumerate([3, 2, 1, 0]):
        frame_names: List[str] = []
        for t in range(n_frames):
            fname = f"m_{sat}_{i}_{t}.tif"
            _write_sat_tif(root / SAT_SUBDIRS[sat] / fname, sat, seed)
            frame_names.append(fname)
            seed += 1
        gpm_name = f"m_{sat}_{i}_gpm.tif"
        _write_gpm_tif(root / GPM_SUBDIR / gpm_name, seed)
        seed += 1
        rows.append({
            "unique_id": f"m-{i:04d}",
            "name_location": f"loc_m_{i}",
            "satellite_target": sat,
            "datetime": f"2024-02-0{i+1} 06:00:00",
            "last_30_minutes_observation_filename": str(frame_names),
            "gpm_imerg_filename": gpm_name,
        })
    df = pd.DataFrame(rows)
    csv_path = root / "train_dataset.csv"
    df.to_csv(csv_path, index=False)
    return {"root": root, "csv": csv_path, "n_rows": len(df)}
