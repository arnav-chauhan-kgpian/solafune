"""Tests for src/utils/io.py."""
from __future__ import annotations

import numpy as np
import pytest

from src.constants import GPM_SIZE, NATIVE_SIZES, NUM_BANDS_TOTAL
from src.paths import gpm_tif_path, sat_tif_path
from src.utils.io import (
    TIFReadError,
    read_gpm_tif,
    read_satellite_tif,
    validate_tif,
    write_gpm_tif,
)


def test_read_satellite_tif_shape_and_dtype(synthetic_workspace):
    root = synthetic_workspace["root"]
    import pandas as pd
    from src.utils import parse_frame_list
    df = pd.read_csv(synthetic_workspace["csv"])
    row = df.iloc[0]
    frames = parse_frame_list(row["last_30_minutes_observation_filename"])
    path = sat_tif_path(root, row["satellite_target"], frames[0])
    arr, meta = read_satellite_tif(path)
    assert arr.dtype == np.uint8
    assert arr.shape[0] == NUM_BANDS_TOTAL
    assert arr.shape[1:] == NATIVE_SIZES[row["satellite_target"]]
    assert meta.count == NUM_BANDS_TOTAL


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(TIFReadError):
        read_satellite_tif(tmp_path / "does_not_exist.tif")


def test_read_gpm_tif_shape_dtype(synthetic_workspace):
    root = synthetic_workspace["root"]
    import pandas as pd
    df = pd.read_csv(synthetic_workspace["csv"])
    row = df.iloc[0]
    path = gpm_tif_path(root, row["gpm_imerg_filename"])
    arr, meta = read_gpm_tif(path)
    assert arr.dtype == np.float32
    assert arr.shape == GPM_SIZE
    assert np.isfinite(arr).all()
    assert meta.count == 1


def test_write_and_readback_gpm_tif(tmp_path):
    arr = np.random.RandomState(0).rand(*GPM_SIZE).astype(np.float32) * 5.0
    out = tmp_path / "pred.tif"
    write_gpm_tif(out, arr)
    reread, _ = read_gpm_tif(out)
    np.testing.assert_allclose(reread, arr, atol=1e-4)


def test_validate_tif_size_mismatch(synthetic_workspace):
    root = synthetic_workspace["root"]
    import pandas as pd
    from src.utils import parse_frame_list
    df = pd.read_csv(synthetic_workspace["csv"])
    row = df[df["satellite_target"] == "himawari"].iloc[0]
    frames = parse_frame_list(row["last_30_minutes_observation_filename"])
    path = sat_tif_path(root, "himawari", frames[0])
    validate_tif(path, satellite="himawari")   # ok
    with pytest.raises(TIFReadError):
        validate_tif(path, satellite="goes")
