"""Zarr-based cache backend.

Layout::

    <root>/
        himawari.zarr/       zarr array, shape (N_him, T, C, 81, 81), uint8
        goes.zarr/           zarr array, shape (N_goes, T, C, 141, 141), uint8
        meteosat.zarr/       zarr array, shape (N_met, T, C, 144, 144), uint8
        gpm.zarr/            zarr array, shape (N_total, 41, 41), float32
        valid_mask.zarr/     zarr array, shape (N_total, T), uint8

Zarr chunks are sized so a single chunk holds a small run of samples along
axis 0 and one full sample along all other axes; this is optimal for random
per-sample reads by DataLoader workers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import zarr  # type: ignore
    _HAS_ZARR = True
except Exception as _e:  # pragma: no cover
    zarr = None  # type: ignore
    _HAS_ZARR = False
    _ZARR_IMPORT_ERROR = _e

try:
    from numcodecs import Blosc  # type: ignore
    _HAS_BLOSC = True
except Exception:  # pragma: no cover
    Blosc = None  # type: ignore
    _HAS_BLOSC = False

from ...constants import FRAMES_PER_SAMPLE, GPM_SIZE, SATELLITES
from .base import CacheBackend, CacheSpec


class ZarrBackend(CacheBackend):
    name = "zarr"

    def __init__(
        self,
        spec: CacheSpec,
        compressor: str = "lz4",
        chunk_size: int = 16,
        clevel: int = 3,
    ):
        super().__init__(spec)
        self._compressor_name = compressor
        self._chunk_size = int(chunk_size)
        self._clevel = int(clevel)
        self._sat_arrays: dict = {}
        self._gpm_array = None
        self._valid_mask_array = None
        self._require_zarr()

    @staticmethod
    def _require_zarr() -> None:
        if not _HAS_ZARR:
            raise ImportError(
                "zarr is required for ZarrBackend. Original import error: "
                f"{_ZARR_IMPORT_ERROR!r}"
            )

    def _make_compressor(self):
        if self._compressor_name == "none":
            return None
        if not _HAS_BLOSC:
            return None
        cname_map = {"lz4": "lz4", "zstd": "zstd", "blosclz": "blosclz"}
        cname = cname_map.get(self._compressor_name, "lz4")
        try:
            return Blosc(cname=cname, clevel=self._clevel, shuffle=Blosc.SHUFFLE)
        except Exception:
            return None

    def _open_new(self, path, shape, chunks, dtype, compressor):
        """Open a new zarr array using zarr_format=2 for compat with numcodecs
        Blosc (v3-native codecs are optional in this project).
        """
        return zarr.open(
            str(path), mode="w",
            shape=shape, chunks=chunks, dtype=dtype,
            compressor=compressor, zarr_format=2,
        )

    # -- create ---------------------------------------------------------
    def create(self) -> None:
        self.spec.root.mkdir(parents=True, exist_ok=True)
        compressor = self._make_compressor()
        for sat in SATELLITES:
            n = self.spec.per_sat_counts[sat]
            t, c, h, w = self.spec.per_sat_shapes[sat]
            path = self.spec.root / f"{sat}.zarr"
            chunk_n = min(self._chunk_size, max(n, 1))
            self._sat_arrays[sat] = self._open_new(
                path, (n, t, c, h, w), (chunk_n, t, c, h, w), "u1", compressor,
            )
        gpm_path = self.spec.root / "gpm.zarr"
        self._gpm_array = self._open_new(
            gpm_path,
            (self.spec.n_total, GPM_SIZE[0], GPM_SIZE[1]),
            (min(64, max(self.spec.n_total, 1)), GPM_SIZE[0], GPM_SIZE[1]),
            "f4", compressor,
        )
        vm_path = self.spec.root / "valid_mask.zarr"
        self._valid_mask_array = self._open_new(
            vm_path,
            (self.spec.n_total, FRAMES_PER_SAMPLE),
            (min(1024, max(self.spec.n_total, 1)), FRAMES_PER_SAMPLE),
            "u1", compressor,
        )

    def _open_readonly(self) -> None:
        for sat in SATELLITES:
            if sat not in self._sat_arrays:
                path = self.spec.root / f"{sat}.zarr"
                self._sat_arrays[sat] = zarr.open(str(path), mode="r")
        if self._gpm_array is None:
            self._gpm_array = zarr.open(str(self.spec.root / "gpm.zarr"), mode="r")
        if self._valid_mask_array is None:
            self._valid_mask_array = zarr.open(
                str(self.spec.root / "valid_mask.zarr"), mode="r"
            )

    def _open_append(self) -> None:
        for sat in SATELLITES:
            if sat not in self._sat_arrays:
                path = self.spec.root / f"{sat}.zarr"
                self._sat_arrays[sat] = zarr.open(str(path), mode="a")
        if self._gpm_array is None:
            self._gpm_array = zarr.open(str(self.spec.root / "gpm.zarr"), mode="a")
        if self._valid_mask_array is None:
            self._valid_mask_array = zarr.open(
                str(self.spec.root / "valid_mask.zarr"), mode="a"
            )

    # -- write ----------------------------------------------------------
    def write_sat_sample(self, satellite: str, local_idx: int, data: np.ndarray) -> None:
        if satellite not in self._sat_arrays:
            self._open_append()
        arr = self._sat_arrays[satellite]
        if data.shape != arr.shape[1:]:
            raise ValueError(
                f"{satellite}: expected sample shape {arr.shape[1:]}, got {data.shape}"
            )
        if data.dtype != np.uint8:
            data = data.astype(np.uint8, copy=False)
        arr[local_idx] = data

    def write_gpm_sample(self, global_idx: int, data: np.ndarray) -> None:
        if self._gpm_array is None:
            self._open_append()
        if data.shape != GPM_SIZE:
            raise ValueError(f"gpm sample shape must be {GPM_SIZE}, got {data.shape}")
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        self._gpm_array[global_idx] = data

    def write_valid_mask(self, global_idx: int, mask: np.ndarray) -> None:
        if self._valid_mask_array is None:
            self._open_append()
        if mask.shape != (FRAMES_PER_SAMPLE,):
            raise ValueError(f"valid mask shape must be ({FRAMES_PER_SAMPLE},), got {mask.shape}")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8, copy=False)
        self._valid_mask_array[global_idx] = mask

    # -- read -----------------------------------------------------------
    def read_sat_sample(self, satellite: str, local_idx: int) -> np.ndarray:
        if satellite not in self._sat_arrays:
            self._open_readonly()
        arr = self._sat_arrays[satellite]
        out = arr[local_idx]
        return np.asarray(out, dtype=np.uint8)

    def read_gpm_sample(self, global_idx: int) -> np.ndarray:
        if self._gpm_array is None:
            self._open_readonly()
        return np.asarray(self._gpm_array[global_idx], dtype=np.float32)

    def read_valid_mask(self, global_idx: int) -> np.ndarray:
        if self._valid_mask_array is None:
            self._open_readonly()
        return np.asarray(self._valid_mask_array[global_idx], dtype=np.uint8)

    def close(self) -> None:
        self._sat_arrays.clear()
        self._gpm_array = None
        self._valid_mask_array = None
