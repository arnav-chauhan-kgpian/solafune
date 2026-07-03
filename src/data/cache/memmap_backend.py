"""NumPy memmap cache backend.

Layout::

    <root>/
        himawari.bin       raw memmap, shape (N_him, T, C, 81, 81), uint8
        goes.bin           raw memmap, shape (N_goes, T, C, 141, 141), uint8
        meteosat.bin       raw memmap, shape (N_met, T, C, 144, 144), uint8
        gpm.bin            raw memmap, shape (N_total, 41, 41), float32
        valid_mask.bin     raw memmap, shape (N_total, T), uint8
        shapes.json        JSON metadata describing all shapes/dtypes

Memmap has the lowest per-sample latency but uses uncompressed storage.
Use only when disk is not a constraint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ...constants import FRAMES_PER_SAMPLE, GPM_SIZE, SATELLITES
from ...utils import read_json, write_json
from .base import CacheBackend, CacheSpec


class MemmapBackend(CacheBackend):
    name = "memmap"

    def __init__(self, spec: CacheSpec):
        super().__init__(spec)
        self._sat_arrays: Dict[str, np.memmap] = {}
        self._gpm_array: Optional[np.memmap] = None
        self._valid_mask_array: Optional[np.memmap] = None
        self._meta_path = self.spec.root / "shapes.json"

    # -- create ---------------------------------------------------------
    def create(self) -> None:
        self.spec.root.mkdir(parents=True, exist_ok=True)
        meta = {"per_sat_shapes": {}, "gpm_shape": list(GPM_SIZE), "n_total": self.spec.n_total}
        for sat in SATELLITES:
            n = self.spec.per_sat_counts[sat]
            t, c, h, w = self.spec.per_sat_shapes[sat]
            shape = (max(n, 1), t, c, h, w)
            path = self.spec.root / f"{sat}.bin"
            arr = np.memmap(str(path), dtype=np.uint8, mode="w+", shape=shape)
            arr[:] = 0
            arr.flush()
            self._sat_arrays[sat] = arr
            meta["per_sat_shapes"][sat] = list(shape)
        gpm_path = self.spec.root / "gpm.bin"
        gpm_shape = (max(self.spec.n_total, 1), GPM_SIZE[0], GPM_SIZE[1])
        gpm = np.memmap(str(gpm_path), dtype=np.float32, mode="w+", shape=gpm_shape)
        gpm[:] = 0.0
        gpm.flush()
        self._gpm_array = gpm
        vm_path = self.spec.root / "valid_mask.bin"
        vm_shape = (max(self.spec.n_total, 1), FRAMES_PER_SAMPLE)
        vm = np.memmap(str(vm_path), dtype=np.uint8, mode="w+", shape=vm_shape)
        vm[:] = 0
        vm.flush()
        self._valid_mask_array = vm
        write_json(self._meta_path, meta)

    def _open(self, mode: str) -> None:
        if not self._meta_path.exists():
            raise FileNotFoundError(f"cache metadata not found: {self._meta_path}")
        meta = read_json(self._meta_path)
        for sat in SATELLITES:
            path = self.spec.root / f"{sat}.bin"
            shape = tuple(int(x) for x in meta["per_sat_shapes"][sat])
            self._sat_arrays[sat] = np.memmap(str(path), dtype=np.uint8, mode=mode, shape=shape)
        gpm_shape = (int(meta["n_total"]) or 1, GPM_SIZE[0], GPM_SIZE[1])
        self._gpm_array = np.memmap(
            str(self.spec.root / "gpm.bin"), dtype=np.float32, mode=mode, shape=gpm_shape
        )
        self._valid_mask_array = np.memmap(
            str(self.spec.root / "valid_mask.bin"),
            dtype=np.uint8, mode=mode, shape=(gpm_shape[0], FRAMES_PER_SAMPLE),
        )

    def _ensure_read(self) -> None:
        if not self._sat_arrays:
            self._open(mode="r")

    def _ensure_write(self) -> None:
        if not self._sat_arrays:
            self._open(mode="r+")

    # -- write ----------------------------------------------------------
    def write_sat_sample(self, satellite: str, local_idx: int, data: np.ndarray) -> None:
        self._ensure_write()
        arr = self._sat_arrays[satellite]
        if data.shape != arr.shape[1:]:
            raise ValueError(
                f"{satellite}: expected sample shape {arr.shape[1:]}, got {data.shape}"
            )
        if data.dtype != np.uint8:
            data = data.astype(np.uint8, copy=False)
        arr[local_idx] = data

    def write_gpm_sample(self, global_idx: int, data: np.ndarray) -> None:
        self._ensure_write()
        assert self._gpm_array is not None
        if data.shape != GPM_SIZE:
            raise ValueError(f"gpm sample shape must be {GPM_SIZE}, got {data.shape}")
        if data.dtype != np.float32:
            data = data.astype(np.float32, copy=False)
        self._gpm_array[global_idx] = data

    def write_valid_mask(self, global_idx: int, mask: np.ndarray) -> None:
        self._ensure_write()
        assert self._valid_mask_array is not None
        if mask.shape != (FRAMES_PER_SAMPLE,):
            raise ValueError(f"valid mask shape must be ({FRAMES_PER_SAMPLE},), got {mask.shape}")
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8, copy=False)
        self._valid_mask_array[global_idx] = mask

    def flush(self) -> None:
        for a in self._sat_arrays.values():
            a.flush()
        if self._gpm_array is not None:
            self._gpm_array.flush()
        if self._valid_mask_array is not None:
            self._valid_mask_array.flush()

    # -- read -----------------------------------------------------------
    def read_sat_sample(self, satellite: str, local_idx: int) -> np.ndarray:
        self._ensure_read()
        arr = self._sat_arrays[satellite]
        return np.asarray(arr[local_idx], dtype=np.uint8)

    def read_gpm_sample(self, global_idx: int) -> np.ndarray:
        self._ensure_read()
        assert self._gpm_array is not None
        return np.asarray(self._gpm_array[global_idx], dtype=np.float32)

    def read_valid_mask(self, global_idx: int) -> np.ndarray:
        self._ensure_read()
        assert self._valid_mask_array is not None
        return np.asarray(self._valid_mask_array[global_idx], dtype=np.uint8)

    def close(self) -> None:
        """Release memmap handles.

        On Windows, deleting the numpy.memmap reference is not sufficient to
        release the underlying file handle — the private `_mmap` must be
        explicitly closed. Failing to do so blocks `os.remove` / `rmtree` on
        the cache directory and slowly leaks handles across DataLoader
        worker restarts.
        """
        for a in self._sat_arrays.values():
            self._close_one(a)
        self._sat_arrays.clear()
        for name in ("_gpm_array", "_valid_mask_array"):
            arr = getattr(self, name, None)
            if arr is not None:
                self._close_one(arr)
                setattr(self, name, None)

    @staticmethod
    def _close_one(arr) -> None:
        try:
            arr.flush()
        except Exception:
            pass
        try:
            mm = getattr(arr, "_mmap", None)
            if mm is not None:
                mm.close()
        except Exception:
            pass
