"""Abstract cache backend contract.

A `CacheBackend` stores per-satellite native-resolution uint8 tensors of shape
    (N_satellite, T=3, C_cached, H_native, W_native)

and a single float32 GPM array of shape
    (N_total, 41, 41)

plus a `valid_mask` uint8 array of shape (N_total, T) indicating which frames
are present (1) vs. zero-padded (0). The backend is responsible for
allocation, writing, reading and integrity verification. It is NOT responsible
for band selection, resizing or normalization — those happen in the Dataset.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ...constants import FRAMES_PER_SAMPLE, GPM_SIZE, SATELLITES


@dataclass(frozen=True)
class CacheSpec:
    """Immutable description of the storage layout of a cache."""

    root: Path
    n_total: int
    per_sat_counts: Mapping[str, int]  # {"himawari": N_him, "goes": N_g, "meteosat": N_m}
    per_sat_shapes: Mapping[str, Tuple[int, int, int, int]]  # (T, C, H, W) per sat
    dtype: str = "uint8"

    def validate(self) -> None:
        if not self.root.exists():
            raise FileNotFoundError(f"cache root does not exist: {self.root}")
        for sat in SATELLITES:
            if sat not in self.per_sat_counts:
                raise ValueError(f"missing satellite in spec: {sat}")
            if sat not in self.per_sat_shapes:
                raise ValueError(f"missing shape in spec: {sat}")
            t, c, h, w = self.per_sat_shapes[sat]
            if t != FRAMES_PER_SAMPLE:
                raise ValueError(f"{sat}: expected T={FRAMES_PER_SAMPLE}, got {t}")
            if c <= 0 or h <= 0 or w <= 0:
                raise ValueError(f"{sat}: invalid shape {(t, c, h, w)}")

    def to_dict(self) -> Dict:
        return {
            "root": str(self.root),
            "n_total": self.n_total,
            "per_sat_counts": dict(self.per_sat_counts),
            "per_sat_shapes": {k: list(v) for k, v in self.per_sat_shapes.items()},
            "dtype": self.dtype,
        }

    @staticmethod
    def from_dict(d: Dict) -> "CacheSpec":
        return CacheSpec(
            root=Path(d["root"]),
            n_total=int(d["n_total"]),
            per_sat_counts={k: int(v) for k, v in d["per_sat_counts"].items()},
            per_sat_shapes={
                k: tuple(int(x) for x in v) for k, v in d["per_sat_shapes"].items()
            },
            dtype=str(d.get("dtype", "uint8")),
        )


class CacheBackend(ABC):
    """Abstract cache backend."""

    name: str = "abstract"

    def __init__(self, spec: CacheSpec):
        self.spec = spec

    # -- create / write --------------------------------------------------
    @abstractmethod
    def create(self) -> None:
        """Allocate the empty on-disk arrays according to `self.spec`."""

    @abstractmethod
    def write_sat_sample(
        self,
        satellite: str,
        local_idx: int,
        data: np.ndarray,  # shape (T, C, H, W) uint8
    ) -> None:
        """Write a single satellite sample at position `local_idx`."""

    @abstractmethod
    def write_gpm_sample(self, global_idx: int, data: np.ndarray) -> None:
        """Write a single GPM sample at position `global_idx`."""

    @abstractmethod
    def write_valid_mask(self, global_idx: int, mask: np.ndarray) -> None:
        """Write the per-frame validity mask for a sample. `mask` shape: (T,) uint8."""

    # -- read ------------------------------------------------------------
    @abstractmethod
    def read_sat_sample(self, satellite: str, local_idx: int) -> np.ndarray:
        """Return a (T, C, H, W) uint8 array."""

    @abstractmethod
    def read_gpm_sample(self, global_idx: int) -> np.ndarray:
        """Return a (H, W) float32 array. May be zeros for eval placeholder samples."""

    @abstractmethod
    def read_valid_mask(self, global_idx: int) -> np.ndarray:
        """Return a (T,) uint8 array."""

    # -- lifecycle -------------------------------------------------------
    @abstractmethod
    def close(self) -> None:
        """Release all open handles."""

    def flush(self) -> None:
        """Persist pending writes. Default: no-op."""

    def verify(self) -> Dict[str, int]:
        """Basic integrity check.

        Returns per-satellite sample counts observed on disk. Subclasses may
        override for stricter checks.
        """
        counts: Dict[str, int] = {}
        for sat in SATELLITES:
            n = self.spec.per_sat_counts[sat]
            if n == 0:
                counts[sat] = 0
                continue
            # touch a sample at each end
            _ = self.read_sat_sample(sat, 0)
            _ = self.read_sat_sample(sat, n - 1)
            counts[sat] = n
        counts["gpm"] = self.spec.n_total
        return counts

    # -- context management ---------------------------------------------
    def __enter__(self) -> "CacheBackend":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def gpm_shape() -> Tuple[int, int]:
    return GPM_SIZE
