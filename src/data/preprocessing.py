"""Cache builder.

Given a CSV (train or eval), preprocess all satellite + GPM TIFs into a
`CacheBackend` at native resolution. The cache is intentionally lightweight:
    * satellite images kept at native resolution (81/141/144 px)
    * band selection at cache-build time (ir_only vs all)
    * missing frames materialized as zero, with `valid_mask` tracking presence
    * GPM stored as float32 (identity for training; placeholder noise for eval)
    * a JSON index maps global CSV row → (satellite, local index)

Resize is NOT performed here; it is performed inside the Dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..constants import (
    COL_FRAMES,
    COL_GPM,
    COL_LOCATION,
    COL_SATELLITE,
    COL_UID,
    FRAMES_PER_SAMPLE,
    GPM_SIZE,
    NATIVE_SIZES,
    REQUIRED_COLUMNS,
    SATELLITES,
    band_indices_for,
)
from ..logger import get_logger
from ..paths import gpm_tif_path, sat_tif_path
from ..utils import parse_frame_list, write_json
from ..utils.io import read_gpm_tif, read_satellite_tif
from .cache.base import CacheBackend, CacheSpec

log = get_logger(__name__)


@dataclass
class CacheIndex:
    """Global-to-local index mapping."""
    global_ids: List[str]                  # unique_id per row
    locations: List[str]
    satellites: List[str]                  # per row
    global_to_local: List[int]             # position within per-satellite cache
    per_sat_global_indices: Dict[str, List[int]]  # inverse mapping

    def to_dict(self) -> Dict:
        return {
            "global_ids": self.global_ids,
            "locations": self.locations,
            "satellites": self.satellites,
            "global_to_local": self.global_to_local,
            "per_sat_global_indices": self.per_sat_global_indices,
        }


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {missing}")
    return df


def build_cache_spec(
    df: pd.DataFrame,
    cache_root: Path,
    band_mode: str,
) -> Tuple[CacheSpec, CacheIndex]:
    """Analyze the CSV and produce a CacheSpec + CacheIndex."""
    per_sat_counts: Dict[str, int] = {s: 0 for s in SATELLITES}
    per_sat_global: Dict[str, List[int]] = {s: [] for s in SATELLITES}
    global_to_local: List[int] = [-1] * len(df)
    satellites_col: List[str] = []
    locations_col: List[str] = []
    global_ids: List[str] = []

    for row_idx, row in df.iterrows():
        sat = str(row[COL_SATELLITE]).lower().strip()
        if sat not in SATELLITES:
            raise ValueError(f"row {row_idx}: unknown satellite {sat!r}")
        local = per_sat_counts[sat]
        per_sat_counts[sat] += 1
        per_sat_global[sat].append(int(row_idx))
        global_to_local[int(row_idx)] = local
        satellites_col.append(sat)
        locations_col.append(str(row[COL_LOCATION]))
        global_ids.append(str(row[COL_UID]))

    per_sat_shapes: Dict[str, Tuple[int, int, int, int]] = {}
    for sat in SATELLITES:
        c = len(band_indices_for(sat, band_mode))
        h, w = NATIVE_SIZES[sat]
        per_sat_shapes[sat] = (FRAMES_PER_SAMPLE, c, h, w)

    spec = CacheSpec(
        root=cache_root,
        n_total=int(len(df)),
        per_sat_counts=per_sat_counts,
        per_sat_shapes=per_sat_shapes,
        dtype="uint8",
    )
    index = CacheIndex(
        global_ids=global_ids,
        locations=locations_col,
        satellites=satellites_col,
        global_to_local=global_to_local,
        per_sat_global_indices=per_sat_global,
    )
    return spec, index


def _read_frame_or_zeros(
    root: Path,
    satellite: str,
    filename: Optional[str],
    band_idx: Tuple[int, ...],
    stats: Optional[Dict[str, int]] = None,
) -> Tuple[np.ndarray, bool]:
    """Return (C, H, W) uint8 array and a `valid` flag.

    Args:
        stats: optional counter dict; the keys "missing" and "corrupt" are
            incremented distinguishing missing filenames from unreadable
            files.
    """
    h, w = NATIVE_SIZES[satellite]
    c = len(band_idx)
    if not filename:
        if stats is not None:
            stats["missing"] = stats.get("missing", 0) + 1
        return np.zeros((c, h, w), dtype=np.uint8), False
    p = sat_tif_path(root, satellite, filename)
    try:
        arr, _ = read_satellite_tif(p, expected_size=(h, w))
    except Exception as e:
        if stats is not None:
            stats["corrupt"] = stats.get("corrupt", 0) + 1
        log.warning("frame %s failed to load: %r; using zeros", p, e)
        return np.zeros((c, h, w), dtype=np.uint8), False
    # arr shape: (16, H, W); select bands
    arr = arr[list(band_idx), :, :]
    return arr, True


def _read_gpm_or_zeros(root: Path, filename: Optional[str], required: bool) -> np.ndarray:
    if not filename:
        return np.zeros(GPM_SIZE, dtype=np.float32)
    p = gpm_tif_path(root, filename)
    try:
        arr, _ = read_gpm_tif(p)
        return arr
    except Exception as e:
        if required:
            log.warning("gpm %s failed to load: %r; using zeros", p, e)
        return np.zeros(GPM_SIZE, dtype=np.float32)


def build_cache(
    csv_path: Path,
    data_root: Path,
    cache_root: Path,
    backend: CacheBackend,
    band_mode: str,
    load_gpm: bool = True,
    verbose_every: int = 500,
    limit: Optional[int] = None,
) -> Tuple[CacheSpec, CacheIndex]:
    """Preprocess every sample in `csv_path` and write it to `backend`.

    Args:
        csv_path: train_dataset.csv or evaluation_target.csv.
        data_root: root directory containing the {satellite}/, gpm_imerg/ subdirs.
        cache_root: where the cache backend stores its arrays.
        backend: an already-instantiated CacheBackend (unopened).
        band_mode: "ir_only" | "all" | "visible_only".
        load_gpm: if True, read GPM TIFs (for train). If False, GPM is zeroed
            (for eval placeholders).
        limit: optional cap on rows processed (for debugging).
    """
    df = _load_csv(csv_path)
    if limit is not None:
        df = df.head(int(limit)).reset_index(drop=True)

    spec, index = build_cache_spec(df, cache_root, band_mode)
    # strict spec match — n_total, per-satellite counts, per-satellite shapes.
    if backend.spec.n_total != spec.n_total:
        raise RuntimeError(
            f"backend spec n_total={backend.spec.n_total} != csv-derived n_total={spec.n_total}"
        )
    for sat in SATELLITES:
        if backend.spec.per_sat_counts[sat] != spec.per_sat_counts[sat]:
            raise RuntimeError(
                f"backend spec {sat} count mismatch: "
                f"backend={backend.spec.per_sat_counts[sat]} csv={spec.per_sat_counts[sat]}"
            )
        if tuple(backend.spec.per_sat_shapes[sat]) != tuple(spec.per_sat_shapes[sat]):
            raise RuntimeError(
                f"backend spec {sat} shape mismatch: "
                f"backend={backend.spec.per_sat_shapes[sat]} csv={spec.per_sat_shapes[sat]}"
            )
    backend.create()

    band_idx_by_sat = {s: band_indices_for(s, band_mode) for s in SATELLITES}
    local_ptr = {s: 0 for s in SATELLITES}
    read_stats: Dict[str, int] = {"missing": 0, "corrupt": 0}

    for row_idx, row in df.iterrows():
        sat = str(row[COL_SATELLITE]).lower().strip()
        frame_names = parse_frame_list(row[COL_FRAMES])
        padded: List[Optional[str]] = list(frame_names[:FRAMES_PER_SAMPLE])
        while len(padded) < FRAMES_PER_SAMPLE:
            padded.append(None)

        frames = np.zeros(
            (FRAMES_PER_SAMPLE, len(band_idx_by_sat[sat]), *NATIVE_SIZES[sat]),
            dtype=np.uint8,
        )
        valid = np.zeros((FRAMES_PER_SAMPLE,), dtype=np.uint8)
        for t, fname in enumerate(padded):
            arr, is_valid = _read_frame_or_zeros(
                data_root, sat, fname, band_idx_by_sat[sat], stats=read_stats,
            )
            frames[t] = arr
            valid[t] = 1 if is_valid else 0

        local = local_ptr[sat]
        backend.write_sat_sample(sat, local, frames)
        backend.write_valid_mask(int(row_idx), valid)
        local_ptr[sat] += 1

        gpm_name = row[COL_GPM] if not (isinstance(row[COL_GPM], float) and np.isnan(row[COL_GPM])) else None
        gpm = _read_gpm_or_zeros(data_root, gpm_name if load_gpm else None, required=load_gpm)
        backend.write_gpm_sample(int(row_idx), gpm)

        if verbose_every > 0 and (int(row_idx) + 1) % verbose_every == 0:
            log.info("processed %d/%d samples", int(row_idx) + 1, len(df))

    backend.flush()
    log.info(
        "cache read stats: missing_frame_entries=%d corrupt_frame_files=%d",
        read_stats["missing"], read_stats["corrupt"],
    )

    # persist index
    write_json(cache_root / "index.json", index.to_dict())
    write_json(cache_root / "spec.json", spec.to_dict())
    log.info("cache built at %s (%d samples)", cache_root, spec.n_total)
    return spec, index


def verify_cache(backend: CacheBackend) -> Dict[str, int]:
    """Run the backend's verify() and log a summary."""
    counts = backend.verify()
    log.info("cache verify counts: %s", counts)
    return counts
