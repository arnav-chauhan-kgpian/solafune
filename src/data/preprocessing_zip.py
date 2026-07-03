"""Cache builder that reads TIFs directly from a ZIP archive (streaming).

Used on Kaggle where extracting the raw dataset (~31 GB) would blow the 20 GB
`/kaggle/working` quota. The two dataset ZIPs live on the read-only
`/kaggle/input` mount (unlimited), and we stream each TIF's bytes through
`rasterio.MemoryFile` — never materialising the extracted tree.

Mirrors the file-based `preprocessing.build_cache` API and produces an
identical cache layout, so the Dataset code is unchanged.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from ..utils import parse_frame_list, write_json
from ..utils.io import (
    read_gpm_tif_from_bytes,
    read_satellite_tif_from_bytes,
)
from .cache.base import CacheBackend
from .preprocessing import build_cache_spec  # reuse spec builder

log = get_logger(__name__)


def _build_zip_index(zf: zipfile.ZipFile) -> Tuple[str, Dict[str, str]]:
    """Return (root_prefix, filename -> full_zip_entry_name) index.

    Our zips were created with the dataset root directory included at the
    top, e.g. ``train_dataset_.../himawari/xxx.tif``. This function detects
    that root prefix and builds a filename-only lookup so callers can pass
    the bare TIF filenames from the CSV.
    """
    entries: Dict[str, str] = {}
    root_prefix: Optional[str] = None
    for name in zf.namelist():
        # names use forward slashes on all platforms
        parts = name.split("/")
        if len(parts) < 2:
            continue
        if not parts[-1]:  # directory entry
            continue
        if root_prefix is None:
            root_prefix = parts[0]
        # index by bare filename
        entries[parts[-1]] = name
    return (root_prefix or ""), entries


def build_cache_from_zips(
    csv_path: Path,
    sat_zip_path: Path,
    gpm_zip_path: Optional[Path],
    cache_root: Path,
    backend: CacheBackend,
    band_mode: str,
    load_gpm: bool = True,
    verbose_every: int = 1000,
    limit: Optional[int] = None,
) -> None:
    """Stream a cache directly from one or two ZIP archives.

    Args:
        csv_path: train_dataset.csv or evaluation_target.csv.
        sat_zip_path: ZIP containing the satellite TIFs (and, for train,
            usually the GPM TIFs too, all under one root dir).
        gpm_zip_path: Optional separate ZIP containing GPM TIFs. If None,
            GPM TIFs are looked up inside `sat_zip_path`.
        cache_root: where the cache backend stores its arrays.
        backend: an already-instantiated CacheBackend (unopened).
        band_mode: "ir_only" | "all" | "visible_only".
        load_gpm: if False, GPM entries are zeroed (eval placeholders).
    """
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(int(limit)).reset_index(drop=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing columns: {missing}")

    spec, index = build_cache_spec(df, cache_root, band_mode)
    # sanity match
    if backend.spec.n_total != spec.n_total:
        raise RuntimeError(
            f"backend n_total={backend.spec.n_total} != csv-derived {spec.n_total}"
        )
    for sat in SATELLITES:
        if backend.spec.per_sat_counts[sat] != spec.per_sat_counts[sat]:
            raise RuntimeError(f"backend {sat} count mismatch")
        if tuple(backend.spec.per_sat_shapes[sat]) != tuple(spec.per_sat_shapes[sat]):
            raise RuntimeError(f"backend {sat} shape mismatch")
    backend.create()

    band_idx_by_sat = {s: band_indices_for(s, band_mode) for s in SATELLITES}
    local_ptr = {s: 0 for s in SATELLITES}
    read_stats = {"missing": 0, "corrupt": 0, "gpm_missing": 0}

    zf_sat = zipfile.ZipFile(sat_zip_path, "r")
    zf_gpm = zipfile.ZipFile(gpm_zip_path, "r") if (gpm_zip_path and gpm_zip_path != sat_zip_path) else zf_sat
    try:
        _, sat_lookup = _build_zip_index(zf_sat)
        _, gpm_lookup = _build_zip_index(zf_gpm)
        log.info("sat zip entries: %d | gpm zip entries: %d",
                 len(sat_lookup), len(gpm_lookup))

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
                if not fname:
                    read_stats["missing"] += 1
                    continue
                zip_entry = sat_lookup.get(fname)
                if zip_entry is None:
                    read_stats["corrupt"] += 1
                    log.warning("sat entry not found in zip: %s", fname)
                    continue
                try:
                    with zf_sat.open(zip_entry) as f:
                        data = f.read()
                    arr, _ = read_satellite_tif_from_bytes(
                        data, label=fname, expected_size=NATIVE_SIZES[sat],
                    )
                except Exception as e:
                    read_stats["corrupt"] += 1
                    log.warning("sat %s failed: %r", fname, e)
                    continue
                arr = arr[list(band_idx_by_sat[sat]), :, :]
                frames[t] = arr
                valid[t] = 1

            backend.write_sat_sample(sat, local_ptr[sat], frames)
            backend.write_valid_mask(int(row_idx), valid)
            local_ptr[sat] += 1

            # GPM
            gpm_arr = np.zeros(GPM_SIZE, dtype=np.float32)
            if load_gpm:
                gpm_name = row[COL_GPM]
                if not (isinstance(gpm_name, float) and np.isnan(gpm_name)):
                    zip_entry = gpm_lookup.get(str(gpm_name))
                    if zip_entry is not None:
                        try:
                            with zf_gpm.open(zip_entry) as f:
                                data = f.read()
                            gpm_arr, _ = read_gpm_tif_from_bytes(data, label=str(gpm_name))
                        except Exception as e:
                            read_stats["gpm_missing"] += 1
                            log.warning("gpm %s failed: %r", gpm_name, e)
                    else:
                        read_stats["gpm_missing"] += 1
            backend.write_gpm_sample(int(row_idx), gpm_arr)

            if verbose_every > 0 and (int(row_idx) + 1) % verbose_every == 0:
                log.info("processed %d/%d samples", int(row_idx) + 1, len(df))

    finally:
        try:
            zf_sat.close()
        except Exception:
            pass
        try:
            if zf_gpm is not zf_sat:
                zf_gpm.close()
        except Exception:
            pass

    backend.flush()
    write_json(cache_root / "index.json", index.to_dict())
    write_json(cache_root / "spec.json", spec.to_dict())
    log.info("streamed cache built at %s (%d samples). read_stats=%s",
             cache_root, spec.n_total, read_stats)


def compute_norm_stats_from_zip(
    zip_path: Path,
    csv_path: Path,
    max_files_per_satellite: int = 300,
    pixel_stride: int = 2,
    seed: int = 0,
):
    """Compute per-satellite normalization stats by streaming TIFs from a ZIP."""
    from .normalization import BandStats, NormStats, _Welford  # local import
    from ..constants import NUM_BANDS_TOTAL

    df = pd.read_csv(csv_path)
    zf = zipfile.ZipFile(zip_path, "r")
    _, lookup = _build_zip_index(zf)
    rng = np.random.default_rng(seed)
    per_sat: Dict[str, "BandStats"] = {}
    try:
        for sat in SATELLITES:
            # collect candidate filenames for this satellite
            candidates: List[str] = []
            for _, row in df.iterrows():
                if str(row[COL_SATELLITE]).lower().strip() != sat:
                    continue
                for f in parse_frame_list(row[COL_FRAMES]):
                    if f in lookup:
                        candidates.append(f)
            if not candidates:
                log.warning("no files for %s in zip", sat)
                per_sat[sat] = BandStats(
                    mean=[0.0] * NUM_BANDS_TOTAL,
                    std=[1.0] * NUM_BANDS_TOTAL,
                    n_pixels=0,
                )
                continue
            n_use = min(len(candidates), max_files_per_satellite)
            sampled = rng.choice(candidates, size=n_use, replace=False)
            acc = _Welford(n_bands=NUM_BANDS_TOTAL)
            for fname in sampled:
                try:
                    with zf.open(lookup[fname]) as f:
                        data = f.read()
                    arr, _ = read_satellite_tif_from_bytes(
                        data, label=fname, expected_size=NATIVE_SIZES[sat],
                    )
                except Exception as e:
                    log.warning("skip %s: %r", fname, e)
                    continue
                if pixel_stride > 1:
                    arr = arr[:, ::pixel_stride, ::pixel_stride]
                c, h, w = arr.shape
                pixels = arr.reshape(c, h * w).T.astype(np.float64)
                acc.update_batch(pixels)
            mean, std, npx = acc.result()
            per_sat[sat] = BandStats(mean=mean, std=std, n_pixels=npx)
            log.info("%s: %d files, %d pixels, mean[b07]=%.2f std[b07]=%.2f",
                     sat, n_use, npx, mean[6], std[6])
    finally:
        zf.close()
    return NormStats(per_satellite=per_sat)
