"""Rasterio-based TIF reader/writer with defensive error handling.

Satellite TIFs contract:
    * 16 bands, uint8
    * shapes: Himawari 81x81, GOES 141x141, Meteosat 144x144
    * identity CRS transform (no georeferencing)

GPM TIFs contract:
    * 1 band, float32
    * 41x41
    * mm/h units, values in [0, ~55]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

try:
    import rasterio  # type: ignore
    from rasterio.errors import RasterioIOError  # type: ignore
    from rasterio.transform import Affine  # type: ignore
    _HAS_RASTERIO = True
except Exception as _e:  # pragma: no cover
    rasterio = None  # type: ignore
    RasterioIOError = Exception  # type: ignore
    Affine = None  # type: ignore
    _HAS_RASTERIO = False
    _RASTERIO_IMPORT_ERROR = _e

from ..constants import (
    GPM_RAW_DTYPE,
    GPM_SIZE,
    NATIVE_SIZES,
    NUM_BANDS_TOTAL,
    SAT_RAW_DTYPE,
)

PathLike = Union[str, Path]


class TIFReadError(RuntimeError):
    """Raised when a TIF file cannot be read or fails a contract check."""


@dataclass(frozen=True)
class TIFMetadata:
    """Metadata associated with a TIF file we care about."""

    path: str
    height: int
    width: int
    count: int
    dtype: str
    crs: Optional[str]
    transform: Optional[Tuple[float, float, float, float, float, float]]


def _require_rasterio() -> None:
    if not _HAS_RASTERIO:
        raise ImportError(
            "rasterio is required for TIF I/O but is not installed. "
            f"Original import error: {_RASTERIO_IMPORT_ERROR!r}"
        )


def _open(path: PathLike):
    _require_rasterio()
    p = Path(path)
    if not p.exists():
        raise TIFReadError(f"File does not exist: {p}")
    try:
        return rasterio.open(str(p))
    except RasterioIOError as e:
        raise TIFReadError(f"rasterio failed to open {p}: {e}") from e


def _extract_metadata(src, path: PathLike) -> TIFMetadata:
    tr = src.transform
    transform_tuple: Optional[Tuple[float, float, float, float, float, float]]
    transform_tuple = (tr.a, tr.b, tr.c, tr.d, tr.e, tr.f) if tr is not None else None
    return TIFMetadata(
        path=str(path),
        height=int(src.height),
        width=int(src.width),
        count=int(src.count),
        dtype=str(src.dtypes[0]) if src.count > 0 else "unknown",
        crs=str(src.crs) if src.crs else None,
        transform=transform_tuple,
    )


def read_satellite_tif(
    path: PathLike,
    expected_size: Optional[Tuple[int, int]] = None,
    expected_bands: int = NUM_BANDS_TOTAL,
    validate_dtype: bool = True,
) -> Tuple[np.ndarray, TIFMetadata]:
    """Read a satellite TIF.

    Args:
        path: Path to the TIF file.
        expected_size: Optional (H, W) shape to validate.
        expected_bands: Expected number of bands (default 16).
        validate_dtype: If True, raise unless the file is uint8.

    Returns:
        (array, metadata) where array has shape (C, H, W) as uint8.

    Raises:
        TIFReadError: on any contract failure.
    """
    with _open(path) as src:
        meta = _extract_metadata(src, path)
        if meta.count != expected_bands:
            raise TIFReadError(
                f"{path}: expected {expected_bands} bands, got {meta.count}"
            )
        if validate_dtype and meta.dtype != SAT_RAW_DTYPE:
            raise TIFReadError(
                f"{path}: expected dtype {SAT_RAW_DTYPE}, got {meta.dtype}"
            )
        if expected_size is not None and (meta.height, meta.width) != expected_size:
            raise TIFReadError(
                f"{path}: expected size {expected_size}, "
                f"got ({meta.height}, {meta.width})"
            )
        try:
            arr = src.read()
        except RasterioIOError as e:
            raise TIFReadError(f"{path}: read failed: {e}") from e
        if arr.ndim != 3:
            raise TIFReadError(
                f"{path}: expected 3D array (C, H, W), got shape {arr.shape}"
            )
        if arr.dtype != np.uint8 and validate_dtype:
            arr = arr.astype(np.uint8, copy=False)
        return arr, meta


def read_gpm_tif(
    path: PathLike,
    expected_size: Tuple[int, int] = GPM_SIZE,
) -> Tuple[np.ndarray, TIFMetadata]:
    """Read a GPM IMERG TIF.

    Returns:
        (array, metadata) where array has shape (H, W) as float32.
    """
    with _open(path) as src:
        meta = _extract_metadata(src, path)
        if meta.count != 1:
            raise TIFReadError(
                f"{path}: expected 1 band GPM tif, got {meta.count}"
            )
        if (meta.height, meta.width) != expected_size:
            raise TIFReadError(
                f"{path}: expected {expected_size}, got ({meta.height}, {meta.width})"
            )
        try:
            arr = src.read(1)
        except RasterioIOError as e:
            raise TIFReadError(f"{path}: read failed: {e}") from e
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        if not np.isfinite(arr).all():
            raise TIFReadError(f"{path}: contains non-finite values")
        return arr, meta


def write_gpm_tif(
    path: PathLike,
    data: np.ndarray,
    reference_metadata: Optional[TIFMetadata] = None,
) -> None:
    """Write a float32 41x41 GPM prediction TIF.

    The file uses the same identity transform as the placeholder submission
    TIFs. If `reference_metadata` is provided, its transform is reused.
    """
    _require_rasterio()
    if data.ndim != 2:
        raise ValueError(f"data must be 2D, got shape {data.shape}")
    if data.shape != GPM_SIZE:
        raise ValueError(f"data must be {GPM_SIZE}, got {data.shape}")
    if data.dtype != np.float32:
        data = data.astype(np.float32, copy=False)
    if not np.isfinite(data).all():
        raise ValueError("data contains non-finite values")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if reference_metadata is not None and reference_metadata.transform is not None:
        a, b, c, d, e, f = reference_metadata.transform
        transform = Affine(a, b, c, d, e, f)
    else:
        transform = Affine.identity()

    profile = {
        "driver": "GTiff",
        "height": int(GPM_SIZE[0]),
        "width": int(GPM_SIZE[1]),
        "count": 1,
        "dtype": GPM_RAW_DTYPE,
        "transform": transform,
        "compress": "lzw",
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    with rasterio.open(str(tmp), "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(p)


def validate_tif(path: PathLike, satellite: Optional[str] = None) -> TIFMetadata:
    """Lightly validate a TIF file without loading its pixel data.

    If `satellite` is provided, validates the expected native size for that
    sensor. Returns the parsed metadata.
    """
    with _open(path) as src:
        meta = _extract_metadata(src, path)
    if satellite is not None:
        expected = NATIVE_SIZES.get(satellite)
        if expected is None:
            raise ValueError(f"Unknown satellite: {satellite!r}")
        if (meta.height, meta.width) != expected:
            raise TIFReadError(
                f"{path}: expected {expected} for {satellite}, "
                f"got ({meta.height}, {meta.width})"
            )
    return meta
