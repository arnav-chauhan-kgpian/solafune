"""Frozen dataset and satellite constants.

These constants are derived from the frozen implementation specification and
verified empirically against the workspace dataset. They are the source of
truth for satellite band structure, native spatial dimensions, satellite
identifiers and expected data types.
"""
from __future__ import annotations

from typing import Final, Mapping, Tuple

# ---------------------------------------------------------------------------
# Satellite identifiers
# ---------------------------------------------------------------------------
HIMAWARI: Final[str] = "himawari"
GOES: Final[str] = "goes"
METEOSAT: Final[str] = "meteosat"

SATELLITES: Final[Tuple[str, ...]] = (HIMAWARI, GOES, METEOSAT)

SATELLITE_ID: Final[Mapping[str, int]] = {
    HIMAWARI: 0,
    GOES: 1,
    METEOSAT: 2,
}
SATELLITE_NAME: Final[Mapping[int, str]] = {v: k for k, v in SATELLITE_ID.items()}

# ---------------------------------------------------------------------------
# Native spatial dimensions (height, width)
# ---------------------------------------------------------------------------
NATIVE_SIZES: Final[Mapping[str, Tuple[int, int]]] = {
    HIMAWARI: (81, 81),
    GOES: (141, 141),
    METEOSAT: (144, 144),
}

# GPM target resolution
GPM_SIZE: Final[Tuple[int, int]] = (41, 41)

# ---------------------------------------------------------------------------
# Band configuration
# All satellite TIFs have 16 bands (uint8, 0-255). Bands are 1-indexed in the
# original satellite documentation. Here we use 0-indexed band positions when
# slicing arrays.
# ---------------------------------------------------------------------------
NUM_BANDS_TOTAL: Final[int] = 16

# IR band indices (0-indexed within the 16-band stack).
# Empirically confirmed by day/night analysis:
#   * Himawari/GOES: bands 7-16 are thermal IR (indices 6-15)
#   * Meteosat: bands 9-16 are thermal IR (indices 8-15)
IR_BAND_INDICES: Final[Mapping[str, Tuple[int, ...]]] = {
    HIMAWARI: tuple(range(6, 16)),   # 10 bands (b07..b16)
    GOES: tuple(range(6, 16)),       # 10 bands (b07..b16)
    METEOSAT: tuple(range(8, 16)),   # 8 bands (b09..b16)
}

VISIBLE_BAND_INDICES: Final[Mapping[str, Tuple[int, ...]]] = {
    HIMAWARI: tuple(range(0, 6)),
    GOES: tuple(range(0, 6)),
    METEOSAT: tuple(range(0, 8)),
}

ALL_BAND_INDICES: Final[Mapping[str, Tuple[int, ...]]] = {
    sat: tuple(range(NUM_BANDS_TOTAL)) for sat in SATELLITES
}

BAND_MODES: Final[Tuple[str, ...]] = ("ir_only", "all", "visible_only")


def band_indices_for(satellite: str, mode: str) -> Tuple[int, ...]:
    """Return the band indices to use for a satellite in the requested mode."""
    if satellite not in SATELLITES:
        raise ValueError(f"Unknown satellite: {satellite!r}")
    if mode == "ir_only":
        return IR_BAND_INDICES[satellite]
    if mode == "all":
        return ALL_BAND_INDICES[satellite]
    if mode == "visible_only":
        return VISIBLE_BAND_INDICES[satellite]
    raise ValueError(f"Unknown band mode: {mode!r}. Expected one of {BAND_MODES}")


# Maximum number of active channels across satellites for a given mode.
# Used to determine the encoder input channel width.
def max_active_channels(mode: str) -> int:
    return max(len(band_indices_for(s, mode)) for s in SATELLITES)


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------
FRAMES_PER_SAMPLE: Final[int] = 3

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
SAT_RAW_DTYPE: Final[str] = "uint8"
GPM_RAW_DTYPE: Final[str] = "float32"

# ---------------------------------------------------------------------------
# CSV column names
# ---------------------------------------------------------------------------
COL_UID: Final[str] = "unique_id"
COL_LOCATION: Final[str] = "name_location"
COL_SATELLITE: Final[str] = "satellite_target"
COL_DATETIME: Final[str] = "datetime"
COL_FRAMES: Final[str] = "last_30_minutes_observation_filename"
COL_GPM: Final[str] = "gpm_imerg_filename"

REQUIRED_COLUMNS: Final[Tuple[str, ...]] = (
    COL_UID, COL_LOCATION, COL_SATELLITE, COL_DATETIME, COL_FRAMES, COL_GPM,
)

# ---------------------------------------------------------------------------
# Filesystem subdirectory names
# ---------------------------------------------------------------------------
SAT_SUBDIRS: Final[Mapping[str, str]] = {
    HIMAWARI: "himawari",
    GOES: "goes",
    METEOSAT: "meteosat",
}
GPM_SUBDIR: Final[str] = "gpm_imerg"

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
CACHE_BACKENDS: Final[Tuple[str, ...]] = ("zarr", "memmap")
DEFAULT_CACHE_BACKEND: Final[str] = "zarr"

# ---------------------------------------------------------------------------
# Rain threshold (mm/h) for rain/no-rain classification
# ---------------------------------------------------------------------------
DEFAULT_RAIN_THRESHOLD_MM_H: Final[float] = 0.1
