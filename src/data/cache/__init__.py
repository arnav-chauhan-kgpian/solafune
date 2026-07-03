"""Cache backends for preprocessed satellite/GPM arrays."""
from .base import CacheBackend, CacheSpec
from .memmap_backend import MemmapBackend
from .zarr_backend import ZarrBackend

BACKENDS = {
    "zarr": ZarrBackend,
    "memmap": MemmapBackend,
}


def get_backend(name: str) -> type[CacheBackend]:
    """Return a backend class by name."""
    name = name.lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown cache backend: {name!r}. Available: {list(BACKENDS)}")
    return BACKENDS[name]


__all__ = [
    "CacheBackend",
    "CacheSpec",
    "ZarrBackend",
    "MemmapBackend",
    "get_backend",
    "BACKENDS",
]
