"""DataLoader factory.

Handles:
    * WeightedRandomSampler for precipitation-stratified sampling
    * pinned memory + persistent workers
    * deterministic per-worker seeding
    * distributed-ready signature (world_size/rank args, currently pass-through)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

try:
    import torch  # type: ignore
    from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler, WeightedRandomSampler  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

from functools import partial

from ..seed import worker_init_fn as _seed_worker_init_fn
from .dataset import SolafuneDataset


# The DataLoader.worker_init_fn is called with a single arg (worker_id).
# functools.partial with a bound base_seed is picklable under Windows spawn
# and — unlike a module-level mutable global — cannot be corrupted by
# concurrent DataLoader construction.
def _make_worker_init(base_seed: int):
    return partial(_seed_worker_init_fn, base_seed=int(base_seed))


@dataclass
class DataLoaderConfig:
    batch_size: int = 16
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True
    shuffle_train: bool = True


def _build_precip_weights(
    dataset: SolafuneDataset,
    precip_weight_scale: float,
) -> np.ndarray:
    """Weight each sample by log1p(max GPM in scene) * scale + 1.

    Reads the GPM array in bulk (single slice) rather than one sample at a
    time so weight construction scales to 40k+ samples in <1 second.
    """
    n = len(dataset)
    backend = dataset._get_backend()  # noqa: SLF001 - documented internal
    indices = np.asarray(dataset._indices, dtype=np.int64)   # noqa: SLF001

    # Fast paths first: the two supported backends both expose their
    # underlying array (a zarr.Array or a numpy.memmap), which supports
    # advanced indexing directly.
    arr = getattr(backend, "_gpm_array", None)
    if arr is None:
        # Backend not yet open; open it via a read
        _ = backend.read_gpm_sample(int(indices[0]))
        arr = getattr(backend, "_gpm_array", None)
    if arr is None:
        raise RuntimeError("cache backend did not expose a gpm array")

    # zarr / memmap both accept ndarray fancy indexing
    gpm_slice = np.asarray(arr[indices], dtype=np.float32)  # (n, 41, 41)
    max_per_sample = np.nan_to_num(
        gpm_slice.reshape(n, -1).max(axis=1), nan=0.0, posinf=0.0, neginf=0.0
    )
    weights = 1.0 + precip_weight_scale * np.log1p(np.clip(max_per_sample, 0.0, None))
    return weights.astype(np.float64, copy=False)


def build_sampler(
    dataset: SolafuneDataset,
    strategy: str,
    precip_weight_scale: float = 3.0,
    generator: Optional["torch.Generator"] = None,
) -> "Sampler":
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required to build samplers")
    strategy = strategy.lower()
    if strategy in ("uniform", "random"):
        return RandomSampler(dataset, generator=generator)
    if strategy == "sequential":
        return SequentialSampler(dataset)
    if strategy == "precip_stratified":
        weights = _build_precip_weights(dataset, precip_weight_scale)
        return WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    raise ValueError(f"unknown sampling strategy: {strategy!r}")


def build_dataloader(
    dataset: SolafuneDataset,
    cfg: DataLoaderConfig,
    sampler: Optional["Sampler"] = None,
    shuffle: Optional[bool] = None,
    base_seed: int = 0,
) -> "DataLoader":
    """Return a `DataLoader` configured for AMP + pinned memory + persistent workers.

    If a `sampler` is passed, `shuffle` must be None.
    """
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required")
    kwargs = dict(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
        collate_fn=default_collate,
        worker_init_fn=_make_worker_init(base_seed),
    )
    if cfg.num_workers > 0:
        kwargs["persistent_workers"] = cfg.persistent_workers
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    if sampler is not None:
        kwargs["sampler"] = sampler
    else:
        kwargs["shuffle"] = cfg.shuffle_train if shuffle is None else shuffle
    return DataLoader(**kwargs)


def default_collate(batch):
    """Custom collate: keeps `unique_id` as a plain list of strings."""
    if not _HAS_TORCH:
        raise ImportError("PyTorch is required")
    from torch.utils.data._utils.collate import default_collate as _default_collate  # type: ignore
    uids = [item.pop("unique_id") for item in batch]
    collated = _default_collate(batch)
    collated["unique_id"] = uids
    return collated


def to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    """Move all tensors in a batch dict to `device`, preserving non-tensors."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out
