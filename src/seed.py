"""Deterministic seeding utilities.

The goal is bit-for-bit reproducibility on identical hardware and the same
software stack. On different GPUs / CuDNN versions minor numerical drift
remains — this is a PyTorch limitation, not a bug in this module.
"""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed every source of randomness we use.

    Args:
        seed: The global seed.
        deterministic: When True, forces cuDNN into deterministic mode (slower
            but reproducible). Set to False for benchmarking runs.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int, base_seed: Optional[int] = None) -> None:
    """DataLoader worker seeding hook.

    Each worker receives a distinct but deterministic seed derived from
    `base_seed + worker_id`. Use with `torch.utils.data.DataLoader(worker_init_fn=...)`.
    """
    if base_seed is None:
        base_seed = int(os.environ.get("PYTHONHASHSEED", "0"))
    seed = (base_seed + worker_id) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    if _HAS_TORCH:
        torch.manual_seed(seed)
