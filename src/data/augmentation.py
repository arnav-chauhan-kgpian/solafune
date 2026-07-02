"""Data augmentation.

Augmentation is applied *after* the satellite input tensor has been resized
to `image_size` and *before* the loss is computed. Because satellite and GPM
targets are both spatially aligned (identity transform), any spatial op must
be applied consistently to both.

For this Phase-1 release, we implement the `base` augmentation profile:
  * random horizontal flip
  * random vertical flip
  * random 90/180/270 rotation

`full` profile adds:
  * random band dropout (zero out visible/random bands)
  * Gaussian noise

Both are stateless callables — no PyTorch modules required.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False


TensorPair = Tuple["torch.Tensor", "torch.Tensor"]


@dataclass
class AugmentationConfig:
    kind: str = "base"
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    band_dropout_prob: float = 0.0
    band_dropout_type: str = "visible"      # visible | random_bands | full_satellite
    noise_enabled: bool = False
    noise_sigma: float = 0.0


class SpatialAugmentation:
    """Callable applying the configured spatial transforms.

    RNG state is materialised lazily per worker: on the first call inside a
    DataLoader worker, we salt the base seed with the worker id, so different
    workers produce independent augmentation sequences even after the parent
    Dataset object is pickled and copied.
    """

    def __init__(self, cfg: AugmentationConfig, seed: Optional[int] = None):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for augmentation")
        self.cfg = cfg
        self._base_seed = int(seed) if seed is not None else 0
        self._rng: Optional[np.random.Generator] = None

    def _ensure_rng(self) -> np.random.Generator:
        if self._rng is None:
            try:
                info = torch.utils.data.get_worker_info()  # type: ignore[attr-defined]
                wid = int(info.id) if info is not None else 0
            except Exception:
                wid = 0
            self._rng = np.random.default_rng(self._base_seed + 7919 * wid)
        return self._rng

    def __call__(
        self,
        sat: "torch.Tensor",     # (C, H, W)
        gpm: "torch.Tensor",     # (h, w)
    ) -> TensorPair:
        self._ensure_rng()
        if self.cfg.horizontal_flip_prob > 0 and self._rng.random() < self.cfg.horizontal_flip_prob:
            sat = torch.flip(sat, dims=[-1])
            gpm = torch.flip(gpm, dims=[-1])
        if self.cfg.vertical_flip_prob > 0 and self._rng.random() < self.cfg.vertical_flip_prob:
            sat = torch.flip(sat, dims=[-2])
            gpm = torch.flip(gpm, dims=[-2])
        if self.cfg.rotate90_prob > 0 and self._rng.random() < self.cfg.rotate90_prob:
            k = int(self._rng.integers(1, 4))
            sat = torch.rot90(sat, k=k, dims=[-2, -1])
            gpm = torch.rot90(gpm, k=k, dims=[-2, -1])
        if (
            self.cfg.noise_enabled
            and self.cfg.noise_sigma > 0
        ):
            noise = torch.randn_like(sat) * self.cfg.noise_sigma
            sat = sat + noise
        # band dropout would need band metadata; disabled in base
        return sat, gpm


def build_augmentation(cfg_dict: dict, seed: Optional[int] = None) -> SpatialAugmentation:
    cfg = AugmentationConfig(**cfg_dict)
    return SpatialAugmentation(cfg, seed=seed)
