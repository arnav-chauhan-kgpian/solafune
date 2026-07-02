"""SolafuneDataset — lazy per-sample loader with in-getitem resize.

The dataset reads native-resolution uint8 satellite arrays from a
`CacheBackend`, applies band selection, per-satellite z-score normalization,
resize to the target image size, temporal diff frame computation, and
optional augmentation. It emits a dict of tensors ready for the model.

Design constraints (from the frozen spec):
    * cache is native resolution; resize is here, not baked in
    * image_size is configurable per experiment without rebuilding cache
    * band mode is configurable
    * missing frames are handled by repeat-last or zero-padding
    * augmentation is applied consistently to satellite and GPM target
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    import torch.nn.functional as F  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Dataset = object  # type: ignore
    F = None  # type: ignore
    _HAS_TORCH = False

from ..constants import (
    COL_DATETIME,
    COL_FRAMES,
    COL_GPM,
    COL_LOCATION,
    COL_SATELLITE,
    COL_UID,
    FRAMES_PER_SAMPLE,
    GPM_SIZE,
    NUM_BANDS_TOTAL,
    SATELLITE_ID,
    SATELLITES,
    band_indices_for,
    max_active_channels,
)
from ..logger import get_logger
from ..utils import parse_frame_list, read_json
from .cache.base import CacheBackend
from .cache import get_backend
from .normalization import NormStats, load_norm_stats
from .cache.base import CacheSpec

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass for the dataset
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Immutable dataset configuration."""

    cache_dir: Path
    csv_path: Path
    norm_stats_path: Path
    image_size: int = 96
    interpolation: str = "bilinear"           # bilinear | nearest | bicubic
    resize_backend: str = "torch"             # torch | cv2
    bands: str = "ir_only"                    # ir_only | all | visible_only
    include_diff_frames: bool = True
    missing_frame_strategy: str = "repeat_last"  # repeat_last | zero
    rain_threshold: float = 0.1
    cache_backend: str = "zarr"
    pad_channels_to_max: bool = True


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SolafuneDataset(Dataset):
    """Sample dict emitted by __getitem__::

        {
            "sat":          Tensor  (C_in, H, W) float32
            "gpm_log1p":    Tensor  (41, 41)     float32
            "gpm_raw":      Tensor  (41, 41)     float32
            "rain_mask":    Tensor  (41, 41)     float32
            "sat_id":       Tensor  ()           int64
            "location_id":  Tensor  ()           int64
            "has_data":     Tensor  ()           float32   (1.0 if all frames present)
            "valid_mask":   Tensor  (T,)         float32
            "aux":          Tensor  (aux_dim,)   float32
            "unique_id":    str
        }
    """

    def __init__(
        self,
        cfg: DatasetConfig,
        df: Optional[pd.DataFrame] = None,
        indices: Optional[Sequence[int]] = None,
        transform: Optional[Any] = None,
    ):
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required for SolafuneDataset")
        self.cfg = cfg
        self._csv_path = Path(cfg.csv_path)
        self._cache_dir = Path(cfg.cache_dir)
        self._df_all = df if df is not None else pd.read_csv(self._csv_path)
        if indices is None:
            self._indices: List[int] = list(range(len(self._df_all)))
        else:
            self._indices = list(int(i) for i in indices)
        self._transform = transform

        # locations mapping
        unique_locations = sorted(self._df_all[COL_LOCATION].unique().tolist())
        self._location_to_id: Dict[str, int] = {
            loc: i for i, loc in enumerate(unique_locations)
        }

        # load cache index + spec
        idx_path = self._cache_dir / "index.json"
        spec_path = self._cache_dir / "spec.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"cache index not found: {idx_path}")
        if not spec_path.exists():
            raise FileNotFoundError(f"cache spec not found: {spec_path}")
        self._index = read_json(idx_path)
        self._spec = CacheSpec.from_dict(read_json(spec_path))
        self._global_to_local = self._index["global_to_local"]

        # normalization stats
        if not Path(cfg.norm_stats_path).exists():
            raise FileNotFoundError(f"norm stats not found: {cfg.norm_stats_path}")
        self._norm_stats: NormStats = load_norm_stats(Path(cfg.norm_stats_path))

        # cache backend — instantiated lazily per worker
        self._backend_cls = get_backend(cfg.cache_backend)
        self._backend: Optional[CacheBackend] = None

        # band indices per satellite (into the *cached* 16-band or IR-subset array)
        cached_mode = "all"  # We infer cache layout from spec channel count
        # But cache was built with a specific band_mode; the spec's C encodes it.
        # For safety: if the cache C equals ir_only length, treat cache as ir_only.
        self._cache_band_mode = self._infer_cache_band_mode()
        self._active_band_indices = self._compute_active_indices()

        # precompute normalization arrays per satellite (in float32) for the
        # *active* bands only.
        self._mean = {}
        self._std = {}
        for sat in SATELLITES:
            m, s = self._norm_stats.mean_std_arrays(sat)
            active = self._active_band_indices[sat]
            # If cache stored subset, these active indices refer to cache positions,
            # but norm stats are keyed on full 16-band positions. Translate:
            full_positions = self._cache_position_to_full_band(sat)
            band_full_idx = [full_positions[i] for i in active]
            self._mean[sat] = m[band_full_idx].astype(np.float32)
            self._std[sat] = s[band_full_idx].astype(np.float32)

        # pad channels to a uniform width across satellites (for batching)
        self._max_active_c = max_active_channels(self.cfg.bands)

    # ------------------------------------------------------------------
    # Cache layout introspection
    # ------------------------------------------------------------------
    def _infer_cache_band_mode(self) -> str:
        """Given the cache spec channel counts, figure out which band mode was cached."""
        expected = {
            m: {s: len(band_indices_for(s, m)) for s in SATELLITES}
            for m in ("ir_only", "all", "visible_only")
        }
        cache_c = {s: self._spec.per_sat_shapes[s][1] for s in SATELLITES}
        for mode, exp in expected.items():
            if all(cache_c[s] == exp[s] for s in SATELLITES):
                return mode
        raise RuntimeError(
            f"cache channel counts {cache_c} do not match any known band mode"
        )

    def _cache_position_to_full_band(self, satellite: str) -> Tuple[int, ...]:
        """Return the 16-band indices that correspond to cache positions."""
        return band_indices_for(satellite, self._cache_band_mode)

    def _compute_active_indices(self) -> Dict[str, Tuple[int, ...]]:
        """Given cache layout + requested band mode, compute per-satellite
        indices *into the cached array* for the active bands.
        """
        result: Dict[str, Tuple[int, ...]] = {}
        for sat in SATELLITES:
            cache_positions = self._cache_position_to_full_band(sat)
            wanted_full = band_indices_for(sat, self.cfg.bands)
            # each wanted full-band idx must be in cache_positions
            try:
                positions = tuple(cache_positions.index(b) for b in wanted_full)
            except ValueError as e:
                raise ValueError(
                    f"cache mode {self._cache_band_mode} does not contain all bands "
                    f"required for {self.cfg.bands} on {sat}: {e}"
                )
            result[sat] = positions
        return result

    # ------------------------------------------------------------------
    # Backend handle (created lazily so each worker owns its own)
    # ------------------------------------------------------------------
    def _get_backend(self) -> CacheBackend:
        if self._backend is None:
            self._backend = self._backend_cls(self._spec)  # type: ignore[call-arg]
        return self._backend

    def __del__(self) -> None:
        # Best-effort backend close on GC; DataLoader workers may not call this
        # explicitly. Wrapped to avoid interpreter-shutdown NoneType errors.
        try:
            b = getattr(self, "_backend", None)
            if b is not None:
                b.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        global_idx = self._indices[i]
        row = self._df_all.iloc[global_idx]
        sat = str(row[COL_SATELLITE]).lower().strip()
        sat_id = SATELLITE_ID[sat]
        location = str(row[COL_LOCATION])
        location_id = self._location_to_id[location]

        backend = self._get_backend()

        # (T, C_cached, H_native, W_native) uint8
        local_idx = self._global_to_local[global_idx]
        sat_raw = backend.read_sat_sample(sat, int(local_idx))
        gpm = backend.read_gpm_sample(int(global_idx))
        valid_mask = backend.read_valid_mask(int(global_idx))

        # band selection
        active = list(self._active_band_indices[sat])
        sat_active = sat_raw[:, active, :, :]  # (T, C_active, H, W)

        # missing frame strategy
        sat_active, valid_mask = self._apply_missing_frame_strategy(sat_active, valid_mask)

        # normalize (per-satellite per-band z-score)
        mean = self._mean[sat]
        std = self._std[sat]
        sat_norm = (sat_active.astype(np.float32) - mean[None, :, None, None]) / std[None, :, None, None]

        # to torch, resize
        sat_t = torch.from_numpy(sat_norm)  # (T, C, H, W)
        H = W = self.cfg.image_size
        if sat_t.shape[-2] != H or sat_t.shape[-1] != W:
            sat_t = F.interpolate(
                sat_t,
                size=(H, W),
                mode=self.cfg.interpolation,
                align_corners=False if self.cfg.interpolation in ("bilinear", "bicubic") else None,
            )
        # temporal diff frames
        if self.cfg.include_diff_frames:
            diff = sat_t[1:] - sat_t[:-1]           # (T-1, C, H, W)
            stacked = torch.cat([sat_t, diff], dim=0)  # (2T-1, C, H, W)
        else:
            stacked = sat_t

        # flatten temporal into channel dim: (T*C, H, W)
        Tf, Cf, Hf, Wf = stacked.shape
        stacked = stacked.reshape(Tf * Cf, Hf, Wf).contiguous()

        # pad channels to uniform width
        if self.cfg.pad_channels_to_max:
            expected_c = self._expected_input_channels()
            if stacked.shape[0] < expected_c:
                pad_c = expected_c - stacked.shape[0]
                pad = torch.zeros((pad_c, Hf, Wf), dtype=stacked.dtype)
                stacked = torch.cat([stacked, pad], dim=0)

        # augmentation (transform applies to sat + gpm together)
        gpm_t = torch.from_numpy(gpm.astype(np.float32))
        if self._transform is not None:
            stacked, gpm_t = self._transform(stacked, gpm_t)

        # target: log1p + rain mask
        gpm_raw = gpm_t
        gpm_log1p = torch.log1p(torch.clamp(gpm_raw, min=0.0))
        rain_mask = (gpm_raw > self.cfg.rain_threshold).float()

        has_data = float(valid_mask.sum() > 0)

        # aux features
        aux = self._build_aux(row, sat_id)

        return {
            "sat": stacked.float(),
            "gpm_log1p": gpm_log1p,
            "gpm_raw": gpm_raw,
            "rain_mask": rain_mask,
            "sat_id": torch.tensor(sat_id, dtype=torch.long),
            "location_id": torch.tensor(location_id, dtype=torch.long),
            "has_data": torch.tensor(has_data, dtype=torch.float32),
            "valid_mask": torch.from_numpy(valid_mask.astype(np.float32)),
            "aux": aux,
            "unique_id": str(row[COL_UID]),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _apply_missing_frame_strategy(
        self,
        sat: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fill zeroed frames according to configured strategy.

        The fill logic is driven strictly by the boolean mask; it never
        inspects tensor contents (a valid frame could legitimately be all
        zero for some band slices).
        """
        if self.cfg.missing_frame_strategy == "zero":
            return sat, mask
        if self.cfg.missing_frame_strategy != "repeat_last":
            raise ValueError(
                f"unknown missing_frame_strategy: {self.cfg.missing_frame_strategy!r}"
            )
        out = sat.copy()
        # Track which frames are "logically valid" (originally valid OR filled).
        filled_from: Optional[int] = None
        # Forward pass: propagate the last-seen valid frame forward.
        for t in range(sat.shape[0]):
            if mask[t] == 1:
                filled_from = t
            elif filled_from is not None:
                out[t] = out[filled_from]
        # Backward pass: leading missing frames (before any valid frame) get
        # filled from the first subsequent valid frame.
        if filled_from is None:
            # No valid frames anywhere; leave zeros, mask stays all-zero.
            return out, mask
        first_valid: Optional[int] = None
        for t in range(sat.shape[0]):
            if mask[t] == 1:
                first_valid = t
                break
        if first_valid is not None and first_valid > 0:
            for t in range(first_valid):
                if mask[t] == 0:
                    out[t] = out[first_valid]
        return out, mask

    def _expected_input_channels(self) -> int:
        """The channel count the model input tensor is padded to."""
        c_max = self._max_active_c
        n_temporal = FRAMES_PER_SAMPLE + (FRAMES_PER_SAMPLE - 1 if self.cfg.include_diff_frames else 0)
        return c_max * n_temporal

    def _build_aux(self, row: pd.Series, sat_id: int) -> "torch.Tensor":
        """Build the auxiliary feature vector.

        Layout (dim=6):
            [sat_onehot(3), cos(solar_zenith_proxy), sin(hour), cos(hour)]
        We compute a cheap solar-zenith proxy from hour-of-day. A precise
        astronomical zenith requires lat/lon which we don't have per sample;
        the proxy is adequate as an auxiliary feature.
        """
        onehot = np.zeros(len(SATELLITES), dtype=np.float32)
        onehot[sat_id] = 1.0
        dt = pd.to_datetime(row[COL_DATETIME])
        hour = dt.hour + dt.minute / 60.0
        angle = 2 * np.pi * hour / 24.0
        sin_h = float(np.sin(angle))
        cos_h = float(np.cos(angle))
        # zenith proxy: peaks around 12h; night ~ -1, noon ~ +1
        zenith_proxy = float(np.cos(2 * np.pi * (hour - 12.0) / 24.0))
        aux = np.concatenate([onehot, [zenith_proxy, sin_h, cos_h]], dtype=np.float32)
        return torch.from_numpy(aux)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def input_channels(self) -> int:
        return self._expected_input_channels()

    @property
    def n_locations(self) -> int:
        return len(self._location_to_id)


def split_indices_by_location(
    df: pd.DataFrame,
    val_locations: Sequence[str],
) -> Tuple[List[int], List[int]]:
    """Return (train_indices, val_indices) using a geographic holdout."""
    val_set = {v.lower() for v in val_locations}
    is_val = df[COL_LOCATION].str.lower().isin(val_set)
    val_indices = df.index[is_val].tolist()
    train_indices = df.index[~is_val].tolist()
    return train_indices, val_indices
