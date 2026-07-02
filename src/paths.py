"""Path resolution utilities.

The dataset directory layout (verified on the workspace):
    <root>/
        train_dataset.csv or evaluation_target.csv
        himawari/
            <filename>.tif
        goes/
            <filename>.tif
        meteosat/
            <filename>.tif
        gpm_imerg/
            <filename>.tif  (train only; for eval this directory contains placeholders)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .constants import GPM_SUBDIR, SAT_SUBDIRS

PathLike = Union[str, Path]


@dataclass(frozen=True)
class DataPaths:
    """Container for resolved data paths."""

    train_csv: Path
    eval_csv: Path
    train_root: Path
    eval_root: Path
    cache_dir: Path
    norm_stats_path: Path

    @staticmethod
    def from_config(cfg) -> "DataPaths":
        """Build a DataPaths from a Hydra data config."""
        return DataPaths(
            train_csv=Path(cfg.train_csv),
            eval_csv=Path(cfg.eval_csv),
            train_root=Path(cfg.train_root),
            eval_root=Path(cfg.eval_root),
            cache_dir=Path(cfg.cache_dir),
            norm_stats_path=Path(cfg.norm_stats_path),
        )


def sat_tif_path(root: PathLike, satellite: str, filename: str) -> Path:
    """Return the absolute path of a satellite TIF file."""
    if satellite not in SAT_SUBDIRS:
        raise ValueError(f"Unknown satellite: {satellite!r}")
    return Path(root) / SAT_SUBDIRS[satellite] / filename


def gpm_tif_path(root: PathLike, filename: str) -> Path:
    """Return the absolute path of a GPM IMERG TIF file."""
    return Path(root) / GPM_SUBDIR / filename


def ensure_dir(path: PathLike) -> Path:
    """Create the directory if it does not exist and return it."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
