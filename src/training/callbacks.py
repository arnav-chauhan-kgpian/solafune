"""Training callbacks: early stopping + checkpoint management."""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class EarlyStopping:
    patience: int = 10
    mode: str = "min"
    min_delta: float = 1e-5
    _best: Optional[float] = None
    _bad: int = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if self._best is None:
            self._best = metric
            return False
        improve = (self._best - metric) if self.mode == "min" else (metric - self._best)
        if improve > self.min_delta:
            self._best = metric
            self._bad = 0
            return False
        self._bad += 1
        return self._bad >= self.patience


class CheckpointSaver:
    """Rolling checkpoint saver.

    Writes `epoch_N.pt` per epoch and keeps only the last `keep_last_n`.
    Writes `best.pt` when a monitored metric improves.
    Always writes `last.pt` (a copy of the most recent epoch).
    """

    def __init__(
        self,
        out_dir: Path,
        keep_last_n: int = 3,
        monitor: str = "val_rmse",
        mode: str = "min",
    ):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = int(keep_last_n)
        self.monitor = monitor
        self.mode = mode
        self._best: Optional[float] = None
        self._epoch_paths: list[Path] = []

    def save(self, state: Dict[str, Any], epoch: int,
             val_metric: Optional[float] = None) -> Dict[str, Path]:
        state = dict(state)
        state["epoch"] = int(epoch)
        p_epoch = self.out_dir / f"epoch_{epoch:03d}.pt"
        torch.save(state, p_epoch)
        self._epoch_paths.append(p_epoch)
        while len(self._epoch_paths) > self.keep_last_n:
            old = self._epoch_paths.pop(0)
            try:
                old.unlink()
            except OSError:
                pass
        # last
        p_last = self.out_dir / "last.pt"
        shutil.copyfile(p_epoch, p_last)
        # best
        saved = {"epoch": p_epoch, "last": p_last}
        if val_metric is not None:
            better = (self._best is None
                      or (self.mode == "min" and val_metric < self._best)
                      or (self.mode == "max" and val_metric > self._best))
            if better:
                self._best = float(val_metric)
                p_best = self.out_dir / "best.pt"
                shutil.copyfile(p_epoch, p_best)
                saved["best"] = p_best
        return saved

    @staticmethod
    def find_last(out_dir: Path) -> Optional[Path]:
        p = Path(out_dir) / "last.pt"
        return p if p.exists() else None
