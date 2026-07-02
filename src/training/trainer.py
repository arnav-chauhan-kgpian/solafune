"""Training engine.

Feature set:
    * mixed precision (AMP)
    * gradient accumulation
    * gradient clipping
    * EMA
    * per-epoch validation with raw + EMA weights
    * checkpoint save + resume + early stopping
    * per-satellite / per-location metrics via MetricAccumulator
    * CSV + TensorBoard logging (optional)
    * VRAM and ETA reporting
"""
from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from ..logger import get_logger
from ..utils import format_bytes
from .callbacks import CheckpointSaver, EarlyStopping
from .ema import ExponentialMovingAverage
from .losses import CompositeLoss
from .metrics import MetricAccumulator, running_rmse_from_batch

log = get_logger(__name__)


@dataclass
class TrainerConfig:
    epochs: int = 50
    grad_accum_steps: int = 1
    grad_clip: float = 1.0
    amp: bool = True
    channels_last: bool = False
    ema_enabled: bool = True
    ema_decay: float = 0.9999
    ema_validate: bool = True
    early_stop_patience: int = 10
    monitor: str = "val_rmse"
    monitor_mode: str = "min"
    log_every_n_steps: int = 50
    keep_last_n_ckpt: int = 3
    output_dir: str = "outputs"
    step_scheduler_each_batch: bool = True
    rain_threshold_mm_h: float = 0.1
    heavy_threshold_mm_h: float = 10.0
    non_blocking: bool = True
    use_tensorboard: bool = True


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: CompositeLoss,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        cfg: TrainerConfig,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        if cfg.channels_last:
            self.model.to(memory_format=torch.channels_last)

        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=cfg.amp and self.device.type == "cuda",
        )
        self.ema = ExponentialMovingAverage(self.model, decay=cfg.ema_decay) if cfg.ema_enabled else None

        self.out_dir = Path(cfg.output_dir)
        self.ckpt = CheckpointSaver(
            self.out_dir / "checkpoints",
            keep_last_n=cfg.keep_last_n_ckpt,
            monitor=cfg.monitor, mode=cfg.monitor_mode,
        )
        self.early = EarlyStopping(patience=cfg.early_stop_patience, mode=cfg.monitor_mode)

        self._epoch = 0
        self._best: Optional[float] = None
        self._tb = None
        if cfg.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
                self._tb = SummaryWriter(str(self.out_dir / "tensorboard"))
            except Exception:  # pragma: no cover - tensorboard optional
                self._tb = None

        self._csv_train = self.out_dir / "train_metrics.csv"
        self._csv_val = self.out_dir / "val_metrics.csv"
        self._csv_train.parent.mkdir(parents=True, exist_ok=True)
        self._csv_headers: Dict[Path, List[str]] = {}

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def _append_csv(self, path: Path, row: Dict[str, Any]) -> None:
        headers = list(row.keys())
        prior = self._csv_headers.get(path)
        if prior is None:
            self._csv_headers[path] = headers
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(headers)
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row.get(h, "") for h in self._csv_headers[path]])

    def _tb_log(self, prefix: str, d: Mapping[str, float], step: int) -> None:
        if self._tb is None:
            return
        for k, v in d.items():
            try:
                self._tb.add_scalar(f"{prefix}/{k}", float(v), step)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    def resume_from(self, ckpt_path: Path) -> None:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(state["scheduler"])
            except Exception as e:  # pragma: no cover
                log.warning("scheduler resume failed: %r", e)
        if state.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(state["scaler"])
            except Exception:
                pass
        if state.get("ema") is not None and self.ema is not None:
            self.ema.load_state_dict(state["ema"])
        self._epoch = int(state.get("epoch", 0)) + 1
        self._best = state.get("best_val_metric")
        log.info("resumed from %s (epoch %d)", ckpt_path, self._epoch - 1)

    def try_auto_resume(self) -> None:
        p = CheckpointSaver.find_last(self.out_dir / "checkpoints")
        if p is not None:
            self.resume_from(p)

    # ------------------------------------------------------------------
    # Batch move
    # ------------------------------------------------------------------
    def _move(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, Tensor):
                out[k] = v.to(self.device, non_blocking=self.cfg.non_blocking)
            else:
                out[k] = v
        if self.cfg.channels_last and isinstance(out.get("sat"), Tensor):
            out["sat"] = out["sat"].to(memory_format=torch.channels_last)
        return out

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def fit(self) -> Optional[float]:
        cfg = self.cfg
        n_batches = len(self.train_loader)
        for epoch in range(self._epoch, cfg.epochs):
            self._epoch = epoch
            self.model.train()
            t0 = time.perf_counter()
            step_losses: Dict[str, float] = {}
            n_seen = 0
            self.optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(self.train_loader):
                batch = self._move(batch)
                with torch.amp.autocast("cuda", enabled=cfg.amp and self.device.type == "cuda"):
                    pred = self.model(batch["sat"], batch["aux"])
                    losses = self.loss_fn(pred, batch)
                    total = losses["total"] / cfg.grad_accum_steps
                self.scaler.scale(total).backward()

                if (step + 1) % cfg.grad_accum_steps == 0:
                    if cfg.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), cfg.grad_clip,
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)
                    if self.scheduler is not None and cfg.step_scheduler_each_batch:
                        try:
                            self.scheduler.step()
                        except Exception:
                            pass

                bs = batch["sat"].shape[0]
                n_seen += bs
                for k, v in losses.items():
                    step_losses[k] = step_losses.get(k, 0.0) + float(v.detach().item()) * bs
                # running RMSE (in log1p space is fine for progress display)
                if (step + 1) % cfg.log_every_n_steps == 0 or (step + 1) == n_batches:
                    lr = self.optimizer.param_groups[0]["lr"]
                    avg = {k: v / n_seen for k, v in step_losses.items()}
                    log.info(
                        "epoch %d [%d/%d] lr=%.2e loss=%.4f mse=%.4f bce=%.4f",
                        epoch, step + 1, n_batches, lr,
                        avg.get("total", 0.0), avg.get("mse", 0.0), avg.get("bce", 0.0),
                    )
                    row = {"epoch": epoch, "step": step + 1, "lr": lr, **avg}
                    self._append_csv(self._csv_train, row)
                    self._tb_log("train", avg, epoch * n_batches + step)

            epoch_secs = time.perf_counter() - t0
            self._tb_log("train", {"epoch_time_s": epoch_secs}, epoch)

            # ---------- validation ----------
            val_metric: Optional[float] = None
            if self.val_loader is not None:
                metrics_raw = self.validate(use_ema=False)
                self._log_val(metrics_raw, tag="raw")
                if self.ema is not None and cfg.ema_validate:
                    metrics_ema = self.validate(use_ema=True)
                    self._log_val(metrics_ema, tag="ema")
                    val_metric = metrics_ema.get(cfg.monitor.replace("val_", ""))
                if val_metric is None:
                    val_metric = metrics_raw.get(cfg.monitor.replace("val_", ""))

            # ---------- scheduler (epoch) ----------
            if self.scheduler is not None and not cfg.step_scheduler_each_batch:
                try:
                    if val_metric is not None:
                        self.scheduler.step(val_metric)
                    else:
                        self.scheduler.step()
                except Exception:
                    pass

            # ---------- checkpoint ----------
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict(),
                "ema": self.ema.state_dict() if self.ema is not None else None,
                "best_val_metric": self._best,
                "cfg": vars(cfg),
            }
            self.ckpt.save(state, epoch=epoch, val_metric=val_metric)

            if self.device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(self.device)
                log.info("epoch %d done in %.1fs; peak VRAM %s",
                         epoch, epoch_secs, format_bytes(peak))
                torch.cuda.reset_peak_memory_stats(self.device)

            if val_metric is not None:
                if self._best is None or (
                    (cfg.monitor_mode == "min" and val_metric < self._best)
                    or (cfg.monitor_mode == "max" and val_metric > self._best)
                ):
                    self._best = float(val_metric)
                if self.early.step(val_metric):
                    log.info("early stopping at epoch %d", epoch)
                    break

        if self._tb is not None:
            try:
                self._tb.close()
            except Exception:
                pass
        return self._best

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self, use_ema: bool = False) -> Dict[str, float]:
        assert self.val_loader is not None
        acc = MetricAccumulator(
            rain_threshold=self.cfg.rain_threshold_mm_h,
            heavy_threshold=self.cfg.heavy_threshold_mm_h,
        )
        ctx = (self.ema.apply(self.model) if (use_ema and self.ema is not None)
               else _null_context())
        self.model.eval()
        t0 = time.perf_counter()
        with ctx:
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = self._move(batch)
                    with torch.amp.autocast("cuda", enabled=self.cfg.amp
                                             and self.device.type == "cuda"):
                        pred = self.model(batch["sat"], batch["aux"])
                    # back to mm/h
                    pred_mm = torch.expm1(pred["mean"].float().clamp_min(0.0))
                    rain_prob = torch.sigmoid(pred["rain_logit"].float())
                    pred_mm = pred_mm * (rain_prob > 0.5).float()
                    acc.update(
                        pred_mm=pred_mm,
                        target_mm=batch["gpm_raw"].float(),
                        sat_id=batch["sat_id"],
                        location_id=batch["location_id"],
                    )
        elapsed = time.perf_counter() - t0
        out = acc.compute()
        out["val_time_s"] = float(elapsed)
        return out

    def _log_val(self, metrics: Mapping[str, float], tag: str) -> None:
        row = {"epoch": self._epoch, "tag": tag, **{k: v for k, v in metrics.items()}}
        self._append_csv(self._csv_val, row)
        # log the headline metrics
        headline = {k: metrics.get(k, float("nan"))
                    for k in ("rmse", "mae", "bias", "rain_f1", "heavy_rmse", "csi", "pod")}
        log.info("val[%s] " + " ".join(f"{k}={v:.4f}" for k, v in headline.items()), tag)
        # tb
        self._tb_log(f"val/{tag}", metrics, self._epoch)


class _null_context:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False
