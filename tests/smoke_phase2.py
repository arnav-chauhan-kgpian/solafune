"""Phase-2 end-to-end smoke test.

Verifies the entire pipeline on a small synthetic-workspace-backed cache:

    build cache → dataset → dataloader → model forward → loss → backward
    → optimizer step → validation → checkpoint save → checkpoint reload
    → EMA validation → inference → prediction saving

Also runs a single-batch overfit test as a sanity check that gradients flow.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.constants import SATELLITES, max_active_channels
from src.data.cache import get_backend
from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
from src.data.dataset import DatasetConfig, SolafuneDataset
from src.data.normalization import compute_norm_stats, save_norm_stats
from src.data.preprocessing import build_cache, build_cache_spec
from src.inference.predict import PredictionConfig, predict
from src.logger import get_logger
from src.models import build_model
from src.paths import sat_tif_path
from src.seed import seed_everything
from src.training.ema import ExponentialMovingAverage
from src.training.losses import build_loss
from src.training.metrics import MetricAccumulator
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer, TrainerConfig
from src.utils import parse_frame_list

log = get_logger("smoke")


# Import the synthetic-workspace fixture logic from conftest
from tests.conftest import _write_sat_tif, _write_gpm_tif, HAVE_RASTERIO  # noqa: E402


def build_tiny_workspace(root: Path):
    """3 samples per satellite = 9 rows total."""
    from src.constants import GPM_SUBDIR, SAT_SUBDIRS, FRAMES_PER_SAMPLE
    seed = 0
    rows = []
    for sat_idx, sat in enumerate(SATELLITES):
        for i in range(4):
            frames = []
            for t in range(FRAMES_PER_SAMPLE):
                fname = f"synth_{sat}_{i}_{t}.tif"
                _write_sat_tif(root / SAT_SUBDIRS[sat] / fname, sat, seed)
                frames.append(fname); seed += 1
            gpm_name = f"synth_{sat}_{i}_gpm.tif"
            _write_gpm_tif(root / GPM_SUBDIR / gpm_name, seed); seed += 1
            rows.append({
                "unique_id": f"{sat_idx:02d}-{i:04d}",
                "name_location": f"loc_{sat}_{i}",
                "satellite_target": sat,
                "datetime": f"2024-01-0{i+1} 12:00:00",
                "last_30_minutes_observation_filename": str(frames),
                "gpm_imerg_filename": gpm_name,
            })
    df = pd.DataFrame(rows)
    csv_path = root / "train_dataset.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def build_workspace_cache(root: Path, csv: Path, cache_dir: Path):
    df = pd.read_csv(csv)
    spec, _ = build_cache_spec(df, cache_dir, "ir_only")
    backend = get_backend("zarr")(spec, compressor="lz4")
    build_cache(csv, root, cache_dir, backend, "ir_only", load_gpm=True, verbose_every=0)
    backend.close()
    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[row["satellite_target"]].append(sat_tif_path(root, row["satellite_target"], f))
    stats = compute_norm_stats(paths, max_files_per_satellite=20, pixel_stride=1)
    norm_path = cache_dir / "norm.json"
    save_norm_stats(norm_path, stats)
    return norm_path


def build_smoke_model(train_ds):
    c_per = max_active_channels("ir_only")
    return build_model({
        "in_channels_per_frame": c_per,
        "n_frames": 3,
        "n_diff_frames": 2,
        "encoder": "resnet34",
        "temporal": "none",
        "decoder": "unet",
        "probabilistic": True,
        "encoder_kwargs": {"norm": "group"},
        "decoder_kwargs": {"norm": "group",
                           "decoder_channels": [128, 64, 32, 16]},
    })


def main() -> int:
    if not HAVE_RASTERIO:
        log.error("rasterio not available; cannot run smoke test")
        return 1

    seed_everything(42)
    workspace = Path(tempfile.mkdtemp(prefix="smoke_phase2_"))
    try:
        # ---------------- Setup ----------------
        log.info("[1/12] build synthetic workspace")
        csv, df = build_tiny_workspace(workspace)
        log.info("[2/12] build cache + norm stats")
        cache_dir = workspace / "cache"
        norm_path = build_workspace_cache(workspace, csv, cache_dir)

        log.info("[3/12] build datasets + dataloaders")
        ds_cfg = DatasetConfig(
            cache_dir=cache_dir, csv_path=csv, norm_stats_path=norm_path,
            image_size=64, bands="ir_only", include_diff_frames=True,
        )
        train_ds = SolafuneDataset(ds_cfg, df=df, indices=list(range(6)))
        val_ds = SolafuneDataset(ds_cfg, df=df, indices=list(range(6, 12)))
        dl_cfg = DataLoaderConfig(batch_size=2, num_workers=0, pin_memory=False,
                                    persistent_workers=False, drop_last=False, prefetch_factor=2)
        sampler = build_sampler(train_ds, "precip_stratified")
        train_loader = build_dataloader(train_ds, dl_cfg, sampler=sampler, base_seed=0)
        val_loader = build_dataloader(val_ds, dl_cfg, shuffle=False, base_seed=0)

        log.info("[4/12] build model")
        model = build_smoke_model(train_ds)
        n_params = sum(p.numel() for p in model.parameters())
        log.info("model params: %d", n_params)

        log.info("[5/12] one forward pass on real batch")
        batch = next(iter(train_loader))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(batch_dev["sat"], batch_dev["aux"])
        assert "mean" in out and "rain_logit" in out and "log_var" in out
        assert out["mean"].shape == (2, 41, 41)
        assert not torch.isnan(out["mean"]).any()
        log.info("[6/12] forward OK, output shape %s", tuple(out["mean"].shape))

        log.info("[7/12] loss + backward + optimizer step")
        loss_fn = build_loss({
            "mse_weight": 1.0, "bce_weight": 0.5, "nll_weight": 0.3,
            "gradient_weight": 0.01, "ssim_weight": 0.05,
            "rain_weighted": True, "rain_weight_scale": 3.0,
        })
        optimizer = build_optimizer(model, {"name": "adamw", "lr": 1e-3})
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(batch_dev["sat"], batch_dev["aux"])
            losses = loss_fn(pred, batch_dev)
        assert torch.isfinite(losses["total"]), f"non-finite loss: {losses}"
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        assert torch.isfinite(gn), f"non-finite grad norm: {gn}"
        log.info("loss=%.4f grad_norm=%.4f", float(losses["total"]), float(gn))

        log.info("[8/12] single-batch overfit sanity")
        _overfit_check(model, batch_dev, loss_fn, device, steps=30)

        log.info("[9/12] full validation pass (metrics)")
        scheduler, step_each_batch = build_scheduler(
            optimizer, {"name": "cosine"}, steps_per_epoch=len(train_loader), epochs=2,
        )
        tcfg = TrainerConfig(
            epochs=2, amp=(device.type == "cuda"), ema_enabled=True, ema_decay=0.9,
            output_dir=str(workspace / "out"),
            log_every_n_steps=1, use_tensorboard=False,
            step_scheduler_each_batch=step_each_batch,
            early_stop_patience=10,
        )
        trainer = Trainer(model, optimizer, scheduler, loss_fn,
                          train_loader, val_loader, tcfg, device=device)
        val_raw = trainer.validate(use_ema=False)
        val_ema = trainer.validate(use_ema=True)
        log.info("val raw rmse=%.3f csi=%.3f  ema rmse=%.3f csi=%.3f",
                 val_raw["rmse"], val_raw["csi"], val_ema["rmse"], val_ema["csi"])
        assert np.isfinite(val_raw["rmse"])

        log.info("[10/12] 2-epoch training (verifies checkpoint save + resume)")
        best = trainer.fit()
        log.info("training done. best val metric: %s", best)
        ckpt_dir = Path(workspace) / "out" / "checkpoints"
        assert (ckpt_dir / "last.pt").exists()
        assert (ckpt_dir / "best.pt").exists()

        log.info("[11/12] resume test")
        model2 = build_smoke_model(train_ds).to(device)
        optimizer2 = build_optimizer(model2, {"name": "adamw", "lr": 1e-3})
        sched2, seb2 = build_scheduler(
            optimizer2, {"name": "cosine"}, steps_per_epoch=len(train_loader), epochs=2,
        )
        tcfg2 = TrainerConfig(**{**vars(tcfg), "epochs": 3})
        tcfg2.output_dir = str(workspace / "out")
        tcfg2.step_scheduler_each_batch = seb2
        trainer2 = Trainer(model2, optimizer2, sched2, loss_fn,
                           train_loader, val_loader, tcfg2, device=device)
        trainer2.try_auto_resume()
        assert trainer2._epoch >= 1, f"resume did not advance epoch: {trainer2._epoch}"
        # Model weights should match the saved checkpoint
        loaded = torch.load(ckpt_dir / "last.pt", map_location="cpu", weights_only=False)
        for k, v in loaded["model"].items():
            assert torch.allclose(model2.state_dict()[k].cpu(), v.cpu()), f"{k} mismatch"
        log.info("resume verified (epoch=%d)", trainer2._epoch)

        log.info("[12/12] inference + submission")
        pcfg = PredictionConfig(amp=(device.type == "cuda"), tta=True, rain_mask_threshold=0.15)
        preds = predict(model2, val_loader, pcfg, device=device)
        assert preds.shape[1:] == (41, 41)
        assert np.isfinite(preds).all()
        log.info("inference OK, preds shape %s min=%.3f max=%.3f",
                 preds.shape, preds.min(), preds.max())

        # NaN input handling
        log.info("[NaN check] injecting NaN into input")
        bad = batch_dev.copy()
        bad["sat"] = bad["sat"].clone()
        bad["sat"][0, 0, 0, 0] = float("nan")
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out_bad = model2(bad["sat"], bad["aux"])
        # Expect NaN to propagate (the model does not silently swallow it —
        # we want the training loop to skip such batches via has_data mask
        # or the loss to detect it).
        has_nan = bool(torch.isnan(out_bad["mean"]).any())
        log.info("model propagates NaN as expected: %s", has_nan)

        log.info("=== SMOKE TEST: ALL 12 STAGES PASSED ===")
        return 0

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _overfit_check(model, batch, loss_fn, device, steps: int = 30) -> None:
    """Train for `steps` iterations on a single batch and assert loss decreases."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    initial: float = float("nan")
    final: float = float("nan")
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            pred = model(batch["sat"], batch["aux"])
            losses = loss_fn(pred, batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(opt); scaler.update()
        if step == 0:
            initial = float(losses["total"].detach())
        if step == steps - 1:
            final = float(losses["total"].detach())
    log.info("overfit: initial=%.4f final=%.4f (ratio=%.3f)", initial, final, final / max(initial, 1e-6))
    assert final < initial * 0.95, f"single-batch overfit failed: {initial:.4f} → {final:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
