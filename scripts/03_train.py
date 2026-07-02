"""Main training entry point.

Usage::
    python scripts/03_train.py                        # baseline
    python scripts/03_train.py +experiment=exp0_baseline
    python scripts/03_train.py training.epochs=2 data.batch_size=8   # ad-hoc override
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
from src.data.dataset import DatasetConfig, SolafuneDataset, split_indices_by_location
from src.experiment.tracker import snapshot_run
from src.logger import get_logger
from src.models import build_model
from src.seed import seed_everything
from src.training.losses import build_loss
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer, TrainerConfig


log = get_logger("train")


def _build_datasets(cfg: DictConfig):
    ds_cfg = DatasetConfig(
        cache_dir=Path(cfg.data.cache_dir) / "train",
        csv_path=Path(cfg.data.train_csv),
        norm_stats_path=Path(cfg.data.norm_stats_path),
        image_size=int(cfg.data.image_size),
        interpolation=str(cfg.data.interpolation),
        resize_backend=str(cfg.data.resize_backend),
        bands=str(cfg.data.bands),
        include_diff_frames=bool(cfg.data.include_diff_frames),
        missing_frame_strategy=str(cfg.data.missing_frame_strategy),
        rain_threshold=float(cfg.data.rain_threshold),
        cache_backend=str(cfg.data.cache.backend),
    )
    df = pd.read_csv(ds_cfg.csv_path)
    train_idx, val_idx = split_indices_by_location(df, list(cfg.data.val_locations))
    train_ds = SolafuneDataset(ds_cfg, df=df, indices=train_idx)
    val_ds = SolafuneDataset(ds_cfg, df=df, indices=val_idx)
    return train_ds, val_ds


def _build_loaders(cfg: DictConfig, train_ds, val_ds):
    dl_cfg = DataLoaderConfig(
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
        drop_last=bool(cfg.data.drop_last),
        shuffle_train=bool(cfg.data.shuffle_train),
    )
    sampler = build_sampler(
        train_ds,
        strategy=str(cfg.data.sampling.strategy),
        precip_weight_scale=float(cfg.data.sampling.precip_weight_scale),
    )
    train_loader = build_dataloader(train_ds, dl_cfg, sampler=sampler,
                                    base_seed=int(cfg.seed))
    val_dl_cfg = DataLoaderConfig(
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor),
        drop_last=False,
        shuffle_train=False,
    )
    val_loader = build_dataloader(val_ds, val_dl_cfg, shuffle=False,
                                  base_seed=int(cfg.seed))
    return train_loader, val_loader


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    snapshot_run(Path(cfg.output_dir), cfg, repo_root=_ROOT)
    log.info("output dir: %s", cfg.output_dir)

    train_ds, val_ds = _build_datasets(cfg)
    log.info("train samples: %d  val samples: %d", len(train_ds), len(val_ds))
    train_loader, val_loader = _build_loaders(cfg, train_ds, val_ds)

    # model input channel width comes from the dataset
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model_cfg["in_channels_per_frame"] = int(_max_active_channels(cfg))
    model_cfg["n_frames"] = int(cfg.data.frames)
    model_cfg["n_diff_frames"] = int(cfg.data.frames - 1) if cfg.data.include_diff_frames else 0
    model = build_model(model_cfg)
    log.info("model params: %d", model.num_parameters())

    loss_fn = build_loss(OmegaConf.to_container(cfg.loss))
    optimizer = build_optimizer(model, OmegaConf.to_container(cfg.optimizer))
    scheduler, step_each_batch = build_scheduler(
        optimizer, OmegaConf.to_container(cfg.scheduler),
        steps_per_epoch=len(train_loader), epochs=int(cfg.training.epochs),
    )
    tcfg = TrainerConfig(**OmegaConf.to_container(cfg.training))
    tcfg.output_dir = str(cfg.output_dir)
    tcfg.step_scheduler_each_batch = step_each_batch
    trainer = Trainer(model, optimizer, scheduler, loss_fn,
                       train_loader, val_loader, tcfg)
    trainer.try_auto_resume()
    trainer.fit()


def _max_active_channels(cfg: DictConfig) -> int:
    from src.constants import max_active_channels
    return max_active_channels(str(cfg.data.bands))


if __name__ == "__main__":
    main()
