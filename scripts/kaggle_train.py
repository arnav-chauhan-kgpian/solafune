"""Kaggle single-file entry point.

Copy-paste into a Kaggle notebook cell (with the workspace added as a Dataset)
and run. It wires the full pipeline end-to-end with sensible Kaggle defaults:

    1. Detect cached backend recommendation (or build cache if missing)
    2. Compute norm stats if missing
    3. Train the baseline (or the experiment named via env var EXPERIMENT)
    4. Generate a submission from the best checkpoint

Configure via environment variables:
    KAGGLE_DATA_ROOT       — root dir containing train + eval subdirs
    OUTPUT_DIR             — where to write cache + checkpoints (default /kaggle/working)
    EXPERIMENT             — experiment name (default exp0_baseline)
    EPOCHS                 — override training epochs (default from config)
    RESUME                 — 1 to resume from OUTPUT_DIR/last.pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# repository must be sys.path[0]
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _env(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v else default


def main() -> int:
    data_root = Path(_env("KAGGLE_DATA_ROOT", "/kaggle/input/solafune-precip"))
    out_root = Path(_env("OUTPUT_DIR", "/kaggle/working"))
    experiment = _env("EXPERIMENT", "exp0_baseline")

    cache_root = out_root / "cache"
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    train_root = data_root / "train_dataset_b1c74968f2f24eaeb2852b47b80a581e"
    eval_root = data_root / "evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d"
    if not train_root.exists():
        train_root = data_root
    if not eval_root.exists():
        eval_root = data_root
    train_csv = train_root / "train_dataset.csv"
    eval_csv = eval_root / "evaluation_target.csv"

    print(f"data_root={data_root}")
    print(f"train_csv={train_csv}")
    print(f"out_root={out_root}")
    print(f"experiment={experiment}")

    # 1) Benchmark backend if not already picked
    backend_json = cache_root / "backend.json"
    if not backend_json.exists():
        from src.data.cache.benchmark import run_benchmark
        print("running cache backend benchmark...")
        run_benchmark(output_path=backend_json, n_samples=200, channels=10, hw=81)
    import json
    backend_name = json.loads(backend_json.read_text())["recommended"]
    print(f"cache backend: {backend_name}")

    # 2) Build train + eval caches if missing
    from src.constants import SATELLITES
    from src.data.cache import get_backend
    from src.data.preprocessing import build_cache, build_cache_spec
    from src.data.normalization import compute_norm_stats, save_norm_stats
    from src.paths import sat_tif_path
    from src.utils import parse_frame_list
    import pandas as pd

    def _build_if_needed(csv, root, out_dir, load_gpm):
        spec_path = out_dir / "spec.json"
        if spec_path.exists():
            print(f"cache present: {out_dir}")
            return
        df = pd.read_csv(csv)
        spec, _ = build_cache_spec(df, out_dir, "ir_only")
        cls = get_backend(backend_name)
        b = cls(spec, compressor="lz4") if backend_name == "zarr" else cls(spec)
        build_cache(csv, root, out_dir, b, "ir_only",
                    load_gpm=load_gpm, verbose_every=1000)
        b.close()
        print(f"cache built: {out_dir}")

    _build_if_needed(train_csv, train_root, cache_root / "train", load_gpm=True)
    _build_if_needed(eval_csv, eval_root, cache_root / "eval", load_gpm=False)

    # 3) Norm stats
    norm_path = cache_root / "norm_stats.json"
    if not norm_path.exists():
        print("computing norm stats...")
        train_df = pd.read_csv(train_csv)
        paths = {s: [] for s in SATELLITES}
        for _, row in train_df.iterrows():
            for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
                paths[row["satellite_target"]].append(
                    sat_tif_path(train_root, row["satellite_target"], f)
                )
        stats = compute_norm_stats(paths, max_files_per_satellite=500, pixel_stride=2)
        save_norm_stats(norm_path, stats)
    print(f"norm stats ready: {norm_path}")

    # 4) Training via Hydra
    from hydra import compose, initialize_config_dir
    import hydra
    hydra_dir = str(_REPO / "configs")
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=hydra_dir, version_base=None):
        overrides = [
            f"data.cache_dir={cache_root}/train",
            f"data.norm_stats_path={norm_path}",
            f"data.train_csv={train_csv}",
            f"data.eval_csv={eval_csv}",
            f"data.train_root={train_root}",
            f"data.eval_root={eval_root}",
            f"data.cache.backend={backend_name}",
            f"training.output_dir={out_root}/training",
            f"+experiment={experiment}",
        ]
        epochs_env = os.environ.get("EPOCHS")
        if epochs_env:
            overrides.append(f"training.epochs={int(epochs_env)}")
        cfg = compose(config_name="config", overrides=overrides)
    print("config assembled; launching training...")

    # Import and call the train script's main body directly to reuse setup
    from scripts import __init__  # noqa: F401 - ensures scripts is a package
    sys.path.insert(0, str(_REPO / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_module", _REPO / "scripts" / "03_train.py")
    mod = importlib.util.module_from_spec(spec)
    # sys.argv-safe: hydra.main not used here; call our helpers directly
    from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
    from src.data.dataset import DatasetConfig, SolafuneDataset, split_indices_by_location
    from src.experiment.tracker import snapshot_run
    from src.models import build_model
    from src.seed import seed_everything
    from src.training.losses import build_loss
    from src.training.schedulers import build_optimizer, build_scheduler
    from src.training.trainer import Trainer, TrainerConfig
    from src.constants import max_active_channels
    from omegaconf import OmegaConf

    seed_everything(int(cfg.seed))
    snapshot_run(Path(cfg.training.output_dir), cfg, repo_root=_REPO)

    ds_cfg = DatasetConfig(
        cache_dir=Path(cfg.data.cache_dir),
        csv_path=Path(cfg.data.train_csv),
        norm_stats_path=Path(cfg.data.norm_stats_path),
        image_size=int(cfg.data.image_size),
        bands=str(cfg.data.bands),
        include_diff_frames=bool(cfg.data.include_diff_frames),
        missing_frame_strategy=str(cfg.data.missing_frame_strategy),
        cache_backend=backend_name,
    )
    df = pd.read_csv(ds_cfg.csv_path)
    train_idx, val_idx = split_indices_by_location(df, list(cfg.data.val_locations))
    train_ds = SolafuneDataset(ds_cfg, df=df, indices=train_idx)
    val_ds = SolafuneDataset(ds_cfg, df=df, indices=val_idx)
    dl = DataLoaderConfig(
        batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers),
        pin_memory=bool(cfg.data.pin_memory), persistent_workers=bool(cfg.data.persistent_workers),
        prefetch_factor=int(cfg.data.prefetch_factor), drop_last=bool(cfg.data.drop_last),
    )
    train_loader = build_dataloader(
        train_ds, dl, sampler=build_sampler(train_ds, str(cfg.data.sampling.strategy),
        precip_weight_scale=float(cfg.data.sampling.precip_weight_scale)),
        base_seed=int(cfg.seed),
    )
    val_loader = build_dataloader(val_ds, dl, shuffle=False, base_seed=int(cfg.seed))
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    mcfg["in_channels_per_frame"] = max_active_channels(str(cfg.data.bands))
    mcfg["n_frames"] = int(cfg.data.frames)
    mcfg["n_diff_frames"] = int(cfg.data.frames - 1) if cfg.data.include_diff_frames else 0
    model = build_model(mcfg)
    loss_fn = build_loss(OmegaConf.to_container(cfg.loss))
    opt = build_optimizer(model, OmegaConf.to_container(cfg.optimizer))
    sched, seb = build_scheduler(opt, OmegaConf.to_container(cfg.scheduler),
                                  steps_per_epoch=len(train_loader),
                                  epochs=int(cfg.training.epochs))
    tcfg = TrainerConfig(**OmegaConf.to_container(cfg.training))
    tcfg.step_scheduler_each_batch = seb
    trainer = Trainer(model, opt, sched, loss_fn, train_loader, val_loader, tcfg)
    if os.environ.get("RESUME") == "1":
        trainer.try_auto_resume()
    trainer.fit()

    # 5) Inference + submission
    print("running inference on eval set...")
    from src.inference.predict import PredictionConfig, predict
    from src.inference.submission import write_submission
    from src.training.callbacks import CheckpointSaver
    import torch
    ckpt = CheckpointSaver.find_last(Path(tcfg.output_dir) / "checkpoints")
    if ckpt is None:
        raise RuntimeError("no checkpoint found for inference")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    eval_ds_cfg = DatasetConfig(
        cache_dir=Path(cache_root) / "eval",
        csv_path=Path(cfg.data.eval_csv),
        norm_stats_path=Path(cfg.data.norm_stats_path),
        image_size=int(cfg.data.image_size), bands="ir_only",
        include_diff_frames=True, cache_backend=backend_name,
    )
    eval_ds = SolafuneDataset(eval_ds_cfg)
    eval_loader = build_dataloader(
        eval_ds, DataLoaderConfig(batch_size=32, num_workers=2, pin_memory=True,
                                    persistent_workers=True, prefetch_factor=2,
                                    drop_last=False),
        shuffle=False, base_seed=int(cfg.seed),
    )
    preds = predict(model, eval_loader,
                    PredictionConfig(amp=True, tta=True, rain_mask_threshold=0.15))
    sub_dir = Path(cfg.training.output_dir) / "submission" / "test_files"
    write_submission(preds, Path(cfg.data.eval_csv), sub_dir)
    print(f"submission written to {sub_dir}")
    # copy the evaluation_target.csv into submission dir per Solafune format
    import shutil
    shutil.copy(Path(cfg.data.eval_csv), sub_dir.parent / "evaluation_target.csv")
    print("=== Kaggle pipeline complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
