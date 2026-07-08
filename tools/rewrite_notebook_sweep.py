"""Rewrite the Kaggle notebook to use a single sweep cell that trains
+ predicts every (EXPERIMENT, SEED) in one Save & Run All pass.

Deletes standalone cells 11 (build model), 12 (trainer.fit),
16 (inference), 17 (submission) — all folded into the sweep loop.

Keeps cells 13/14 (plots) and 18 (audit) but retargets them to
`SWEEP[-1]`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB = Path("D:/solafune/notebooks/solafune_kaggle.ipynb")

MD_SWEEP = """## 5. Sweep — Training + Inference (all runs in one pass)

Set `SWEEP` below to a list of `(EXPERIMENT, SEED, EPOCHS)` tuples. The cell trains, predicts, and writes a submission for each one, storing them under `submissions/<experiment>_seed<seed>/`. Section 8 then ensembles whatever's in there.

**Timing per run** (T4, batch=64, cosine scheduler):
- Cache + norm-stats already built → ~55 min per training run
- +5 min per inference pass
- **Budget: 60 min × N runs**. Session limit is 12h; a 3-run sweep fits comfortably.

Default sweep produces a 2-seed ensemble of the improved baseline. Uncomment the third row for a stronger 3-model ensemble with Conv3D temporal integration.

**No cell-by-cell interaction needed** — a single Save & Run All runs the whole sweep."""

SWEEP_CODE = '''import sys
if "/kaggle/working/solafune" not in sys.path:
    sys.path.insert(0, "/kaggle/working/solafune")

import gc, time, torch, shutil, pandas as pd
from pathlib import Path
from omegaconf import OmegaConf

from src.constants import max_active_channels
from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
from src.data.dataset import DatasetConfig, SolafuneDataset, split_indices_by_location
from src.experiment.tracker import snapshot_run
from src.models import build_model
from src.seed import seed_everything
from src.training.losses import build_loss
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer, TrainerConfig
from src.training.ema import ExponentialMovingAverage
from src.inference.predict import PredictionConfig, predict
from src.inference.submission import write_submission

# ---------------------------------------------------------------------------
# SWEEP — list of (EXPERIMENT, SEED, EPOCHS). Each row = one full training +
# inference run. Every run writes its submission to
# /kaggle/working/submissions/<experiment>_seed<seed>/.
# ---------------------------------------------------------------------------
SWEEP = [
    ("exp0_baseline", 42, 30),
    ("exp0_baseline", 43, 30),
    # ("exp2_conv3d",           42, 30),   # uncomment for a 3-model ensemble
    # ("exp2_conv3d_ensemble",  42, 30),
    # ("exp6_efficientnet",     42, 30),
]

# ---------------------------------------------------------------------------
# TRAINING PROFILE — baked in for every sweep run
# ---------------------------------------------------------------------------
TRAINING_OVERRIDES = [
    "training.ema_enabled=false",
    "training.early_stop_patience=12",
    "data.batch_size=64",
    "scheduler=cosine",
    "scheduler.warmup_epochs=2",
    "optimizer.lr=1.5e-3",
    "loss.rain_weight_scale=1.5",
    "loss.mae_weight=0.3",
]

# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------
from hydra import compose, initialize_config_dir
import hydra
HYDRA_DIR = str(Path(REPO_DIR) / "configs")


def _make_cfg(experiment, seed, epochs, training_dir):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    overrides = [
        f"data.cache_dir={CACHE_ROOT}/train",
        f"data.norm_stats_path={NORM_STATS}",
        f"data.train_csv={TRAIN_CSV}",
        f"data.eval_csv={EVAL_CSV}",
        f"data.train_root={CACHE_ROOT}/train",
        f"data.eval_root={CACHE_ROOT}/eval",
        f"data.cache.backend={BACKEND}",
        f"training.output_dir={training_dir}",
        f"training.epochs={epochs}",
        f"seed={seed}",
        f"experiment={experiment}",
    ] + TRAINING_OVERRIDES
    with initialize_config_dir(config_dir=HYDRA_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


def _run_one(experiment, seed, epochs):
    print(f"\\n{'=' * 70}\\n=== TRAINING {experiment} seed={seed} epochs={epochs}\\n{'=' * 70}")
    t_start = time.perf_counter()

    training_dir = Path(OUT_DIR) / "training" / experiment / f"seed_{seed}"
    training_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_cfg(experiment, seed, epochs, training_dir)

    seed_everything(int(cfg.seed))
    snapshot_run(training_dir, cfg, repo_root=Path(REPO_DIR))

    # ----- data -----
    ds_cfg = DatasetConfig(
        cache_dir=Path(cfg.data.cache_dir),
        csv_path=Path(cfg.data.train_csv),
        norm_stats_path=Path(cfg.data.norm_stats_path),
        image_size=int(cfg.data.image_size),
        bands=str(cfg.data.bands),
        include_diff_frames=bool(cfg.data.include_diff_frames),
        missing_frame_strategy=str(cfg.data.missing_frame_strategy),
        cache_backend=BACKEND,
    )
    df = pd.read_csv(ds_cfg.csv_path)
    train_idx, val_idx = split_indices_by_location(df, list(cfg.data.val_locations))
    train_ds = SolafuneDataset(ds_cfg, df=df, indices=train_idx)
    val_ds = SolafuneDataset(ds_cfg, df=df, indices=val_idx)
    print(f"    train: {len(train_ds)}   val: {len(val_ds)}")

    dl = DataLoaderConfig(
        batch_size=int(cfg.data.batch_size), num_workers=int(cfg.data.num_workers),
        pin_memory=True, persistent_workers=True, prefetch_factor=2, drop_last=True,
    )
    train_loader = build_dataloader(
        train_ds, dl,
        sampler=build_sampler(train_ds, str(cfg.data.sampling.strategy),
                              precip_weight_scale=float(cfg.data.sampling.precip_weight_scale)),
        base_seed=int(cfg.seed),
    )
    val_loader = build_dataloader(val_ds, dl, shuffle=False, base_seed=int(cfg.seed))

    # ----- model -----
    mcfg = OmegaConf.to_container(cfg.model, resolve=True)
    mcfg["in_channels_per_frame"] = max_active_channels(str(cfg.data.bands))
    mcfg["n_frames"] = int(cfg.data.frames)
    mcfg["n_diff_frames"] = int(cfg.data.frames - 1) if cfg.data.include_diff_frames else 0
    model = build_model(mcfg)
    print(f"    params: {sum(p.numel() for p in model.parameters()):,}")

    # ----- training -----
    loss_fn = build_loss(OmegaConf.to_container(cfg.loss))
    opt = build_optimizer(model, OmegaConf.to_container(cfg.optimizer))
    sched, seb = build_scheduler(
        opt, OmegaConf.to_container(cfg.scheduler),
        steps_per_epoch=len(train_loader), epochs=int(cfg.training.epochs),
    )
    tcfg = TrainerConfig(**OmegaConf.to_container(cfg.training))
    tcfg.step_scheduler_each_batch = seb
    trainer = Trainer(model, opt, sched, loss_fn, train_loader, val_loader, tcfg)
    trainer.try_auto_resume()
    best_val = trainer.fit()
    print(f"    best val metric: {best_val}")

    # ----- load best.pt for inference -----
    best_ckpt = training_dir / "checkpoints" / "best.pt"
    state = torch.load(best_ckpt, map_location="cpu", weights_only=False)
    weight_source = state.get("best_source", "raw")
    if weight_source == "ema" and state.get("ema") is not None:
        ema = ExponentialMovingAverage(model, decay=cfg.training.ema_decay)
        ema.load_state_dict(state["ema"])
        ema.apply(model).__enter__()
        print("    loaded EMA weights")
    else:
        model.load_state_dict(state["model"])
        print("    loaded RAW weights")

    # ----- inference -----
    eval_ds_cfg = DatasetConfig(
        cache_dir=CACHE_ROOT / "eval",
        csv_path=EVAL_CSV,
        norm_stats_path=NORM_STATS,
        image_size=int(cfg.data.image_size), bands="ir_only",
        include_diff_frames=True, cache_backend=BACKEND,
    )
    eval_ds = SolafuneDataset(eval_ds_cfg)
    eval_loader = build_dataloader(
        eval_ds,
        DataLoaderConfig(batch_size=32, num_workers=2, pin_memory=True,
                         persistent_workers=True, prefetch_factor=2, drop_last=False),
        shuffle=False, base_seed=int(cfg.seed),
    )
    preds = predict(model, eval_loader,
                    PredictionConfig(amp=True, tta=True, rain_mask_threshold=0.15))
    print(f"    predictions: shape={preds.shape} min/max={preds.min():.3f}/{preds.max():.3f}"
          f" mean={preds.mean():.4f} nonzero%={(preds > 0).mean() * 100:.1f}%")

    # ----- submission -----
    sub_root = Path(OUT_DIR) / "submissions" / f"{experiment}_seed{seed}"
    test_files = sub_root / "test_files"
    n_written = write_submission(preds, EVAL_CSV, test_files)
    shutil.copy(EVAL_CSV, sub_root / "evaluation_target.csv")
    archive = shutil.make_archive(str(Path(OUT_DIR) / f"submission_{experiment}_seed{seed}"),
                                    "zip", str(sub_root))
    # stable-name copy for the audit cell
    stable_sub = Path(OUT_DIR) / "submission"
    if stable_sub.exists():
        shutil.rmtree(stable_sub)
    shutil.copytree(sub_root, stable_sub)

    dt = time.perf_counter() - t_start
    print(f"    files: {n_written}   archive: {archive} ({Path(archive).stat().st_size / 1e6:.1f} MB)")
    print(f"    elapsed: {dt:.0f}s ({dt / 60:.1f} min)")

    # free memory before next run
    del trainer, model, opt, sched, train_loader, val_loader, eval_loader, preds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val


results = []
for experiment, seed, epochs in SWEEP:
    val = _run_one(experiment, seed, epochs)
    results.append((experiment, seed, val))

print(f"\\n\\n{'=' * 70}\\n=== SWEEP COMPLETE ===\\n{'=' * 70}")
for exp, seed, val in results:
    print(f"  {exp:<25} seed={seed}   best_val_metric={val}")
'''

PLOT_CODE = '''import sys
if "/kaggle/working/solafune" not in sys.path:
    sys.path.insert(0, "/kaggle/working/solafune")

# Plot the LAST run in the SWEEP.
from src.visualization import plot_training_curves, plot_val_curves
last_exp, last_seed, _ = SWEEP[-1]
last_dir = Path(OUT_DIR) / "training" / last_exp / f"seed_{last_seed}"
plot_training_curves(last_dir / "train_metrics.csv", last_dir / "plots" / "train.png")
plot_val_curves(last_dir / "val_metrics.csv", last_dir / "plots" / "val.png")
print(f"plots for {last_exp} seed={last_seed} saved in {last_dir}/plots")
'''

INF_MD = """## 6. Submission audit (last run in sweep)

The sweep cell above already wrote a submission zip per run. This cell only audits format on the LAST run's files. The stable path `/kaggle/working/submission/` always points to the last completed run.

For per-run submission zips, look at `/kaggle/working/submission_<experiment>_seed<seed>.zip`."""

ITER_MD = """## 9. Iteration workflow — one Save & Run All

Everything above runs top-to-bottom in a single Save & Run All pass:

1. Cell 2: clone repo, wire paths
2. Cell 4: cache backend benchmark (~30 s, cached after first run)
3. Cell 6: build train + eval Zarr caches (~2 h, cached after first run)
4. Cell 8: norm stats (~2 min, cached after first run)
5. **Cell 10 (sweep):** trains + predicts every `(EXPERIMENT, SEED)` in `SWEEP`. ~60 min per row.
6. Plots + audit cells reference the LAST sweep run automatically.
7. Cell 20 (ensemble): averages every subdirectory under `submissions/`, writes `submission_ensemble.zip`.

**To change what runs**: edit the `SWEEP` list at the top of cell 10. Comment / uncomment rows. Nothing else needs touching."""


def _replace_source(cell, text):
    lines = text.split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    if "outputs" in cell:
        cell["outputs"] = []
    if "execution_count" in cell:
        cell["execution_count"] = None


def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Replace cell 9 (markdown intro to training) with the sweep-intro
    _replace_source(cells[9], MD_SWEEP)
    cells[9]["cell_type"] = "markdown"

    # Replace cell 10 with the sweep runner
    _replace_source(cells[10], SWEEP_CODE)
    cells[10]["cell_type"] = "code"

    # Delete cells 11, 12, 16, 17 (indices before deletion). Delete high→low.
    for i in sorted([17, 16, 12, 11], reverse=True):
        del cells[i]

    # Find and update plot cell + inference markdown + iteration markdown
    for i, c in enumerate(cells):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        if c["cell_type"] == "code" and "plot_training_curves" in src:
            _replace_source(c, PLOT_CODE)
        elif c["cell_type"] == "markdown" and "Inference + Submission" in src:
            _replace_source(c, INF_MD)
        elif c["cell_type"] == "markdown" and "Iteration workflow" in src:
            _replace_source(c, ITER_MD)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("wrote", NB)
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
        first = src.split("\n")[0][:70]
        print(f"  {i:2d} [{c['cell_type'][:4]}] {first}")


if __name__ == "__main__":
    main()
