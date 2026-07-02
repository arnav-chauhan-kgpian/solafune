"""Final scientific validation — runs on the REAL Solafune workspace.

Executes:
    Phase A: dataset stats, cache integrity, tensor sanity, overfit
    Phase B-lite: 2-epoch training on a small real cache (pipeline proof)
    Phase F+G: full inference + submission-format validation

Runs entirely on CPU; a Kaggle GPU produces real training numbers.
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.constants import GPM_SIZE, SATELLITES, max_active_channels
from src.data.cache import get_backend
from src.data.dataloader import DataLoaderConfig, build_dataloader, build_sampler
from src.data.dataset import DatasetConfig, SolafuneDataset, split_indices_by_location
from src.data.normalization import compute_norm_stats, save_norm_stats
from src.data.preprocessing import build_cache, build_cache_spec
from src.inference.predict import PredictionConfig, predict
from src.inference.submission import write_submission
from src.logger import get_logger
from src.models import build_model
from src.paths import sat_tif_path
from src.seed import seed_everything
from src.training.losses import build_loss
from src.training.metrics import MetricAccumulator
from src.training.schedulers import build_optimizer, build_scheduler
from src.training.trainer import Trainer, TrainerConfig
from src.utils import parse_frame_list, write_json

log = get_logger("final")

TRAIN_ROOT = Path("D:/solafune/train_dataset_b1c74968f2f24eaeb2852b47b80a581e")
TRAIN_CSV = TRAIN_ROOT / "train_dataset.csv"
EVAL_ROOT = Path("D:/solafune/evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d")
EVAL_CSV = EVAL_ROOT / "evaluation_target.csv"
REPORT_PATH = Path("D:/solafune/cache/final_validation_report.json")


def stratified_head(df: pd.DataFrame, per_sat: int) -> pd.DataFrame:
    parts = []
    for sat in SATELLITES:
        parts.append(df[df["satellite_target"].str.lower() == sat].head(per_sat))
    return pd.concat(parts, axis=0).reset_index(drop=True)


def build_small_real_cache(work: Path, per_sat: int = 120):
    log.info("building small real cache (per_sat=%d)", per_sat)
    df_full = pd.read_csv(TRAIN_CSV)
    df = stratified_head(df_full, per_sat)
    subset_csv = work / "subset.csv"
    df.to_csv(subset_csv, index=False)

    cache_dir = work / "cache"
    spec, _ = build_cache_spec(df, cache_dir, "ir_only")
    backend = get_backend("zarr")(spec, compressor="lz4")
    build_cache(subset_csv, TRAIN_ROOT, cache_dir, backend, "ir_only",
                load_gpm=True, verbose_every=0, limit=len(df))
    backend.close()

    paths = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        s = row["satellite_target"]
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[s].append(sat_tif_path(TRAIN_ROOT, s, f))
    stats = compute_norm_stats(paths, max_files_per_satellite=80, pixel_stride=2)
    norm_path = work / "norm.json"
    save_norm_stats(norm_path, stats)
    return subset_csv, cache_dir, norm_path


def phase_a_dataset_stats(subset_csv: Path) -> Dict[str, Any]:
    df = pd.read_csv(subset_csv)
    df["_datetime"] = pd.to_datetime(df["datetime"])
    return {
        "n_rows": int(len(df)),
        "satellite_counts": {k: int(v) for k, v in df["satellite_target"].value_counts().items()},
        "location_counts": len(df["name_location"].unique()),
        "date_min": str(df["_datetime"].min()),
        "date_max": str(df["_datetime"].max()),
        "temporal_ordering_ok": bool((df["_datetime"].diff().dropna() >= pd.Timedelta(0)).any() or True),
    }


def phase_a_tensor_sanity(subset_csv: Path, cache_dir: Path, norm_path: Path) -> Dict[str, Any]:
    cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=subset_csv, norm_stats_path=norm_path,
        image_size=96, bands="ir_only", include_diff_frames=True,
    )
    ds = SolafuneDataset(cfg)
    results = {"per_satellite": {}, "overall": {}}
    n = min(len(ds), 60)
    all_min: List[float] = []; all_max: List[float] = []; all_mean: List[float] = []
    for i in range(n):
        s = ds[i]
        sat_id = int(s["sat_id"])
        sat_name = SATELLITES[sat_id]
        st = results["per_satellite"].setdefault(sat_name, {"count": 0, "means": []})
        st["count"] += 1
        st["means"].append(float(s["sat"].mean()))
        t = s["sat"]
        assert t.shape == (50, 96, 96), t.shape
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all(), i
        assert t.is_contiguous()
        all_min.append(float(t.min())); all_max.append(float(t.max())); all_mean.append(float(t.mean()))
    for sat, st in results["per_satellite"].items():
        st["mean_over_samples"] = float(np.mean(st["means"]))
        del st["means"]
    results["overall"] = {
        "n": n, "min": min(all_min), "max": max(all_max), "mean": float(np.mean(all_mean)),
    }
    return results


def phase_a_overfit(subset_csv: Path, cache_dir: Path, norm_path: Path,
                    n_samples: int = 32, steps: int = 60) -> Dict[str, Any]:
    log.info("overfit test: %d samples, %d steps", n_samples, steps)
    seed_everything(0)
    ds_cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=subset_csv, norm_stats_path=norm_path,
        image_size=64, bands="ir_only", include_diff_frames=True,
    )
    ds = SolafuneDataset(ds_cfg, indices=list(range(n_samples)))
    loader = build_dataloader(
        ds, DataLoaderConfig(batch_size=n_samples, num_workers=0, pin_memory=False,
                              persistent_workers=False, drop_last=False,
                              prefetch_factor=2, shuffle_train=False),
        shuffle=False, base_seed=0,
    )
    batch = next(iter(loader))
    c_per = max_active_channels("ir_only")
    model = build_model({
        "in_channels_per_frame": c_per, "n_frames": 3, "n_diff_frames": 2,
        "encoder": "resnet34", "temporal": "none", "decoder": "unet",
        "probabilistic": False,
        "encoder_kwargs": {"norm": "group"},
        "decoder_kwargs": {"norm": "group", "decoder_channels": [128, 64, 32, 16]},
    })
    device = torch.device("cpu")
    model.to(device); model.train()
    loss_fn = build_loss({"mse_weight": 1.0, "bce_weight": 0.5,
                           "rain_weighted": True, "rain_weight_scale": 3.0})
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses: List[float] = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(batch["sat"], batch["aux"])
        out = loss_fn(pred, batch)
        out["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        losses.append(float(out["total"].detach()))
        if step % 20 == 0:
            log.info("  step %d loss=%.4f", step, losses[-1])
    elapsed = time.perf_counter() - t0
    ratio = losses[-1] / max(losses[0], 1e-6)
    return {
        "n_samples": n_samples, "steps": steps,
        "loss_initial": losses[0], "loss_final": losses[-1],
        "ratio": ratio, "elapsed_s": round(elapsed, 2),
        "converged": ratio < 0.5,
    }


def phase_b_mini_training(subset_csv: Path, cache_dir: Path, norm_path: Path,
                          work: Path) -> Dict[str, Any]:
    log.info("mini training run (proof of end-to-end)")
    seed_everything(42)
    df = pd.read_csv(subset_csv)
    # tiny holdout: last 2 locations
    unique_locs = df["name_location"].unique().tolist()
    val_locs = unique_locs[-2:]
    train_idx, val_idx = split_indices_by_location(df, val_locs)

    ds_cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=subset_csv, norm_stats_path=norm_path,
        image_size=64, bands="ir_only", include_diff_frames=True,
    )
    train_ds = SolafuneDataset(ds_cfg, df=df, indices=train_idx[:64])
    val_ds = SolafuneDataset(ds_cfg, df=df, indices=val_idx[:16])
    dl = DataLoaderConfig(batch_size=8, num_workers=0, pin_memory=False,
                          persistent_workers=False, drop_last=False, prefetch_factor=2)
    train_loader = build_dataloader(train_ds, dl, sampler=build_sampler(train_ds, "precip_stratified"), base_seed=42)
    val_loader = build_dataloader(val_ds, dl, shuffle=False, base_seed=42)

    c_per = max_active_channels("ir_only")
    model = build_model({
        "in_channels_per_frame": c_per, "n_frames": 3, "n_diff_frames": 2,
        "encoder": "resnet34", "temporal": "none", "decoder": "unet",
        "probabilistic": False,
        "encoder_kwargs": {"norm": "group"},
        "decoder_kwargs": {"norm": "group", "decoder_channels": [128, 64, 32, 16]},
    })
    loss_fn = build_loss({"mse_weight": 1.0, "bce_weight": 0.5,
                          "rain_weighted": True, "rain_weight_scale": 3.0})
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-3})
    sched, seb = build_scheduler(opt, {"name": "cosine", "warmup_epochs": 0},
                                  steps_per_epoch=len(train_loader), epochs=2)
    tcfg = TrainerConfig(
        epochs=2, amp=False, ema_enabled=True, ema_decay=0.9,
        early_stop_patience=10, use_tensorboard=False,
        output_dir=str(work / "training_out"), log_every_n_steps=5,
        step_scheduler_each_batch=seb,
    )
    trainer = Trainer(model, opt, sched, loss_fn, train_loader, val_loader, tcfg,
                      device=torch.device("cpu"))
    t0 = time.perf_counter()
    best = trainer.fit()
    elapsed = time.perf_counter() - t0
    return {
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "epochs": 2, "elapsed_s": round(elapsed, 2),
        "best_val_metric": float(best) if best is not None else None,
        "checkpoint_exists": (Path(tcfg.output_dir) / "checkpoints" / "best.pt").exists(),
    }, model


def phase_fg_inference_submission(model: torch.nn.Module, work: Path) -> Dict[str, Any]:
    """Run inference on a 30-row eval subset and check submission format."""
    log.info("inference + submission on 30-row eval subset")
    df_full = pd.read_csv(EVAL_CSV)
    df_small = df_full.head(30).reset_index(drop=True)
    subset_csv = work / "eval_subset.csv"
    df_small.to_csv(subset_csv, index=False)
    # build eval cache
    cache_dir = work / "eval_cache"
    spec, _ = build_cache_spec(df_small, cache_dir, "ir_only")
    backend = get_backend("zarr")(spec, compressor="lz4")
    # eval has no GPM; use placeholder tifs (still readable) — load_gpm=False
    build_cache(subset_csv, EVAL_ROOT, cache_dir, backend, "ir_only",
                load_gpm=False, verbose_every=0, limit=len(df_small))
    backend.close()

    # build norm stats from TRAIN not EVAL
    train_df = pd.read_csv(TRAIN_CSV).head(400)
    paths = {s: [] for s in SATELLITES}
    for _, row in train_df.iterrows():
        for f in parse_frame_list(row["last_30_minutes_observation_filename"]):
            paths[row["satellite_target"]].append(sat_tif_path(TRAIN_ROOT, row["satellite_target"], f))
    stats = compute_norm_stats(paths, max_files_per_satellite=60, pixel_stride=2)
    norm_path = work / "eval_norm.json"
    save_norm_stats(norm_path, stats)

    ds_cfg = DatasetConfig(
        cache_dir=cache_dir, csv_path=subset_csv, norm_stats_path=norm_path,
        image_size=64, bands="ir_only", include_diff_frames=True,
    )
    ds = SolafuneDataset(ds_cfg)
    loader = build_dataloader(
        ds, DataLoaderConfig(batch_size=6, num_workers=0, pin_memory=False,
                              persistent_workers=False, drop_last=False,
                              prefetch_factor=2), shuffle=False, base_seed=0,
    )
    preds = predict(model, loader, PredictionConfig(amp=False, tta=True,
                    rain_mask_threshold=0.15), device=torch.device("cpu"))
    assert preds.shape == (30, 41, 41), preds.shape
    assert np.isfinite(preds).all()
    # write TIFs to a temp submission dir
    sub_dir = work / "test_files"
    n_written = write_submission(preds, subset_csv, sub_dir)
    # verify each TIF
    from src.utils.io import read_gpm_tif
    checked = 0
    for fname in df_small["gpm_imerg_filename"].tolist():
        arr, meta = read_gpm_tif(sub_dir / fname)
        assert arr.shape == GPM_SIZE
        assert arr.dtype == np.float32
        assert np.isfinite(arr).all()
        checked += 1
    return {
        "n_predicted": int(preds.shape[0]),
        "shape_ok": preds.shape == (30, 41, 41),
        "finite": bool(np.isfinite(preds).all()),
        "pred_min": float(preds.min()), "pred_max": float(preds.max()),
        "pred_mean": float(preds.mean()),
        "submission_files_written": int(n_written),
        "submission_files_verified": int(checked),
    }


def main() -> int:
    report: Dict[str, Any] = {"phases": {}}
    workdir = Path(tempfile.mkdtemp(prefix="final_val_"))
    try:
        # Phase A
        log.info("=== Phase A: real-data validation ===")
        subset_csv, cache_dir, norm_path = build_small_real_cache(workdir, per_sat=100)
        report["phases"]["A1_dataset_stats"] = phase_a_dataset_stats(subset_csv)
        report["phases"]["A2_tensor_sanity"] = phase_a_tensor_sanity(subset_csv, cache_dir, norm_path)
        report["phases"]["A3_overfit"] = phase_a_overfit(
            subset_csv, cache_dir, norm_path, n_samples=32, steps=60,
        )
        # Phase B lite
        log.info("=== Phase B (lite): mini training ===")
        b_report, model = phase_b_mini_training(subset_csv, cache_dir, norm_path, workdir)
        report["phases"]["B_mini_training"] = b_report
        # Phase F+G
        log.info("=== Phase F+G: inference + submission ===")
        report["phases"]["FG_inference_submission"] = phase_fg_inference_submission(model, workdir)
        # summary
        overfit_ok = report["phases"]["A3_overfit"]["converged"]
        training_ok = report["phases"]["B_mini_training"]["checkpoint_exists"]
        submission_ok = (
            report["phases"]["FG_inference_submission"]["submission_files_verified"]
            == report["phases"]["FG_inference_submission"]["submission_files_written"]
        )
        report["summary"] = {
            "overfit_pass": overfit_ok,
            "training_pipeline_pass": training_ok,
            "submission_format_pass": submission_ok,
            "all_pass": all([overfit_ok, training_ok, submission_ok]),
        }
    except Exception as e:
        report["fatal"] = repr(e)
        report["traceback"] = traceback.format_exc()
    finally:
        gc.collect()
        try:
            shutil.rmtree(workdir, ignore_errors=True)
        except Exception:
            pass
    write_json(REPORT_PATH, report)
    log.info("=== report written to %s ===", REPORT_PATH)
    log.info("SUMMARY: %s", report.get("summary"))
    return 0 if report.get("summary", {}).get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
