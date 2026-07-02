"""Matplotlib-based visualizations. All functions save to a file and return
the path. Import matplotlib lazily so the training path doesn't pay the
import cost when visualization is disabled.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_prediction_vs_target(
    pred: np.ndarray, target: np.ndarray, out_path: Path,
    title: Optional[str] = None, vmax: Optional[float] = None,
) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    vmax = vmax if vmax is not None else max(float(target.max()), float(pred.max()), 1.0)
    for ax, arr, name in zip(axes, [target, pred, pred - target], ["target", "pred", "pred-target"]):
        if name == "pred-target":
            m = np.max(np.abs(arr))
            im = ax.imshow(arr, vmin=-m, vmax=m, cmap="RdBu_r")
        else:
            im = ax.imshow(arr, vmin=0, vmax=vmax, cmap="Blues")
        ax.set_title(name)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_error_map(pred: np.ndarray, target: np.ndarray, out_path: Path) -> Path:
    plt = _plt()
    err = np.abs(pred - target)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(err, cmap="magma")
    ax.set_title("|pred - target|")
    ax.axis("off")
    fig.colorbar(im, ax=ax)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_training_curves(csv_path: Path, out_path: Path) -> Path:
    import pandas as pd
    plt = _plt()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    if "total" in df.columns:
        ax.plot(df.index, df["total"], label="total")
    for col in ("mse", "bce", "nll", "mae"):
        if col in df.columns:
            ax.plot(df.index, df[col], label=col, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("training loss")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_val_curves(csv_path: Path, out_path: Path,
                    metrics: Sequence[str] = ("rmse", "mae", "csi", "rain_f1")) -> Path:
    import pandas as pd
    plt = _plt()
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in metrics:
        if m in df.columns:
            for tag in df.get("tag", pd.Series(["raw"] * len(df))).unique():
                sub = df[df.get("tag", pd.Series(["raw"] * len(df))) == tag]
                if m in sub.columns:
                    ax.plot(sub.get("epoch", sub.index), sub[m],
                             label=f"{m}/{tag}", marker="o")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric")
    ax.legend()
    ax.set_title("validation metrics")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path
