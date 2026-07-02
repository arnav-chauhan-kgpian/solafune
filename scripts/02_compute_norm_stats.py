"""Compute per-satellite per-band normalization statistics from the train TIFs.

Usage::

    python scripts/02_compute_norm_stats.py \
        --csv D:/solafune/train_dataset_.../train_dataset.csv \
        --root D:/solafune/train_dataset_... \
        --out D:/solafune/cache/norm_stats.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.constants import SATELLITES  # noqa: E402
from src.data.normalization import compute_norm_stats, save_norm_stats  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.paths import sat_tif_path  # noqa: E402
from src.utils import parse_frame_list  # noqa: E402


def _collect_tif_paths(df: pd.DataFrame, root: Path):
    paths_by_sat = {s: [] for s in SATELLITES}
    for _, row in df.iterrows():
        sat = str(row["satellite_target"]).lower().strip()
        if sat not in SATELLITES:
            continue
        frames = parse_frame_list(row["last_30_minutes_observation_filename"])
        for fname in frames:
            paths_by_sat[sat].append(sat_tif_path(root, sat, fname))
    return paths_by_sat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = get_logger("norm_stats")
    df = pd.read_csv(args.csv)
    paths = _collect_tif_paths(df, args.root)
    for s in SATELLITES:
        log.info("%s: %d candidate TIFs", s, len(paths[s]))

    stats = compute_norm_stats(
        paths,
        max_files_per_satellite=args.max_files,
        pixel_stride=args.pixel_stride,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_norm_stats(args.out, stats)
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
