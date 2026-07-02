"""Build the on-disk cache for train and/or eval CSVs.

Usage::

    python scripts/01_build_cache.py --split train --bands ir_only \
        --csv D:/solafune/train_dataset_.../train_dataset.csv \
        --root D:/solafune/train_dataset_... \
        --out  D:/solafune/cache/train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.cache import get_backend  # noqa: E402
from src.data.preprocessing import build_cache, build_cache_spec, verify_cache  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.utils import read_json  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "eval"), required=True)
    parser.add_argument("--bands", default="ir_only", choices=("ir_only", "all", "visible_only"))
    parser.add_argument("--backend", default=None,
                        help="cache backend name; defaults to benchmark result if present")
    parser.add_argument("--backend-json", type=Path, default=_ROOT / "cache" / "backend.json")
    parser.add_argument("--zarr-compressor", default="lz4")
    parser.add_argument("--limit", type=int, default=None,
                        help="process only first N rows (debug)")
    args = parser.parse_args()

    log = get_logger("build_cache")

    backend_name = args.backend
    if backend_name is None:
        if args.backend_json.exists():
            try:
                backend_name = read_json(args.backend_json).get("recommended")
            except Exception:
                backend_name = None
    if backend_name is None:
        backend_name = "zarr"

    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if args.limit is not None:
        df = df.head(int(args.limit)).reset_index(drop=True)
    spec, _ = build_cache_spec(df, args.out, args.bands)

    backend_cls = get_backend(backend_name)
    if backend_name == "zarr":
        backend = backend_cls(spec, compressor=args.zarr_compressor)
    else:
        backend = backend_cls(spec)

    log.info(
        "building cache: split=%s bands=%s backend=%s rows=%d out=%s",
        args.split, args.bands, backend_name, spec.n_total, args.out,
    )
    build_cache(
        csv_path=args.csv,
        data_root=args.root,
        cache_root=args.out,
        backend=backend,
        band_mode=args.bands,
        load_gpm=(args.split == "train"),
        limit=args.limit,
    )
    verify_cache(backend)
    backend.close()
    log.info("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
