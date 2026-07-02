"""Benchmark cache backends and persist the recommendation.

Usage::

    python scripts/00_benchmark_cache.py --output cache/backend.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make src importable when run as a script.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.cache.benchmark import run_benchmark  # noqa: E402
from src.logger import get_logger  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=_ROOT / "cache" / "backend.json")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--channels", type=int, default=10)
    parser.add_argument("--hw", type=int, default=81)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log = get_logger("benchmark")
    log.info("running cache backend benchmark → %s", args.output)
    result = run_benchmark(
        output_path=args.output,
        n_samples=args.n_samples,
        channels=args.channels,
        hw=args.hw,
        seed=args.seed,
    )
    log.info("recommended: %s", result.recommended)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
