"""Benchmark cache backends and pick the fastest.

The benchmark:
  1. Builds a small (default 200-sample) synthetic dataset matching the shapes
     of one satellite (Himawari, 81x81, 10 IR bands, 3 frames, uint8).
  2. Writes it to each candidate backend under a temporary directory.
  3. Measures:
        * write throughput (samples/sec)
        * cold-start random read latency (single-process)
        * warm random read latency (single-process)
        * multi-process random read latency (2 workers)
        * disk footprint
  4. Chooses the backend with the lowest warm+multi-process read latency,
     preferring Zarr on ties (for compression).
  5. Writes the decision + full report to a JSON file.
"""
from __future__ import annotations

import multiprocessing as mp
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ...constants import DEFAULT_CACHE_BACKEND, FRAMES_PER_SAMPLE, HIMAWARI, SATELLITES
from ...logger import get_logger
from ...utils import format_bytes, write_json
from .base import CacheSpec
from .memmap_backend import MemmapBackend
from .zarr_backend import ZarrBackend

log = get_logger(__name__)


@dataclass
class BackendReport:
    name: str
    write_samples_per_sec: float
    cold_read_ms: float
    warm_read_ms: float
    mp_read_ms: float
    disk_bytes: int
    errors: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    recommended: str
    reports: List[BackendReport]

    def to_dict(self) -> Dict:
        return {
            "recommended": self.recommended,
            "reports": [asdict(r) for r in self.reports],
        }


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _make_spec(root: Path, n_samples: int, c: int, hw: int) -> CacheSpec:
    return CacheSpec(
        root=root,
        n_total=n_samples,
        per_sat_counts={s: (n_samples if s == HIMAWARI else 0) for s in SATELLITES},
        per_sat_shapes={
            HIMAWARI: (FRAMES_PER_SAMPLE, c, hw, hw),
            "goes": (FRAMES_PER_SAMPLE, c, 1, 1),
            "meteosat": (FRAMES_PER_SAMPLE, c, 1, 1),
        },
        dtype="uint8",
    )


def _make_random_samples(n: int, c: int, hw: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(n, FRAMES_PER_SAMPLE, c, hw, hw), dtype=np.uint8)


def _time_reads(backend, indices: List[int]) -> float:
    """Return mean read latency in milliseconds."""
    times = []
    for i in indices:
        t0 = time.perf_counter()
        _ = backend.read_sat_sample(HIMAWARI, i)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.mean(times)) if times else float("nan")


def _mp_worker(args):
    backend_name, spec_dict, indices = args
    spec = CacheSpec.from_dict(spec_dict)
    if backend_name == "zarr":
        backend = ZarrBackend(spec)
    else:
        backend = MemmapBackend(spec)
    lat = _time_reads(backend, indices)
    backend.close()
    return lat


def _bench_backend(
    backend_cls,
    name: str,
    root: Path,
    n_samples: int,
    c: int,
    hw: int,
    seed: int,
) -> BackendReport:
    errors: List[str] = []
    spec = _make_spec(root, n_samples, c, hw)
    if backend_cls is ZarrBackend:
        backend = ZarrBackend(spec)
    else:
        backend = MemmapBackend(spec)

    # write
    samples = _make_random_samples(n_samples, c, hw, seed=seed)
    backend.create()
    t0 = time.perf_counter()
    for i in range(n_samples):
        backend.write_sat_sample(HIMAWARI, i, samples[i])
    backend.flush()
    write_elapsed = time.perf_counter() - t0
    write_sps = n_samples / max(write_elapsed, 1e-9)

    # cold read (close handles + reopen)
    backend.close()
    if backend_cls is ZarrBackend:
        backend = ZarrBackend(spec)
    else:
        backend = MemmapBackend(spec)
    rng = np.random.default_rng(seed + 1)
    read_indices = rng.integers(0, n_samples, size=min(64, n_samples)).tolist()
    try:
        cold_ms = _time_reads(backend, read_indices[:8])
        warm_ms = _time_reads(backend, read_indices)
    except Exception as e:
        errors.append(f"read failed: {e!r}")
        cold_ms = float("nan")
        warm_ms = float("nan")
    backend.close()

    # multi-process read
    try:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=2) as pool:
            splits = [read_indices[::2], read_indices[1::2]]
            results = pool.map(
                _mp_worker,
                [(name, spec.to_dict(), s) for s in splits],
            )
        mp_ms = float(statistics.mean([r for r in results if not np.isnan(r)]))
    except Exception as e:  # pragma: no cover - varies per env
        errors.append(f"mp read failed: {e!r}")
        mp_ms = float("nan")

    disk = _dir_size(root)
    return BackendReport(
        name=name,
        write_samples_per_sec=write_sps,
        cold_read_ms=cold_ms,
        warm_read_ms=warm_ms,
        mp_read_ms=mp_ms,
        disk_bytes=disk,
        errors=errors,
    )


def run_benchmark(
    output_path: Path,
    n_samples: int = 200,
    channels: int = 10,
    hw: int = 81,
    seed: int = 0,
    tmp_root: Optional[Path] = None,
) -> BenchmarkResult:
    """Run the benchmark and persist the result to `output_path`."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = Path(tmp_root) if tmp_root else Path(tempfile.mkdtemp(prefix="solafune_bench_"))
    tmp.mkdir(parents=True, exist_ok=True)
    reports: List[BackendReport] = []
    try:
        for name, cls in (("zarr", ZarrBackend), ("memmap", MemmapBackend)):
            root = tmp / name
            root.mkdir(parents=True, exist_ok=True)
            try:
                r = _bench_backend(cls, name, root, n_samples, channels, hw, seed)
                log.info(
                    "%s: write=%.1f sps, cold=%.2fms, warm=%.2fms, mp=%.2fms, disk=%s",
                    name, r.write_samples_per_sec, r.cold_read_ms,
                    r.warm_read_ms, r.mp_read_ms, format_bytes(r.disk_bytes),
                )
            except Exception as e:  # pragma: no cover
                log.error("%s failed: %r", name, e)
                r = BackendReport(name=name, write_samples_per_sec=0.0,
                                  cold_read_ms=float("inf"),
                                  warm_read_ms=float("inf"),
                                  mp_read_ms=float("inf"),
                                  disk_bytes=0,
                                  errors=[repr(e)])
            reports.append(r)
    finally:
        try:
            shutil.rmtree(tmp, ignore_errors=True)
        except Exception:
            pass

    # decision: lowest warm+mp latency, prefer zarr on tie
    def score(r: BackendReport) -> float:
        warm = r.warm_read_ms if np.isfinite(r.warm_read_ms) else 1e9
        mp_ = r.mp_read_ms if np.isfinite(r.mp_read_ms) else 1e9
        return warm + mp_

    valid = [r for r in reports if not r.errors]
    if not valid:
        recommended = DEFAULT_CACHE_BACKEND
    else:
        ranked = sorted(valid, key=lambda r: (score(r), 0 if r.name == "zarr" else 1))
        recommended = ranked[0].name

    result = BenchmarkResult(recommended=recommended, reports=reports)
    write_json(output_path, result.to_dict())
    log.info("recommended backend: %s (written to %s)", recommended, output_path)
    return result
