"""Small general-purpose utilities.

Anything that doesn't naturally belong to a dedicated module lives here.
Keep this file small; if a helper grows, promote it to its own module.
"""
from __future__ import annotations

import ast
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Union

import numpy as np


PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# CSV frame-list parsing
# ---------------------------------------------------------------------------
def parse_frame_list(value: Any) -> List[str]:
    """Parse the `last_30_minutes_observation_filename` cell.

    The competition CSV stores this as a Python list literal in a string:
        "['train_aceh_Himawari_20221231_2330.tif', ...]"
    This function is defensive: it accepts lists, tuples, and strings; empty
    values return an empty list.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    if isinstance(value, float) and np.isnan(value):
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s or s in ("[]", "nan", "NaN"):
            return []
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Could not parse frame list: {value!r}") from e
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"Expected list, got {type(parsed).__name__}: {value!r}")
        return [str(x) for x in parsed]
    raise TypeError(f"Unsupported type for frame list: {type(value).__name__}")


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def write_json(path: PathLike, obj: Any, indent: int = 2) -> None:
    """Atomically write a JSON file (write to .tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default, sort_keys=True)
    tmp.replace(path)


def read_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@contextmanager
def timer(name: str, logger=None):
    """Context manager to measure wall-clock time."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[{name}] took {elapsed:.3f}s"
    if logger is not None:
        logger.info(msg)


def format_bytes(n: int) -> str:
    """Human-readable byte count."""
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


# ---------------------------------------------------------------------------
# Iteration helpers
# ---------------------------------------------------------------------------
def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    """Yield successive `size`-sized chunks from `iterable`."""
    if size <= 0:
        raise ValueError(f"chunk size must be positive, got {size}")
    buf: List = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Dict merges
# ---------------------------------------------------------------------------
def deep_merge(base: Mapping, override: Mapping) -> dict:
    """Recursively merge two dictionaries. `override` wins on conflicts."""
    result = dict(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], Mapping)
            and isinstance(value, Mapping)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result
