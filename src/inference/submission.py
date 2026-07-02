"""Write per-sample GPM prediction TIFs."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ..constants import COL_GPM, GPM_SIZE
from ..utils.io import write_gpm_tif


def write_submission(
    predictions: np.ndarray,
    eval_csv: Path,
    test_files_dir: Path,
) -> int:
    """Overwrite `test_files_dir/<gpm_filename>` with the predicted arrays.

    Returns the number of files written.
    """
    df = pd.read_csv(eval_csv)
    if len(predictions) != len(df):
        raise ValueError(f"predictions len {len(predictions)} != rows {len(df)}")
    if predictions.ndim != 3 or predictions.shape[1:] != GPM_SIZE:
        raise ValueError(f"predictions shape must be (N, 41, 41), got {predictions.shape}")
    test_files_dir = Path(test_files_dir)
    test_files_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for pred, fname in zip(predictions, df[COL_GPM].tolist()):
        write_gpm_tif(test_files_dir / str(fname), pred.astype(np.float32))
        n += 1
    return n
