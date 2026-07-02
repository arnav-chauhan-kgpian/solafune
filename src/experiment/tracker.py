"""Per-run experiment snapshot: config + git hash + environment fingerprint."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional


def _git_hash(repo_root: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return None


def _git_dirty(repo_root: Path) -> Optional[bool]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        return None
    return None


def snapshot_run(out_dir: Path, cfg: Mapping[str, Any],
                 repo_root: Optional[Path] = None) -> Path:
    """Write `run_snapshot.json` documenting config + environment."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        cuda_available = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
        torch_v = torch.__version__
    except Exception:
        cuda_available = False; cuda_name = None; torch_v = None

    try:
        # config may be OmegaConf; convert
        try:
            from omegaconf import OmegaConf  # type: ignore
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
        except Exception:
            cfg_dict = dict(cfg)
    except Exception:
        cfg_dict = dict(cfg) if hasattr(cfg, "items") else str(cfg)

    snap = {
        "config": cfg_dict,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_v,
        "cuda_available": cuda_available,
        "cuda_device": cuda_name,
        "git_hash": _git_hash(repo_root) if repo_root else None,
        "git_dirty": _git_dirty(repo_root) if repo_root else None,
    }
    p = out_dir / "run_snapshot.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)
    return p
