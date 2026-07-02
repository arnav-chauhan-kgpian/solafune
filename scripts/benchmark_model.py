"""Benchmark the model: parameter count, forward/backward latency, memory.

Usage::
    python scripts/benchmark_model.py --encoder resnet34 --image-size 96
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.constants import max_active_channels  # noqa: E402
from src.logger import get_logger  # noqa: E402
from src.models import build_model  # noqa: E402
from src.training.losses import build_loss  # noqa: E402

log = get_logger("bench")


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _flops_estimate(model: torch.nn.Module, sat: torch.Tensor, aux: torch.Tensor) -> int:
    """Naive convolution FLOPs estimate. Ignores non-conv ops which is
    conservative but sufficient for order-of-magnitude reporting."""
    total = 0
    def hook(module, inp, out):
        nonlocal total
        if isinstance(module, torch.nn.Conv2d):
            b = out.shape[0]; c_out = out.shape[1]; h = out.shape[2]; w = out.shape[3]
            k = module.kernel_size[0] * module.kernel_size[1]
            c_in = module.in_channels // module.groups
            total += b * c_out * h * w * c_in * k * 2
        elif isinstance(module, torch.nn.Linear):
            b = inp[0].shape[0]
            total += b * module.in_features * module.out_features * 2
    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        model(sat, aux)
    for h in handles:
        h.remove()
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="resnet34",
                        choices=["resnet34", "efficientnet_b3", "convnext_tiny"])
    parser.add_argument("--temporal", default="none",
                        choices=["none", "conv3d", "convlstm", "attention"])
    parser.add_argument("--decoder", default="unet", choices=["unet", "fpn"])
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--probabilistic", action="store_true")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    c_per = max_active_channels("ir_only")
    model = build_model({
        "in_channels_per_frame": c_per,
        "n_frames": 3, "n_diff_frames": 2,
        "encoder": args.encoder, "temporal": args.temporal, "decoder": args.decoder,
        "probabilistic": args.probabilistic,
    }).to(device)

    C_in = model.temporal.out_channels if args.temporal == "none" else None
    total_c_input = c_per * 5 if args.temporal == "none" else c_per * 5

    sat = torch.randn(args.batch_size, total_c_input, args.image_size, args.image_size,
                       device=device)
    aux = torch.randn(args.batch_size, 6, device=device)

    n_params = _count_params(model)
    flops = _flops_estimate(model, sat, aux)

    loss_fn = build_loss({"mse_weight": 1.0, "bce_weight": 0.5})

    # warm-up
    for _ in range(3):
        with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
            out = model(sat, aux)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    fwd_times = []
    for _ in range(args.iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
            out = model(sat, aux)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd_times.append((time.perf_counter() - t0) * 1000.0)

    # backward
    bwd_times = []
    dummy_batch = {
        "gpm_log1p": torch.zeros(args.batch_size, 41, 41, device=device),
        "gpm_raw": torch.zeros(args.batch_size, 41, 41, device=device),
        "rain_mask": torch.zeros(args.batch_size, 41, 41, device=device),
        "has_data": torch.ones(args.batch_size, device=device),
    }
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))
    for _ in range(args.iters):
        optim.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=(args.amp and device.type == "cuda")):
            out = model(sat, aux)
            losses = loss_fn(out, dummy_batch)
        scaler.scale(losses["total"]).backward()
        scaler.step(optim); scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        bwd_times.append((time.perf_counter() - t0) * 1000.0)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    log.info("=== Model Benchmark ===")
    log.info("encoder=%s temporal=%s decoder=%s image=%d batch=%d",
             args.encoder, args.temporal, args.decoder, args.image_size, args.batch_size)
    log.info("params: %d (%.2f M)", n_params, n_params / 1e6)
    log.info("FLOPs (conv+linear): %.2f G", flops / 1e9)
    log.info("forward: mean=%.2f p90=%.2f ms",
             statistics.mean(fwd_times), sorted(fwd_times)[int(0.9 * len(fwd_times))])
    log.info("backward+step: mean=%.2f p90=%.2f ms",
             statistics.mean(bwd_times), sorted(bwd_times)[int(0.9 * len(bwd_times))])
    log.info("throughput (fwd): %.1f samples/sec",
             args.batch_size / (statistics.mean(fwd_times) / 1000.0))
    log.info("throughput (fwd+bwd): %.1f samples/sec",
             args.batch_size / (statistics.mean(bwd_times) / 1000.0))
    log.info("peak VRAM: %.1f MB", peak_vram_mb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
