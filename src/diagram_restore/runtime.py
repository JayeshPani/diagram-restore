"""Local runtime inventory and bounded, synchronized training benchmarks."""

import importlib.metadata
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn

from .data import read_manifest
from .io import git_revision, load_image, source_fingerprint, write_json
from .models import ConditionalDiT, SmallUNet, cosine_alpha_bar, diffusion_loss, restoration_loss


def choose_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def memory(device: torch.device) -> dict:
    result = {"process_rss_bytes": psutil.Process().memory_info().rss}
    if device.type == "mps":
        result.update(
            {
                "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
                "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
            }
        )
    return result


def environment() -> dict:
    def command(args):
        p = subprocess.run(args, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "memory_bytes": psutil.virtual_memory().total,
        "chip": command(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "os_version": command(["sw_vers"]),
        "device": str(choose_device()),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback_environment": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "Pillow", "scipy", "scikit-image", "psutil", "pytest")
        },
        "git_revision": git_revision(),
        "source_sha256": source_fingerprint(),
        "power_source": command(["pmset", "-g", "batt"]),
        "training_backend_detected_only": True,
    }


def load_batch(root: Path, count: int, split: str = "train") -> tuple:
    if split != "train":
        raise ValueError("Benchmark and training diagnostics use training parents only")
    records = read_manifest(root, split)[:count]
    if len(records) < count:
        raise ValueError("Not enough training examples")
    damaged = torch.from_numpy(np.stack([load_image(root / r["damaged"]) for r in records]))
    clean = torch.from_numpy(np.stack([load_image(root / r["clean"]) for r in records]))
    return damaged[:, None].float() / 255, clean[:, None].float() / 255, records


def benchmark_training(root: Path, output: Path, steps: int = 200, warmup: int = 20) -> dict:
    if steps < 1 or warmup < 0:
        raise ValueError("Invalid benchmark step counts")
    torch.manual_seed(7)
    torch.set_num_threads(4)
    device = choose_device()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "environment.json", environment())
    loading_start = time.perf_counter()
    observed, target, _ = load_batch(root, 8)
    loading_seconds = time.perf_counter() - loading_start
    observed, target = observed.to(device), target.to(device)
    alpha = cosine_alpha_bar().to(device)
    rows = []
    for name, constructor in (("unet", SmallUNet), ("conditional_dit", ConditionalDiT)):
        for batch_size in (4, 8):
            torch.manual_seed(7)
            model = constructor().to(device).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            times, losses = [], []
            maxima = memory(device)
            for step in range(warmup + steps):
                sync(device)
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                x, y = observed[:batch_size], target[:batch_size]
                if name == "unet":
                    loss = restoration_loss(model(x), y)
                else:
                    loss = diffusion_loss(model, y, x, alpha)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                sync(device)
                elapsed = time.perf_counter() - start
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss in {name} benchmark")
                if step >= warmup:
                    times.append(elapsed)
                    losses.append(float(loss.detach().cpu()))
                for key, value in memory(device).items():
                    maxima[key] = max(maxima.get(key, 0), value)
            row = {
                "model": name,
                "batch_size": batch_size,
                "parameters": sum(p.numel() for p in model.parameters()),
                "warmup_steps": warmup,
                "measured_steps": steps,
                "median_seconds_per_step": float(np.median(times)),
                "p95_seconds_per_step": float(np.quantile(times, 0.95)),
                "mean_seconds_per_step": float(np.mean(times)),
                "initial_measured_loss": losses[0],
                "final_measured_loss": losses[-1],
                "memory_sampled_high_water": maxima,
                "estimated_20000_step_hours_excluding_validation": float(
                    np.mean(times) * 20000 / 3600
                ),
            }
            rows.append(row)
            print(
                f"{name} batch={batch_size}: {row['median_seconds_per_step']:.4f} s/step",
                flush=True,
            )
            del model, optimizer, loss
            if device.type == "mps":
                torch.mps.empty_cache()
    result = {
        "device": str(device),
        "dtype": "float32",
        "seed": 7,
        "cpu_threads": 4,
        "data_loading_seconds_8_images": loading_seconds,
        "rows": rows,
        "memory_note": "sampled maxima; RSS and MPS are not additive",
        "purpose": "bounded timing runs on eight training images, not generalization experiments",
        "source_sha256": source_fingerprint(),
    }
    write_json(output / "benchmark_pilot.json", result)
    return result


def benchmark_sparse_mlp(output: Path, repeats: int = 200) -> dict:
    """Untrained operation microbenchmark; no learned router or speedup claim."""
    device = choose_device()
    torch.manual_seed(7)
    layer = nn.Sequential(nn.Linear(128, 512), nn.GELU(), nn.Linear(512, 128)).to(device).eval()
    tokens = torch.randn(8, 256, 128, device=device)
    rows = []
    with torch.inference_mode():
        for keep in (1.0, 0.75, 0.5):
            k = int(256 * keep)

            # Scores are a dummy feature statistic, not an implemented research router.
            def sparse():
                index = tokens.square().mean(-1).topk(k, dim=1).indices
                gathered = tokens.gather(1, index[..., None].expand(-1, -1, 128))
                update = layer(gathered)
                return torch.zeros_like(tokens).scatter(
                    1, index[..., None].expand(-1, -1, 128), update
                )

            index = tokens.square().mean(-1).topk(k, dim=1).indices
            mask = torch.zeros(8, 256, 1, device=device).scatter(1, index[..., None], 1)
            reference = layer(tokens) * mask
            actual = sparse()
            error = float((reference - actual).abs().max().cpu())
            if not torch.allclose(reference, actual, atol=1e-5, rtol=1e-4):
                raise RuntimeError("Sparse MLP does not match dense masked reference")
            timing = {}
            for label, fn in (("dense", lambda: layer(tokens)), ("sparse_with_overhead", sparse)):
                for _ in range(20):
                    fn()
                samples = []
                for _ in range(repeats):
                    sync(device)
                    start = time.perf_counter()
                    fn()
                    sync(device)
                    samples.append(time.perf_counter() - start)
                timing[label] = float(np.median(samples))
            rows.append(
                {
                    "keep_fraction": keep,
                    "mlp_tokens_executed": 8 * k,
                    "dense_tokens": 2048,
                    "max_absolute_parity_error": error,
                    "median_seconds": timing,
                }
            )
    result = {
        "device": str(device),
        "shape": [8, 256, 128],
        "repeats": repeats,
        "rows": rows,
        "note": "isolated MLP only; dummy score + topk + gather/scatter included; "
        "not full-restoration latency or evidence for the routing hypothesis",
    }
    write_json(output / "benchmark_sparse_mlp.json", result)
    return result
