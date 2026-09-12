"""Tiny-set overfit diagnostic (docs/04_EXPERIMENT_PROTOCOL.md §4).

Mandatory before full baseline training. Never reported as a generalization result.
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .data import read_manifest
from .evaluation import evaluate_predictions
from .io import load_image, source_fingerprint, write_json
from .models import ConditionalDiT, SmallUNet, cosine_alpha_bar, diffusion_loss, restoration_loss
from .runtime import choose_device
from .sampling import ddim_sample


def _load_tiny_set(root: Path, count: int) -> tuple:
    records = read_manifest(root, "train")[:count]
    if len(records) < count:
        raise ValueError("Not enough training examples for the requested tiny-set size")
    damaged = torch.from_numpy(np.stack([load_image(root / r["damaged"]) for r in records]))
    clean = torch.from_numpy(np.stack([load_image(root / r["clean"]) for r in records]))
    return damaged[:, None].float() / 255, clean[:, None].float() / 255, records


def _edge_f1(restored: np.ndarray, records: list[dict]) -> dict:
    return evaluate_predictions(restored, records)


def run_training_steps(model: nn.Module, loss_fn, steps: int, lr: float = 1e-4) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss during tiny-set overfit")
        losses.append(float(loss.detach().cpu()))
    return losses


def overfit_tiny_set(
    root: Path,
    output: Path,
    count: int = 12,
    unet_steps: int = 3000,
    dit_steps: int = 10000,
    sampling_steps: int = 50,
    seed: int = 7,
) -> dict:
    device = choose_device()
    damaged, clean, records = _load_tiny_set(root, count)
    damaged, clean = damaged.to(device), clean.to(device)

    torch.manual_seed(seed)
    unet = SmallUNet().to(device).train()
    unet_losses = run_training_steps(unet, lambda: restoration_loss(unet(damaged), clean), unet_steps)
    unet.eval()
    with torch.inference_mode():
        unet_restored = unet.restore(damaged).cpu().numpy()[:, 0]
    unet_metrics = _edge_f1(unet_restored, records)

    torch.manual_seed(seed)
    dit = ConditionalDiT().to(device).train()
    alpha_bar = cosine_alpha_bar().to(device)
    dit_losses = run_training_steps(dit, lambda: diffusion_loss(dit, clean, damaged, alpha_bar), dit_steps)
    dit.eval()
    noise = torch.randn(damaged.shape, generator=torch.Generator().manual_seed(seed)).to(device)
    with torch.inference_mode():
        restored = ddim_sample(dit, damaged, alpha_bar, sampling_steps, noise=noise)
    dit_metrics = _edge_f1(restored.cpu().numpy()[:, 0], records)

    result = {
        "count": count,
        "seed": seed,
        "device": str(device),
        "unet": {
            "steps": unet_steps,
            "initial_loss": unet_losses[0],
            "final_loss": unet_losses[-1],
            "edge_metrics": unet_metrics,
        },
        "conditional_dit": {
            "steps": dit_steps,
            "sampling_steps": sampling_steps,
            "initial_loss": dit_losses[0],
            "final_loss": dit_losses[-1],
            "edge_metrics": dit_metrics,
        },
        "decision_rule": "direct-edge F1 >= 0.95 on this fixed tiny set before full pilot training",
        "unet_passed": unet_metrics["edge_f1"] >= 0.95,
        "conditional_dit_passed": dit_metrics["edge_f1"] >= 0.95,
        "note": "tiny-set overfit diagnostic only; never reported as generalization",
        "source_sha256": source_fingerprint(),
    }
    write_json(output / "tiny_overfit.json", result)
    return result
