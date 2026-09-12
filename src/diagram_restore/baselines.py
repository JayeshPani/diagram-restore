"""Stage 6 baseline suite: B0/B1/U0/U1/T0/D0/D1 (docs/04_EXPERIMENT_PROTOCOL.md §7).

No routing is introduced here. G0/G1/E0/F0/C0/C1 require the Stage 7 oracle and the
Stage 8 learned router; reuse this module's trained U0/U1/T0/D0/D1 checkpoints there
rather than retraining duplicate variants under different names.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

from .data import read_manifest
from .evaluation import evaluate_predictions, normalized_image
from .geometry import NEIGHBORS
from .io import load_image, source_fingerprint, write_json
from .models import (
    ConditionalDiT,
    SmallUNet,
    cosine_alpha_bar,
    diffusion_loss,
    restoration_loss,
)
from .runtime import choose_device
from .sampling import ddim_sample
from .train import run_training_steps


def load_split(root: Path, split: str) -> tuple:
    records = read_manifest(root, split)
    damaged = torch.from_numpy(np.stack([load_image(root / r["damaged"]) for r in records]))
    clean = torch.from_numpy(np.stack([load_image(root / r["clean"]) for r in records]))
    return damaged[:, None].float() / 255, clean[:, None].float() / 255, records


def damaged_input_baseline(damaged: torch.Tensor, records: list[dict]) -> dict:
    """B0: evaluator behavior on unrestored damaged images."""
    return evaluate_predictions(damaged.numpy()[:, 0], records)


def morphological_restore(image: np.ndarray, radius: int, threshold: float = 0.5) -> np.ndarray:
    """B1: binary closing on the thresholded ink mask; radius=0 is the identity."""
    ink = normalized_image(image) < threshold
    if radius:
        ink = ndi.binary_closing(ink, structure=NEIGHBORS, iterations=radius)
    return np.where(ink, 0.0, 1.0).astype(np.float32)


def morphological_baseline(
    damaged: torch.Tensor, records: list[dict], radius: int, threshold: float = 0.5
) -> dict:
    restored = np.stack(
        [morphological_restore(image, radius, threshold) for image in damaged.numpy()[:, 0]]
    )
    return evaluate_predictions(restored, records)


def select_morphological_radius(
    damaged: torch.Tensor, records: list[dict], radii: tuple = (0, 1, 2, 3)
) -> tuple[int, dict]:
    scored = [(radius, morphological_baseline(damaged, records, radius)) for radius in radii]
    return max(scored, key=lambda item: item[1]["edge_f1"])


def _sample_batch(damaged: torch.Tensor, clean: torch.Tensor, batch_size: int) -> tuple:
    index = torch.randint(len(damaged), (min(batch_size, len(damaged)),), device=damaged.device)
    return damaged[index], clean[index]


def train_unet(
    train_damaged: torch.Tensor,
    train_clean: torch.Tensor,
    steps: int,
    seed: int,
    structural_weight: float = 0.0,
) -> tuple[SmallUNet, list[float]]:
    """U0 (structural_weight=0) / U1 (structural_weight>0)."""
    torch.manual_seed(seed)
    model = SmallUNet().to(train_damaged.device).train()

    def loss_fn():
        x, y = _sample_batch(train_damaged, train_clean, 8)
        return restoration_loss(model(x), y, structural_weight)

    return model, run_training_steps(model, loss_fn, steps)


def train_deterministic_transformer(
    train_damaged: torch.Tensor,
    train_clean: torch.Tensor,
    steps: int,
    seed: int,
    structural_weight: float = 0.1,
) -> tuple[ConditionalDiT, list[float]]:
    """T0: same backbone as the diffusion model, a zero noise channel, fixed time, one pass."""
    torch.manual_seed(seed)
    model = ConditionalDiT().to(train_damaged.device).train()

    def loss_fn():
        x, y = _sample_batch(train_damaged, train_clean, 8)
        timestep = torch.zeros(len(x), dtype=torch.long, device=x.device)
        logits = model(torch.zeros_like(x), 2 * x - 1, timestep)
        return restoration_loss(logits, y, structural_weight)

    return model, run_training_steps(model, loss_fn, steps)


def train_conditional_dit(
    train_damaged: torch.Tensor,
    train_clean: torch.Tensor,
    steps: int,
    seed: int,
    structural_weight: float = 0.0,
) -> tuple[ConditionalDiT, list[float]]:
    """D0 (structural_weight=0) / D1 (structural_weight>0)."""
    torch.manual_seed(seed)
    model = ConditionalDiT().to(train_damaged.device).train()
    alpha_bar = cosine_alpha_bar().to(train_damaged.device)

    def loss_fn():
        x, y = _sample_batch(train_damaged, train_clean, 8)
        return diffusion_loss(model, y, x, alpha_bar, structural_weight)

    return model, run_training_steps(model, loss_fn, steps)


@torch.inference_mode()
def restore_unet(model: SmallUNet, damaged: torch.Tensor) -> np.ndarray:
    model.eval()
    return model.restore(damaged).cpu().numpy()[:, 0]


@torch.inference_mode()
def restore_deterministic_transformer(model: ConditionalDiT, damaged: torch.Tensor) -> np.ndarray:
    model.eval()
    timestep = torch.zeros(len(damaged), dtype=torch.long, device=damaged.device)
    logits = model(torch.zeros_like(damaged), 2 * damaged - 1, timestep)
    return (1 - logits.sigmoid()).cpu().numpy()[:, 0]


def restore_conditional_dit(
    model: ConditionalDiT,
    damaged: torch.Tensor,
    alpha_bar: torch.Tensor,
    steps: int,
    seed: int,
) -> np.ndarray:
    model.eval()
    noise = torch.randn(damaged.shape, generator=torch.Generator().manual_seed(seed))
    restored = ddim_sample(model, damaged, alpha_bar, steps, noise=noise.to(damaged.device))
    return restored.cpu().numpy()[:, 0]


def run_baseline_suite(
    root: Path,
    output: Path,
    unet_steps: int = 6000,
    dit_steps: int = 10000,
    sampling_steps: tuple = (10, 20, 50),
    seed: int = 7,
    morphological_radii: tuple = (0, 1, 2, 3),
    structural_weight: float = 0.1,
) -> dict:
    device = choose_device()
    train_damaged, train_clean, _ = load_split(root, "train")
    train_damaged, train_clean = train_damaged.to(device), train_clean.to(device)
    val_damaged, val_clean, val_records = load_split(root, "validation")
    val_damaged, val_clean = val_damaged.to(device), val_clean.to(device)

    result: dict = {"B0": damaged_input_baseline(val_damaged.cpu(), val_records)}

    radius, morphological_metrics = select_morphological_radius(
        val_damaged.cpu(), val_records, morphological_radii
    )
    result["B1"] = {"selected_radius": radius, "edge_metrics": morphological_metrics}

    unet0, unet0_losses = train_unet(train_damaged, train_clean, unet_steps, seed, 0.0)
    result["U0"] = {
        "steps": unet_steps,
        "final_loss": unet0_losses[-1],
        "edge_metrics": evaluate_predictions(restore_unet(unet0, val_damaged), val_records),
    }

    unet1, unet1_losses = train_unet(
        train_damaged, train_clean, unet_steps, seed, structural_weight
    )
    result["U1"] = {
        "steps": unet_steps,
        "structural_weight": structural_weight,
        "final_loss": unet1_losses[-1],
        "edge_metrics": evaluate_predictions(restore_unet(unet1, val_damaged), val_records),
    }

    transformer, transformer_losses = train_deterministic_transformer(
        train_damaged, train_clean, unet_steps, seed, structural_weight
    )
    result["T0"] = {
        "steps": unet_steps,
        "structural_weight": structural_weight,
        "final_loss": transformer_losses[-1],
        "edge_metrics": evaluate_predictions(
            restore_deterministic_transformer(transformer, val_damaged), val_records
        ),
    }

    alpha_bar = cosine_alpha_bar().to(device)
    for row_id, weight in (("D0", 0.0), ("D1", structural_weight)):
        model, losses = train_conditional_dit(
            train_damaged, train_clean, dit_steps, seed, weight
        )
        row = {
            "steps": dit_steps,
            "structural_weight": weight,
            "final_loss": losses[-1],
            "sampling_steps": {},
        }
        for steps in sampling_steps:
            restored = restore_conditional_dit(model, val_damaged, alpha_bar, steps, seed)
            row["sampling_steps"][str(steps)] = evaluate_predictions(restored, val_records)
        result[row_id] = row

    result["source_sha256"] = source_fingerprint()
    write_json(output / "baselines.json", result)
    return result
