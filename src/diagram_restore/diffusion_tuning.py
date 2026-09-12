"""Standard diffusion-training fixes to rule out a fixable defect before concluding that
dense diffusion is architecturally behind the pilot U-Net (results/milestone2/G2_REPORT.md).
"""

import math
from pathlib import Path

import torch
from torch import nn

from .baselines import load_split, restore_conditional_dit
from .evaluation import evaluate_predictions
from .io import source_fingerprint, write_json
from .latency import measure_latency
from .models import ConditionalDiT, cosine_alpha_bar, diffusion_loss
from .runtime import choose_device


class EMA:
    """Exponential moving average of a model's floating-point parameters and buffers."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if value.dtype.is_floating_point:
                self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[key] = value.detach().clone()

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)


def warmup_cosine_lr(
    step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float = 0.0
) -> float:
    if total_steps <= 0 or warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("total_steps must be positive and warmup_steps must fit within it")
    if step < warmup_steps:
        return base_lr * (step / warmup_steps) if warmup_steps else base_lr
    span = total_steps - warmup_steps
    progress = (step - warmup_steps) / span if span else 1.0
    decay = (1 + math.cos(math.pi * progress)) / 2
    return min_lr + decay * (base_lr - min_lr)


def _sample_batch(damaged: torch.Tensor, clean: torch.Tensor, batch_size: int) -> tuple:
    index = torch.randint(len(damaged), (min(batch_size, len(damaged)),), device=damaged.device)
    return damaged[index], clean[index]


def train_conditional_dit_tuned(
    train_damaged: torch.Tensor,
    train_clean: torch.Tensor,
    steps: int,
    seed: int,
    structural_weight: float = 0.0,
    base_lr: float = 1e-4,
    warmup_steps: int = 500,
    ema_decay: float = 0.999,
) -> tuple[ConditionalDiT, list[float]]:
    """Trains with linear warmup + cosine LR decay and returns the EMA-averaged weights."""
    torch.manual_seed(seed)
    device = train_damaged.device
    model = ConditionalDiT().to(device).train()
    alpha_bar = cosine_alpha_bar().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    ema = EMA(model, ema_decay)
    losses = []
    for step in range(steps):
        lr = warmup_cosine_lr(step, steps, min(warmup_steps, steps), base_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        x, y = _sample_batch(train_damaged, train_clean, 8)
        loss = diffusion_loss(model, y, x, alpha_bar, structural_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        ema.update(model)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss during tuned diffusion training")
        losses.append(float(loss.detach().cpu()))
    ema.copy_to(model)
    model.eval()
    return model, losses


def evaluate_diffusion_tuning(
    root: Path,
    output: Path,
    steps: int = 30000,
    sampling_steps: tuple = (10, 20, 50),
    seed: int = 7,
    structural_weight: float = 0.0,
    base_lr: float = 1e-4,
    warmup_steps: int = 1500,
    ema_decay: float = 0.999,
    latency_warmup: int = 20,
    latency_repeats: int = 50,
) -> dict:
    """Trains one EMA + warmup-cosine diffusion model and reports it against the same
    validation split and sampling sweep as diagram_restore.baselines.run_baseline_suite,
    so its "D0" row is directly comparable to results/milestone2/baselines.json."""
    device = choose_device()
    train_damaged, train_clean, _ = load_split(root, "train")
    train_damaged, train_clean = train_damaged.to(device), train_clean.to(device)
    val_damaged, _, val_records = load_split(root, "validation")
    val_damaged = val_damaged.to(device)
    single = val_damaged[:1]

    model, losses = train_conditional_dit_tuned(
        train_damaged, train_clean, steps, seed, structural_weight, base_lr, warmup_steps, ema_decay
    )
    alpha_bar = cosine_alpha_bar().to(device)
    result: dict = {
        "steps": steps,
        "base_lr": base_lr,
        "warmup_steps": warmup_steps,
        "ema_decay": ema_decay,
        "structural_weight": structural_weight,
        "final_training_loss": losses[-1],
        "sampling_steps": {},
    }
    for sample_step in sampling_steps:
        restored = restore_conditional_dit(model, val_damaged, alpha_bar, sample_step, seed)
        result["sampling_steps"][str(sample_step)] = {
            "edge_metrics": evaluate_predictions(restored, val_records),
            "latency": measure_latency(
                lambda sample_step=sample_step: restore_conditional_dit(
                    model, single, alpha_bar, sample_step, seed
                ),
                latency_warmup,
                latency_repeats,
                device,
            ),
        }
    result["source_sha256"] = source_fingerprint()
    write_json(output / "diffusion_tuning.json", result)
    return result
