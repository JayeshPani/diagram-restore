"""DDIM reverse sampling. See docs/04_EXPERIMENT_PROTOCOL.md §4 for the schedule convention:
alpha_bar is indexed by timestep t, with t=0 nearly clean and the last index nearly pure noise.
"""

import torch


def _step_schedule(total: int, steps: int) -> list[int]:
    if not 1 <= steps <= total:
        raise ValueError("steps must lie between one and the number of trained timesteps")
    indices = torch.linspace(total - 1, 0, steps).round().long()
    seen = dict.fromkeys(indices.tolist())
    return list(seen)


@torch.inference_mode()
def ddim_sample(
    model,
    observation: torch.Tensor,
    alpha_bar: torch.Tensor,
    steps: int,
    eta: float = 0.0,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Deterministic (eta=0) reverse trajectory conditioned on a fixed damaged observation.

    `model(xt, conditioning, t)` must predict the diffusion noise epsilon, matching
    the training convention in models.diffusion_loss: both channels live in [-1, 1].
    """
    device = observation.device
    schedule = _step_schedule(len(alpha_bar), steps)
    x = torch.randn_like(observation) if noise is None else noise.to(device)
    conditioning = 2 * observation - 1
    for index, t in enumerate(schedule):
        t_batch = torch.full((observation.shape[0],), t, device=device, dtype=torch.long)
        alpha_t = alpha_bar[t]
        eps = model(x, conditioning, t_batch)
        x0 = (x - (1 - alpha_t).sqrt() * eps) / alpha_t.sqrt()
        x0 = x0.clamp(-1, 1)
        if index + 1 == len(schedule):
            x = x0
            continue
        alpha_prev = alpha_bar[schedule[index + 1]]
        sigma = eta * ((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)).sqrt()
        direction = (1 - alpha_prev - sigma**2).clamp_min(0).sqrt() * eps
        x = alpha_prev.sqrt() * x0 + direction
        if eta:
            x = x + sigma * torch.randn_like(x)
    return ((x + 1) / 2).clamp(0, 1)
