"""Compact dense baselines. No learned routing is introduced before the oracle gate."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.GroupNorm(4, out_channels),
        nn.SiLU(),
    )


class SmallUNet(nn.Module):
    """Three-scale U-Net; input grayscale [0,1], output ink logits."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.enc1 = conv_block(1, width)
        self.enc2 = conv_block(width, width * 2)
        self.bottom = conv_block(width * 2, width * 4)
        self.dec2 = conv_block(width * 6, width * 2)
        self.dec1 = conv_block(width * 3, width)
        self.head = nn.Conv2d(width, 1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(image)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        bottom = self.bottom(F.avg_pool2d(e2, 2))
        d2 = self.dec2(
            torch.cat([F.interpolate(bottom, size=e2.shape[-2:], mode="nearest"), e2], 1)
        )
        d1 = self.dec1(torch.cat([F.interpolate(d2, size=e1.shape[-2:], mode="nearest"), e1], 1))
        return self.head(d1)

    @torch.inference_mode()
    def restore(self, image: torch.Tensor) -> torch.Tensor:
        return 1.0 - self(image).sigmoid()


def sinusoidal(values: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequencies = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=values.device) / half
    )
    phase = values.float().reshape(-1, 1) * frequencies[None]
    return torch.cat([phase.cos(), phase.sin()], dim=-1)


class DiTBlock(nn.Module):
    def __init__(self, width: int = 128, heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, width * 6))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, tokens: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(time).chunk(6, -1)
        q = self.norm1(tokens) * (1 + scale1[:, None]) + shift1[:, None]
        tokens = tokens + gate1[:, None] * self.attention(q, q, q, need_weights=False)[0]
        q = self.norm2(tokens) * (1 + scale2[:, None]) + shift2[:, None]
        return tokens + gate2[:, None] * self.mlp(q)


class ConditionalDiT(nn.Module):
    def __init__(self, size: int = 64, patch: int = 4, width: int = 128, depth: int = 6):
        super().__init__()
        if size % patch or width % 4:
            raise ValueError("Size must divide into patches, width into four positional groups")
        self.size, self.patch, self.width = size, patch, width
        grid = size // patch
        yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
        pos = torch.cat(
            [sinusoidal(xx.flatten(), width // 2), sinusoidal(yy.flatten(), width // 2)], -1
        )
        self.register_buffer("position", pos[None])
        self.embedding = nn.Conv2d(2, width, kernel_size=patch, stride=patch)
        self.time = nn.Sequential(
            nn.Linear(width, width * 4), nn.SiLU(), nn.Linear(width * 4, width)
        )
        self.blocks = nn.ModuleList([DiTBlock(width) for _ in range(depth)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, patch * patch)

    def forward(
        self, noisy: torch.Tensor, observation: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        if noisy.shape != observation.shape or noisy.shape[-2:] != (self.size, self.size):
            raise ValueError("Noisy target and image conditioning must have the configured shape")
        time = self.time(sinusoidal(timestep, self.width))
        x = self.embedding(torch.cat([noisy, observation], dim=1)).flatten(2).transpose(1, 2)
        x = x + self.position
        for block in self.blocks:
            x = block(x, time)
        x = self.head(self.norm(x))
        b, grid, patch = noisy.shape[0], self.size // self.patch, self.patch
        return (
            x.reshape(b, grid, grid, patch, patch)
            .permute(0, 1, 3, 2, 4)
            .reshape(b, 1, self.size, self.size)
        )


def cosine_alpha_bar(steps: int = 1000) -> torch.Tensor:
    time = torch.arange(steps + 1, dtype=torch.float64) / steps
    cumulative = torch.cos((time + 0.008) / 1.008 * math.pi / 2).square()
    cumulative = cumulative / cumulative[0]
    beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(0, 0.999)
    return torch.cumprod(1 - beta, 0).float()


def restoration_loss(
    logits: torch.Tensor, clean: torch.Tensor, structural_weight: float = 0.0
) -> torch.Tensor:
    """BCE + L1 balances direct pixel classification with reconstruction.

    structural_weight adds a soft-clDice term (U1/T0 in docs/04_EXPERIMENT_PROTOCOL.md §7);
    it is a structural proxy, not a topology guarantee.
    """
    ink = 1 - clean
    loss = F.binary_cross_entropy_with_logits(logits, ink) + F.l1_loss(logits.sigmoid(), ink)
    if structural_weight:
        loss = loss + structural_weight * (1 - soft_cldice(logits.sigmoid(), ink)).mean()
    return loss


def diffusion_loss(
    model: ConditionalDiT,
    clean: torch.Tensor,
    observation: torch.Tensor,
    alpha_bar: torch.Tensor,
    structural_weight: float = 0.0,
) -> torch.Tensor:
    """structural_weight adds a soft-clDice term on the estimated x0 (D1 in the same table)."""
    ink_target = 1 - clean
    mapped_clean, mapped_observation = 2 * clean - 1, 2 * observation - 1
    t = torch.randint(len(alpha_bar), (len(clean),), device=clean.device)
    alpha = alpha_bar[t, None, None, None]
    noise = torch.randn_like(mapped_clean)
    xt = alpha.sqrt() * mapped_clean + (1 - alpha).sqrt() * noise
    prediction = model(xt, mapped_observation, t)
    x0 = (xt - (1 - alpha).sqrt() * prediction) / alpha.sqrt()
    gate = (alpha.flatten() >= 0.1).float()
    per_image_l1 = (x0 - mapped_clean).abs().mean((1, 2, 3))
    reconstruction = (per_image_l1 * gate).sum() / gate.sum().clamp_min(1)
    loss = F.mse_loss(prediction, noise) + 0.1 * reconstruction
    if structural_weight:
        pred_ink = ((1 - x0) / 2).clamp(0, 1)
        structural = 1 - soft_cldice(pred_ink, ink_target)
        loss = loss + structural_weight * (structural * gate).sum() / gate.sum().clamp_min(1)
    return loss


def soft_skeleton(image: torch.Tensor, iterations: int = 5) -> torch.Tensor:
    """Pooling-based structure proxy for backend smoke tests, not topology proof."""

    def erode(x):
        return torch.minimum(
            -F.max_pool2d(-x, (3, 1), 1, (1, 0)), -F.max_pool2d(-x, (1, 3), 1, (0, 1))
        )

    def opened(x):
        return F.max_pool2d(erode(x), 3, 1, 1)

    skel = F.relu(image - opened(image))
    for _ in range(iterations):
        image = erode(image)
        delta = F.relu(image - opened(image))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice(
    prediction: torch.Tensor, target: torch.Tensor, iterations: int = 5, eps: float = 1e-6
) -> torch.Tensor:
    """Per-image soft clDice between two foreground-probability maps in [0, 1].

    A structural proxy (Shit et al., docs/02_LITERATURE_REVIEW.md), not a topology guarantee.
    """
    pred_skeleton = soft_skeleton(prediction, iterations)
    target_skeleton = soft_skeleton(target, iterations)
    precision = (pred_skeleton * target).sum((1, 2, 3)) / (pred_skeleton.sum((1, 2, 3)) + eps)
    sensitivity = (target_skeleton * prediction).sum((1, 2, 3)) / (target_skeleton.sum((1, 2, 3)) + eps)
    return 2 * precision * sensitivity / (precision + sensitivity + eps)
