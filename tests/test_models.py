import torch

from diagram_restore.models import (
    ConditionalDiT,
    cosine_alpha_bar,
    diffusion_loss,
    restoration_loss,
    soft_cldice,
)


def test_soft_cldice_is_near_one_for_identical_thin_lines():
    image = torch.zeros(1, 1, 16, 16)
    image[:, :, 8, :] = 1.0
    assert soft_cldice(image, image).item() > 0.99


def test_soft_cldice_is_near_zero_for_disjoint_foregrounds():
    a = torch.zeros(1, 1, 16, 16)
    a[:, :, 2, :] = 1.0
    b = torch.zeros(1, 1, 16, 16)
    b[:, :, 13, :] = 1.0
    assert soft_cldice(a, b).item() < 0.05


def test_soft_cldice_keeps_the_batch_dimension():
    batch = torch.rand(4, 1, 16, 16)
    assert soft_cldice(batch, batch).shape == (4,)


def test_diffusion_loss_accepts_an_optional_structural_weight():
    torch.manual_seed(0)
    model = ConditionalDiT()
    clean = torch.rand(2, 1, 64, 64)
    observation = torch.rand(2, 1, 64, 64)
    alpha_bar = cosine_alpha_bar()
    torch.manual_seed(1)
    plain = diffusion_loss(model, clean, observation, alpha_bar)
    torch.manual_seed(1)
    structural = diffusion_loss(model, clean, observation, alpha_bar, structural_weight=0.1)
    assert torch.isfinite(structural)
    assert structural.item() != plain.item()


def test_restoration_loss_accepts_an_optional_structural_weight():
    logits = torch.randn(2, 1, 16, 16)
    clean = torch.rand(2, 1, 16, 16).round()
    plain = restoration_loss(logits, clean)
    structural = restoration_loss(logits, clean, structural_weight=0.1)
    assert torch.isfinite(structural)
    assert structural.item() != plain.item()
