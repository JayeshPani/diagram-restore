import pytest
import torch

from diagram_restore.sampling import ddim_sample


def linear_alpha_bar(steps: int = 50) -> torch.Tensor:
    return torch.linspace(0.999, 0.02, steps, dtype=torch.float32)


@pytest.mark.parametrize("steps", [0, -1, 51])
def test_rejects_step_counts_outside_the_trained_schedule(steps):
    with pytest.raises(ValueError):
        ddim_sample(lambda x, o, t: torch.zeros_like(x), torch.zeros(1, 1, 4, 4), linear_alpha_bar(), steps)


def test_output_has_observation_shape_and_lies_in_unit_range():
    observation = torch.rand(2, 1, 8, 8)
    restored = ddim_sample(
        lambda x, o, t: torch.zeros_like(x), observation, linear_alpha_bar(), steps=10
    )
    assert restored.shape == observation.shape
    assert restored.min() >= 0 and restored.max() <= 1


def test_deterministic_given_the_same_initial_noise():
    observation = torch.rand(2, 1, 8, 8)
    noise = torch.randn(2, 1, 8, 8)
    def model(x, o, t):
        return (x + o).tanh()
    first = ddim_sample(model, observation, linear_alpha_bar(), steps=10, noise=noise)
    second = ddim_sample(model, observation, linear_alpha_bar(), steps=10, noise=noise)
    assert torch.equal(first, second)


def test_recovers_the_exact_signal_from_a_model_with_perfect_noise_prediction():
    """A model whose epsilon exactly inverts x0_true must return x0_true at every
    step, since the DDIM x0 estimate is then algebraically independent of xt."""
    alpha_bar = linear_alpha_bar()
    x0_true = torch.rand(3, 1, 8, 8) * 2 - 1

    def perfect_model(xt, observation, t):
        alpha_t = alpha_bar[t].view(-1, 1, 1, 1)
        return (xt - alpha_t.sqrt() * x0_true) / (1 - alpha_t).sqrt()

    for steps in (1, 10, 50):
        restored = ddim_sample(perfect_model, torch.zeros(3, 1, 8, 8), alpha_bar, steps=steps)
        expected = ((x0_true + 1) / 2).clamp(0, 1)
        assert torch.allclose(restored, expected, atol=1e-4)
