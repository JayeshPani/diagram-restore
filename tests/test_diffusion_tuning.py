import math

import pytest
import torch
from torch import nn

from diagram_restore.diffusion_tuning import (
    EMA,
    evaluate_diffusion_tuning,
    train_conditional_dit_tuned,
    warmup_cosine_lr,
)


def test_ema_shadow_starts_equal_to_the_initial_weights():
    model = nn.Linear(2, 2, bias=False)
    ema = EMA(model, decay=0.9)
    assert torch.equal(ema.shadow["weight"], model.weight.detach())


def test_ema_update_moves_toward_new_weights_by_one_minus_decay():
    model = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = EMA(model, decay=0.9)
    with torch.no_grad():
        model.weight.fill_(2.0)
    ema.update(model)
    expected = 0.9 * 1.0 + 0.1 * 2.0
    assert ema.shadow["weight"].item() == pytest.approx(expected)


def test_copy_to_loads_the_shadow_weights_into_a_model():
    source = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        source.weight.fill_(3.0)
    ema = EMA(source, decay=0.9)
    target = nn.Linear(1, 1, bias=False)
    ema.copy_to(target)
    assert target.weight.item() == pytest.approx(3.0)


def test_warmup_cosine_lr_ramps_linearly_to_the_base_rate():
    assert warmup_cosine_lr(0, total_steps=100, warmup_steps=10, base_lr=1.0) == pytest.approx(0.0)
    assert warmup_cosine_lr(5, total_steps=100, warmup_steps=10, base_lr=1.0) == pytest.approx(0.5)
    assert warmup_cosine_lr(10, total_steps=100, warmup_steps=10, base_lr=1.0) == pytest.approx(1.0)


def test_warmup_cosine_lr_decays_to_min_lr_at_the_final_step():
    lr = warmup_cosine_lr(99, total_steps=100, warmup_steps=10, base_lr=1.0, min_lr=0.1)
    assert lr == pytest.approx(0.1, abs=1e-2)


def test_warmup_cosine_lr_is_monotonic_within_each_phase():
    total, warmup = 50, 10
    values = [warmup_cosine_lr(s, total, warmup, base_lr=1.0) for s in range(total)]
    assert all(a <= b for a, b in zip(values[:warmup], values[1 : warmup + 1], strict=False))
    assert all(a >= b for a, b in zip(values[warmup:], values[warmup + 1 :], strict=False))


def test_warmup_cosine_lr_rejects_zero_or_negative_total_steps():
    with pytest.raises(ValueError):
        warmup_cosine_lr(0, total_steps=0, warmup_steps=0, base_lr=1.0)


def test_train_conditional_dit_tuned_runs_and_returns_an_eval_ready_model(tiny_dataset):
    from diagram_restore.baselines import load_split

    damaged, clean, _ = load_split(tiny_dataset, "train")
    model, losses = train_conditional_dit_tuned(
        damaged, clean, steps=3, seed=0, warmup_steps=1, ema_decay=0.9
    )
    assert len(losses) == 3
    assert all(math.isfinite(loss) for loss in losses)
    assert not model.training


def test_evaluate_diffusion_tuning_writes_a_report_with_every_sampling_step(
    tmp_path, tiny_dataset
):
    output = tmp_path / "results"
    result = evaluate_diffusion_tuning(
        tiny_dataset,
        output,
        steps=3,
        sampling_steps=(2,),
        seed=0,
        warmup_steps=1,
        latency_warmup=1,
        latency_repeats=2,
    )
    assert output.joinpath("diffusion_tuning.json").exists()
    assert result["steps"] == 3
    entry = result["sampling_steps"]["2"]
    assert "edge_metrics" in entry and "latency" in entry
