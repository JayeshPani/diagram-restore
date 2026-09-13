"""Stage 9: train once on the in-distribution split, evaluate against distribution shifts
the models never saw (docs/04_EXPERIMENT_PROTOCOL.md §9, docs/03_PROJECT_PLAN.md Stage 9).
"""

from pathlib import Path

import numpy as np

from .baselines import (
    load_split,
    morphological_restore,
    restore_conditional_dit,
    restore_deterministic_transformer,
    restore_unet,
    select_morphological_radius,
    train_conditional_dit,
    train_deterministic_transformer,
    train_unet,
)
from .evaluation import aggregate_image_metrics, evaluate_predictions
from .io import source_fingerprint, write_json
from .latency import measure_latency
from .models import cosine_alpha_bar
from .runtime import choose_device


def evaluate_generalization(
    train_root: Path,
    output: Path,
    ood_roots: dict,
    unet_steps: int = 6000,
    dit_steps: int = 30000,
    sampling_steps: tuple = (10, 20, 50),
    seed: int = 7,
    structural_weight: float = 0.1,
    morphological_radii: tuple = (0, 1, 2, 3),
    latency_warmup: int = 20,
    latency_repeats: int = 50,
) -> dict:
    """Trains U0/U1/T0/D0/D1 once on train_root's train split, then evaluates every row
    against train_root's own validation split (in-distribution) and each named OOD root's
    test split, without retraining or re-tuning anything on the shifted distributions."""
    device = choose_device()
    train_damaged, train_clean, _ = load_split(train_root, "train")
    train_damaged, train_clean = train_damaged.to(device), train_clean.to(device)

    unet0, _ = train_unet(train_damaged, train_clean, unet_steps, seed, 0.0)
    unet1, _ = train_unet(train_damaged, train_clean, unet_steps, seed, structural_weight)
    transformer, _ = train_deterministic_transformer(
        train_damaged, train_clean, unet_steps, seed, structural_weight
    )
    alpha_bar = cosine_alpha_bar().to(device)
    dit0, _ = train_conditional_dit(train_damaged, train_clean, dit_steps, seed, 0.0)
    dit1, _ = train_conditional_dit(train_damaged, train_clean, dit_steps, seed, structural_weight)

    val_damaged, _, val_records = load_split(train_root, "validation")
    radius, _ = select_morphological_radius(val_damaged, val_records, morphological_radii)

    def evaluate_on(root: Path, split: str) -> dict:
        damaged, clean, records = load_split(root, split)
        damaged, clean = damaged.to(device), clean.to(device)
        clean_np = clean.cpu().numpy()[:, 0]
        single = damaged[:1]

        def timed(fn) -> dict:
            return measure_latency(fn, latency_warmup, latency_repeats, device)

        def scored(restored: np.ndarray) -> dict:
            return {
                "edge_metrics": evaluate_predictions(restored, records),
                "appearance_metrics": aggregate_image_metrics(restored, clean_np),
            }

        morphological_restored = np.stack(
            [morphological_restore(image, radius) for image in damaged.cpu().numpy()[:, 0]]
        )
        row: dict = {
            "images": len(records),
            "B0": scored(damaged.cpu().numpy()[:, 0]),
            "B1": scored(morphological_restored),
            "U0": {**scored(restore_unet(unet0, damaged)), "latency": timed(lambda: restore_unet(unet0, single))},
            "U1": {**scored(restore_unet(unet1, damaged)), "latency": timed(lambda: restore_unet(unet1, single))},
            "T0": {
                **scored(restore_deterministic_transformer(transformer, damaged)),
                "latency": timed(lambda: restore_deterministic_transformer(transformer, single)),
            },
        }
        for row_id, model in (("D0", dit0), ("D1", dit1)):
            steps_row = {}
            for steps in sampling_steps:
                restored = restore_conditional_dit(model, damaged, alpha_bar, steps, seed)
                steps_row[str(steps)] = {
                    **scored(restored),
                    "latency": timed(
                        lambda steps=steps: restore_conditional_dit(
                            model, single, alpha_bar, steps, seed
                        )
                    ),
                }
            row[row_id] = {"sampling_steps": steps_row}
        return row

    result: dict = {
        "seed": seed,
        "morphological_radius": radius,
        "in_distribution": evaluate_on(train_root, "validation"),
        "ood": {name: evaluate_on(root, "test") for name, root in ood_roots.items()},
        "source_sha256": source_fingerprint(),
    }
    write_json(output / "generalization.json", result)
    return result
