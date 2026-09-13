from diagram_restore.generalization import evaluate_generalization


def test_evaluate_generalization_covers_in_distribution_and_every_ood_set(
    tmp_path, tiny_dataset, tiny_ood_dataset
):
    output = tmp_path / "results"
    result = evaluate_generalization(
        tiny_dataset,
        output,
        {"ood_example": tiny_ood_dataset},
        unet_steps=2,
        dit_steps=2,
        sampling_steps=(2,),
        seed=0,
        latency_warmup=1,
        latency_repeats=2,
    )
    assert output.joinpath("generalization.json").exists()
    assert set(result["in_distribution"]) >= {"B0", "B1", "U0", "U1", "T0", "D0", "D1"}
    assert set(result["ood"]) == {"ood_example"}

    for group in (result["in_distribution"], result["ood"]["ood_example"]):
        assert set(group) >= {"B0", "B1", "U0", "U1", "T0", "D0", "D1"}
        for key in ("U0", "U1", "T0"):
            assert "edge_metrics" in group[key]
            assert "appearance_metrics" in group[key]
            assert "latency" in group[key]
        for key in ("D0", "D1"):
            assert set(group[key]["sampling_steps"]) == {"2"}
            entry = group[key]["sampling_steps"]["2"]
            assert "edge_metrics" in entry and "appearance_metrics" in entry and "latency" in entry
        assert "latency" not in group["B0"]


def test_evaluate_generalization_reuses_one_morphological_radius_everywhere(
    tmp_path, tiny_dataset, tiny_ood_dataset
):
    output = tmp_path / "results"
    result = evaluate_generalization(
        tiny_dataset,
        output,
        {"ood_example": tiny_ood_dataset},
        unet_steps=1,
        dit_steps=1,
        sampling_steps=(1,),
        seed=0,
        latency_warmup=1,
        latency_repeats=1,
        morphological_radii=(0, 1),
    )
    assert isinstance(result["morphological_radius"], int)
    assert "selected_radius" not in result["in_distribution"]["B1"]
