import torch

from diagram_restore.baselines import (
    damaged_input_baseline,
    load_split,
    morphological_restore,
    run_baseline_suite,
    select_morphological_radius,
)
from diagram_restore.evaluation import extract_edges
from diagram_restore.fixtures import make_fixture


def _record(fixture) -> dict:
    return {"nodes": [n.to_dict() for n in fixture.nodes], "edges": fixture.reference}


def test_morphological_restore_at_radius_zero_matches_the_original_extraction():
    fixture = make_fixture("one_pixel_gap")
    restored = morphological_restore(fixture.image, radius=0)
    assert extract_edges(restored, fixture.nodes).edges == extract_edges(
        fixture.image, fixture.nodes
    ).edges


def test_morphological_restore_can_close_a_one_pixel_gap():
    fixture = make_fixture("one_pixel_gap")
    restored = morphological_restore(fixture.image, radius=1)
    assert extract_edges(restored, fixture.nodes).edges == [("A", "B")]


def test_select_morphological_radius_prefers_the_radius_that_recovers_the_edge():
    fixture = make_fixture("one_pixel_gap")
    damaged = torch.from_numpy(fixture.image)[None, None].float() / 255
    radius, metrics = select_morphological_radius(damaged, [_record(fixture)], radii=(0, 1, 2))
    assert radius >= 1
    assert metrics["edge_f1"] == 1.0


def test_damaged_input_baseline_matches_direct_extraction():
    fixture = make_fixture("one_pixel_gap")
    damaged = torch.from_numpy(fixture.image)[None, None].float() / 255
    summary = damaged_input_baseline(damaged, [_record(fixture)])
    assert summary["edge_f1"] == 0.0


def test_load_split_reads_the_requested_split(tiny_dataset):
    damaged, clean, records = load_split(tiny_dataset, "train")
    assert damaged.shape == clean.shape
    assert damaged.shape[0] == len(records) == 4
    assert damaged.dtype == torch.float32
    assert float(damaged.max()) <= 1.0 and float(damaged.min()) >= 0.0


def test_run_baseline_suite_covers_every_row_and_writes_a_report(tmp_path, tiny_dataset):
    output = tmp_path / "results"
    result = run_baseline_suite(
        tiny_dataset,
        output,
        unet_steps=2,
        dit_steps=2,
        sampling_steps=(2,),
        seed=0,
        morphological_radii=(0, 1),
    )
    assert output.joinpath("baselines.json").exists()
    assert set(result) >= {"B0", "B1", "U0", "U1", "T0", "D0", "D1"}
    for key in ("U0", "U1", "T0"):
        assert "edge_metrics" in result[key]
    for key in ("D0", "D1"):
        assert set(result[key]["sampling_steps"]) == {"2"}


def test_run_baseline_suite_measures_batch_one_latency_for_every_trained_row(
    tmp_path, tiny_dataset
):
    output = tmp_path / "results"
    result = run_baseline_suite(
        tiny_dataset,
        output,
        unet_steps=2,
        dit_steps=2,
        sampling_steps=(2,),
        seed=0,
        morphological_radii=(0, 1),
        latency_warmup=1,
        latency_repeats=2,
    )
    for key in ("B1", "U0", "U1", "T0"):
        assert result[key]["latency"]["repeats"] == 2
        assert result[key]["latency"]["median_seconds"] >= 0
    for key in ("D0", "D1"):
        entry = result[key]["sampling_steps"]["2"]
        assert "edge_metrics" in entry and "latency" in entry
        assert entry["latency"]["repeats"] == 2
    assert "latency" not in result["B0"]


def test_run_baseline_suite_reports_appearance_metrics_alongside_edge_metrics(
    tmp_path, tiny_dataset
):
    output = tmp_path / "results"
    result = run_baseline_suite(
        tiny_dataset,
        output,
        unet_steps=2,
        dit_steps=2,
        sampling_steps=(2,),
        seed=0,
        morphological_radii=(0, 1),
        latency_warmup=1,
        latency_repeats=2,
    )
    for key in ("B0", "B1", "U0", "U1", "T0"):
        assert "appearance_metrics" in result[key]
        assert 0 <= result[key]["appearance_metrics"]["mean_ssim"] <= 1
    for key in ("D0", "D1"):
        entry = result[key]["sampling_steps"]["2"]
        assert "appearance_metrics" in entry
