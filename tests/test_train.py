import numpy as np
import pytest

from diagram_restore.fixtures import make_fixture
from diagram_restore.train import _edge_f1, overfit_tiny_set


def test_edge_f1_matches_direct_evaluation_of_each_image():
    fixture_a = make_fixture("intact")
    fixture_b = make_fixture("one_pixel_gap")
    records = [
        {"nodes": [n.to_dict() for n in fixture_a.nodes], "edges": fixture_a.reference},
        {"nodes": [n.to_dict() for n in fixture_b.nodes], "edges": fixture_b.reference},
    ]
    summary = _edge_f1(np.stack([fixture_a.image, fixture_b.image]), records)
    assert (summary["tp"], summary["fn"]) == (1, 1)
    assert summary["edge_f1"] == pytest.approx(2 / 3)


def test_overfit_tiny_set_runs_and_reports_the_expected_fields(tmp_path, tiny_dataset):
    output = tmp_path / "results"
    result = overfit_tiny_set(
        tiny_dataset, output, count=4, unet_steps=3, dit_steps=3, sampling_steps=2, seed=0
    )
    assert output.joinpath("tiny_overfit.json").exists()
    assert result["count"] == 4
    for key in ("unet", "conditional_dit"):
        assert result[key]["steps"] == 3
        assert "edge_metrics" in result[key]
        assert result[key]["initial_loss"] != result[key]["final_loss"]
    assert set(result) >= {"unet_passed", "conditional_dit_passed", "decision_rule"}
