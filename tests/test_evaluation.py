import inspect
from itertools import combinations

import numpy as np
import pytest

from diagram_restore.evaluation import (
    edge_metrics,
    evaluate_predictions,
    extract_edges,
    image_metrics,
    summarize,
)
from diagram_restore.fixtures import all_fixtures, make_fixture


@pytest.mark.parametrize("fixture", all_fixtures(), ids=lambda f: f.name)
def test_hand_specified_graphs(fixture):
    result = extract_edges(fixture.image, fixture.nodes)
    assert set(result.edges) == {tuple(sorted(edge)) for edge in fixture.expected}
    assert len(result.invalid_merges) == fixture.expected_merges
    measured = edge_metrics(result.edges, fixture.reference)
    independent_expected = edge_metrics(fixture.expected, fixture.reference)
    assert measured == independent_expected


def test_no_privileged_route_inputs():
    assert list(inspect.signature(extract_edges).parameters) == ["image", "nodes", "threshold"]
    fixture = make_fixture("invented_diagonal")
    first = extract_edges(fixture.image, fixture.nodes)
    fixture.reference = list(combinations(["A", "B", "C", "D"], 2))
    assert extract_edges(fixture.image, fixture.nodes) == first


def test_metrics_count_direct_edges_and_zero_denominators():
    assert edge_metrics([], [])["f1"] == 1
    assert edge_metrics([], [("A", "B")])["f1"] == 0
    assert edge_metrics([("A", "B")], [])["precision"] == 0
    mixed = edge_metrics([("A", "B"), ("A", "C")], [("A", "B"), ("B", "C")])
    assert (mixed["tp"], mixed["fp"], mixed["fn"]) == (1, 1, 1)
    assert summarize([mixed])["edge_f1"] == 0.5


def test_image_scores_do_not_replace_graph_scores():
    intact = make_fixture("intact")
    broken = make_fixture("one_pixel_gap")
    assert image_metrics(broken.image, intact.image)["foreground_dice"] > 0.99
    assert (
        edge_metrics(extract_edges(broken.image, broken.nodes).edges, intact.reference)["f1"] == 0
    )


@pytest.mark.parametrize(
    "bad", [np.full((64, 64), np.nan), np.ones((32, 64)), np.ones((64, 64)) * 255]
)
def test_invalid_float_images_rejected(bad):
    with pytest.raises(ValueError):
        extract_edges(bad, make_fixture("intact").nodes)


def test_evaluate_predictions_summarizes_direct_edges_per_record():
    intact, gap = make_fixture("intact"), make_fixture("one_pixel_gap")
    records = [
        {"nodes": [n.to_dict() for n in intact.nodes], "edges": intact.reference},
        {"nodes": [n.to_dict() for n in gap.nodes], "edges": gap.reference},
    ]
    summary = evaluate_predictions(np.stack([intact.image, gap.image]), records)
    assert (summary["tp"], summary["fn"]) == (1, 1)
    assert summary["edge_f1"] == pytest.approx(2 / 3)
