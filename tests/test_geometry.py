from pathlib import Path

import numpy as np

from diagram_restore.data import read_config, sample_parent
from diagram_restore.evaluation import extract_edges
from diagram_restore.geometry import render_antialiased

CONFIG = read_config(Path(__file__).parents[1] / "configs/pilot.toml")


def test_render_antialiased_produces_genuine_intermediate_gray_values():
    parent = sample_parent(1, 2, CONFIG)
    image = render_antialiased(parent["nodes"], parent["edges"], CONFIG["image_size"])
    unique = np.unique(image)
    assert image.shape == (CONFIG["image_size"], CONFIG["image_size"])
    assert image.dtype == np.uint8
    assert ((unique > 0) & (unique < 255)).any(), "expected antialiased (non-binary) pixels"


def test_render_antialiased_recovers_the_same_direct_edges_as_the_binary_renderer():
    for seed in range(10):
        parent = sample_parent(seed, seed + 500, CONFIG)
        expected = {tuple(e["nodes"]) for e in parent["edges"]}
        binary_edges = set(extract_edges(parent["clean"], parent["nodes"]).edges)
        antialiased = render_antialiased(parent["nodes"], parent["edges"], CONFIG["image_size"])
        antialiased_edges = set(extract_edges(antialiased, parent["nodes"]).edges)
        assert binary_edges == expected
        assert antialiased_edges == expected


def test_render_antialiased_differs_pixelwise_from_the_binary_renderer():
    parent = sample_parent(3, 4, CONFIG)
    antialiased = render_antialiased(parent["nodes"], parent["edges"], CONFIG["image_size"])
    assert not np.array_equal(antialiased, parent["clean"])
