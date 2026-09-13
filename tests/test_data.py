from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage as ndi

from diagram_restore.data import corrupt, generate, read_config, read_manifest, sample_parent
from diagram_restore.evaluation import extract_edges
from diagram_restore.geometry import NEIGHBORS, Node

CONFIG = read_config(Path(__file__).parents[1] / "configs/pilot.toml")


@pytest.mark.parametrize("seed", range(40))
def test_parent_graph_and_corruption(seed):
    parent = sample_parent(seed, seed + 1000, CONFIG)
    expected = {tuple(e["nodes"]) for e in parent["edges"]}
    assert set(extract_edges(parent["clean"], parent["nodes"]).edges) == expected
    damaged, gap, info = corrupt(parent, "gap", seed + 2000, CONFIG)
    assert info["gap"]["removed_fraction"] <= 0.2
    assert gap.any()
    expected.remove(tuple(info["gap"]["edge"]))
    gap_only = parent["clean"].copy()
    gap_only[gap] = 255
    assert set(extract_edges(gap_only, parent["nodes"]).edges) == expected
    # Subsequent blur can legitimately remove additional thin raster connections.
    assert tuple(info["gap"]["edge"]) not in extract_edges(damaged, parent["nodes"]).edges
    for mask in parent["masks"]:
        assert ndi.label(mask, structure=NEIGHBORS)[1] == 1


def test_parent_and_damage_repeat_exactly():
    a = sample_parent(7, 17, CONFIG)
    b = sample_parent(7, 17, CONFIG)
    assert np.array_equal(a["clean"], b["clean"])
    assert a["nodes"] == b["nodes"] and a["edges"] == b["edges"]
    for kind in CONFIG["corruption_weights"]:
        x, mask_x, metadata_x = corrupt(a, kind, 27, CONFIG)
        y, mask_y, metadata_y = corrupt(b, kind, 27, CONFIG)
        assert np.array_equal(x, y) and np.array_equal(mask_x, mask_y)
        assert metadata_x == metadata_y


def test_read_config_rejects_an_unknown_renderer(tmp_path):
    config_path = tmp_path / "bad.toml"
    config_path.write_text(
        Path("configs/pilot.toml").read_text() + '\nrenderer = "hand-drawn"\n'
    )
    with pytest.raises(ValueError):
        read_config(config_path)


def test_generate_with_the_antialiased_renderer_still_recovers_exact_graphs(tmp_path):
    config_path = tmp_path / "aa.toml"
    config_path.write_text(
        """
version = "test-aa"
image_size = 64
seed = 5
train = 6
validation = 2
test = 2
min_nodes = 2
max_nodes = 4
line_widths = [1, 2]
noise_sigma_max = 0.02
blur_sigma_max = 0.3
max_gap_length = 2
max_removed_fraction = 0.2
renderer = "antialiased"

[corruption_weights]
clean = 0.5
noise = 0.5
blur = 0.0
gap = 0.0
"""
    )
    output = tmp_path / "data"
    generate(config_path, output)
    records = read_manifest(output)
    assert len(records) == 10
    saw_gray = False
    for record in records:
        clean = np.asarray(Image.open(output / record["clean"]))
        nodes = [Node.from_dict(n) for n in record["nodes"]]
        expected = {tuple(e) for e in record["edges"]}
        assert set(extract_edges(clean, nodes).edges) == expected
        saw_gray = saw_gray or bool(((clean > 0) & (clean < 255)).any())
    assert saw_gray, "expected at least one genuinely antialiased (non-binary) pixel"
