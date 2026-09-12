from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage as ndi

from diagram_restore.data import corrupt, read_config, sample_parent
from diagram_restore.evaluation import extract_edges
from diagram_restore.geometry import NEIGHBORS

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
