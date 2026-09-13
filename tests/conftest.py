from pathlib import Path

import pytest

from diagram_restore.data import generate

_TINY_CONFIG = """
version = "test-tiny"
image_size = 64
seed = 1
train = 4
validation = 2
test = 1
min_nodes = 2
max_nodes = 3
line_widths = [1]
noise_sigma_max = 0.02
blur_sigma_max = 0.3
max_gap_length = 2
max_removed_fraction = 0.2

[corruption_weights]
clean = 0.25
noise = 0.25
blur = 0.25
gap = 0.25
"""


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> Path:
    config_path = tmp_path / "pilot.toml"
    config_path.write_text(_TINY_CONFIG)
    output = tmp_path / "data"
    generate(config_path, output)
    return output


_TINY_OOD_CONFIG = """
version = "test-ood"
image_size = 64
seed = 99
train = 1
validation = 1
test = 4
min_nodes = 2
max_nodes = 3
line_widths = [1]
noise_sigma_max = 0.05
blur_sigma_max = 0.5
max_gap_length = 2
max_removed_fraction = 0.2

[corruption_weights]
clean = 0.0
noise = 0.5
blur = 0.0
gap = 0.5
"""


@pytest.fixture
def tiny_ood_dataset(tmp_path: Path) -> Path:
    config_path = tmp_path / "ood.toml"
    config_path.write_text(_TINY_OOD_CONFIG)
    output = tmp_path / "ood_data"
    generate(config_path, output)
    return output
