"""Hand-specified evaluator counterexamples drawn independently of the generator."""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from PIL import Image, ImageDraw

from .geometry import Node


@dataclass
class Fixture:
    name: str
    nodes: list[Node]
    image: np.ndarray
    reference: list[tuple[str, str]]
    expected: list[tuple[str, str]]
    expected_merges: int = 0


CASES = (
    "intact",
    "one_pixel_gap",
    "removed",
    "invented_diagonal",
    "wrong_endpoint",
    "chain_not_triangle",
    "square",
    "square_cut",
    "disconnected",
    "near_miss",
    "zero_edge",
    "blank",
    "all_ink",
    "isolated_fragment",
    "crossed_diagonals",
    "merged_parallel",
    "diagonal_intact",
    "diagonal_gap",
    "boundary_offset",
    "two_port_node",
)


def make_fixture(case: str, shape: str = "box", width: int = 1) -> Fixture:
    if case not in CASES:
        raise ValueError(case)
    centers = {"A": (12, 12), "B": (50, 12), "C": (50, 50), "D": (12, 50)}
    nodes = [Node(k, shape, (x - 4, y - 4, x + 4, y + 4)) for k, (x, y) in centers.items()]
    image = Image.new("L", (64, 64), 255)
    draw = ImageDraw.Draw(image)
    reference = [("A", "B")]
    actual = [("A", "B")]
    expected = actual.copy()
    merges = 0
    if case in {"removed", "zero_edge", "blank", "boundary_offset"}:
        actual, expected = [], []
        if case in {"zero_edge", "boundary_offset"}:
            reference = []
    elif case == "invented_diagonal":
        actual = expected = [("A", "B"), ("A", "C")]
    elif case == "wrong_endpoint":
        actual = expected = [("A", "D")]
    elif case in {"chain_not_triangle", "two_port_node"}:
        reference = actual = expected = [("A", "B"), ("B", "C")]
    elif case in {"square", "square_cut"}:
        reference = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "D")]
        actual = expected = reference if case == "square" else reference[1:]
    elif case in {"disconnected", "merged_parallel"}:
        reference = actual = expected = [("A", "B"), ("C", "D")]
    elif case in {"diagonal_intact", "diagonal_gap"}:
        reference = actual = expected = [("A", "C")]
    elif case == "crossed_diagonals":
        reference = actual = [("A", "C"), ("B", "D")]
        expected, merges = list(combinations(centers, 2)), 1
    elif case == "all_ink":
        expected, merges = list(combinations(centers, 2)), 1
    for a, b in actual:
        draw.line([centers[a], centers[b]], fill=0, width=width)
    # Redraw the independently specified nodes last, clearing the line interiors.
    for node in nodes:
        painter = draw.rectangle if shape == "box" else draw.ellipse
        box = node.bbox
        if case == "boundary_offset":
            box = tuple(v + (1 if i in (0, 2) else 0) for i, v in enumerate(box))
        painter(box, fill=255, outline=0, width=1)
    if case == "one_pixel_gap":
        draw.rectangle((31, 9, 31, 15), fill=255)
        expected = []
    elif case == "near_miss":
        draw.rectangle((44, 9, 45, 15), fill=255)
        expected = []
    elif case == "diagonal_gap":
        draw.rectangle((31, 27, 31, 35), fill=255)
        expected = []
    elif case == "isolated_fragment":
        draw.line([(25, 40), (35, 40)], fill=0, width=width)
    elif case == "merged_parallel":
        draw.line([(31, 12), (31, 50)], fill=0, width=width)
        expected, merges = list(combinations(centers, 2)), 1
    elif case == "blank":
        draw.rectangle((0, 0, 63, 63), fill=255)
    elif case == "all_ink":
        draw.rectangle((0, 0, 63, 63), fill=0)
    return Fixture(
        f"{case}_{shape}_w{width}", nodes, np.asarray(image).copy(), reference, expected, merges
    )


def all_fixtures() -> list[Fixture]:
    return [
        make_fixture(case, shape, width)
        for case in CASES
        for shape in ("box", "circle")
        for width in (1, 2)
    ]
