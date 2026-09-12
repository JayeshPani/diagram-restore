"""Raster geometry shared by rendering and reference-node evaluation.

Only node geometry is shared with the evaluator, never connector annotations.
Coordinates are (x, y); arrays are indexed [y, x]; bounding boxes are inclusive.
"""

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

NEIGHBORS = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Node:
    id: str
    shape: str
    bbox: tuple[int, int, int, int]

    @property
    def center(self) -> tuple[int, int]:
        x0, y0, x1, y1 = self.bbox
        return (x0 + x1) // 2, (y0 + y1) // 2

    def to_dict(self) -> dict:
        return {**asdict(self), "center": list(self.center)}

    @classmethod
    def from_dict(cls, obj: dict) -> "Node":
        return cls(str(obj["id"]), obj["shape"], tuple(obj["bbox"]))


def node_mask(node: Node, size: int) -> np.ndarray:
    if node.shape not in {"box", "circle"}:
        raise ValueError(f"Unsupported node shape: {node.shape}")
    x0, y0, x1, y1 = node.bbox
    if not (0 <= x0 < x1 < size and 0 <= y0 < y1 < size):
        raise ValueError(f"Invalid node bounding box: {node.bbox}")
    image = Image.new("1", (size, size))
    painter = ImageDraw.Draw(image)
    (painter.rectangle if node.shape == "box" else painter.ellipse)(node.bbox, fill=1)
    return np.asarray(image, dtype=bool)


def node_regions(nodes: list[Node], size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    if len({node.id for node in nodes}) != len(nodes):
        raise ValueError("Node IDs must be unique")
    fills = [node_mask(node, size) for node in nodes]
    union = np.zeros((size, size), dtype=bool)
    for fill in fills:
        if (union & fill).any():
            raise ValueError("Overlapping node regions are unsupported")
        union |= fill
    bands = [ndi.binary_dilation(fill, structure=NEIGHBORS) & ~union for fill in fills]
    return union, bands


def polyline_mask(points: list[tuple[int, int]], size: int, width: int = 1) -> np.ndarray:
    image = Image.new("1", (size, size))
    ImageDraw.Draw(image).line(points, fill=1, width=width)
    return np.asarray(image, dtype=bool)


def render(nodes: list[Node], connectors: list[np.ndarray], size: int) -> np.ndarray:
    fills, _ = node_regions(nodes, size)
    outlines = np.zeros_like(fills)
    for node in nodes:
        fill = node_mask(node, size)
        outlines |= fill & ~ndi.binary_erosion(fill, structure=NEIGHBORS)
    ink = np.logical_or.reduce(connectors) if connectors else np.zeros_like(fills)
    ink = (ink & ~fills) | outlines
    return np.where(ink, 0, 255).astype(np.uint8)
