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


def render_antialiased(
    nodes: list[Node], edges: list[dict], size: int, supersample: int = 4
) -> np.ndarray:
    """A distinct rasterization technique for renderer-shift tests: draws the same
    abstract geometry (edge polylines, node bounding boxes) at `supersample`x resolution,
    then box-downsamples, producing genuinely antialiased gray edges instead of `render`'s
    exact binary fills. Takes raw geometry rather than precomputed masks, unlike `render`,
    since supersampling must happen before rasterization, not after.
    """
    big = size * supersample
    offset = supersample / 2
    image = Image.new("L", (big, big), 255)
    draw = ImageDraw.Draw(image)
    for edge in edges:
        # A 1px-wide diagonal stroke covers well under half of each traversed output
        # pixel on average, so a literal width*supersample scaling can dip below the
        # evaluator's 0.5 threshold and break the line; pad the stroke enough to stay
        # solid along any diagonal while still leaving a genuinely antialiased fringe.
        width = edge["width"] * supersample + supersample // 2
        # Box-downsampling averages input blocks [i*supersample, (i+1)*supersample); a
        # stroke centered on the raw pixel coordinate straddles two blocks and comes out
        # as faint gray in both instead of solid in one, so center it on the block instead.
        points = [(x * supersample + offset, y * supersample + offset) for x, y in edge["polyline"]]
        draw.line(points, fill=0, width=width)
        # PIL's line joints are beveled, not mitered; a bevel gap at a bend can survive
        # supersampling as a real break, so plug every interior vertex with a filled disc.
        radius = width / 2
        for x, y in points[1:-1]:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=0)
    for node in nodes:
        box = tuple(v * supersample for v in node.bbox)
        painter = draw.rectangle if node.shape == "box" else draw.ellipse
        painter(box, fill=255, outline=0, width=supersample)
    return np.asarray(image.resize((size, size), Image.Resampling.BOX), dtype=np.uint8)
