"""Direct-edge extraction, with no access to reference edges during extraction."""

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import structural_similarity

from .geometry import NEIGHBORS, Node, node_regions


def normalized_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Expected a square grayscale image")
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
        raise ValueError("Float grayscale images must be finite in [0, 1]")
    return image.astype(np.float32)


@dataclass
class Extraction:
    edges: list[tuple[str, str]]
    components: int
    dangling_components: int
    isolated_components: int
    invalid_merges: list[list[str]]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_edges(image: np.ndarray, nodes: list[Node], threshold: float = 0.5) -> Extraction:
    """Infer all pairs connected outside nodes. Never closes gaps or uses true routes.

    Tracing the full binary mask, rather than a thinning that can move contacts,
    preserves the specified eight-neighbor raster connectivity for width 1 and 2.
    """
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    gray = normalized_image(image)
    fills, bands = node_regions(nodes, gray.shape[0])
    labels, count = ndi.label((gray < threshold) & ~fills, structure=NEIGHBORS)
    contacts = [set() for _ in range(count + 1)]
    for node, band in zip(nodes, bands, strict=True):
        for label in np.unique(labels[band]):
            if label:
                contacts[label].add(node.id)
    edges, merges = set(), []
    dangling = isolated = 0
    for contact in contacts[1:]:
        if len(contact) > 2:
            merges.append(sorted(contact))
        if len(contact) >= 2:
            edges.update(combinations(sorted(contact), 2))
        elif contact:
            dangling += 1
        else:
            isolated += 1
    return Extraction(sorted(edges), int(count), dangling, isolated, merges)


def edge_metrics(predicted: list, reference: list) -> dict:
    prediction = {tuple(sorted(edge)) for edge in predicted}
    truth = {tuple(sorted(edge)) for edge in reference}
    tp, fp, fn = len(prediction & truth), len(prediction - truth), len(truth - prediction)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "exact_graph": prediction == truth,
        "missing_edges": sorted(truth - prediction),
        "invented_edges": sorted(prediction - truth),
    }


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation")
    tp, fp, fn = (sum(row[key] for row in rows) for key in ("tp", "fp", "fn"))
    n = len(rows)
    return {
        "images": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "edge_precision": tp / (tp + fp) if tp + fp else None,
        "edge_recall": tp / (tp + fn) if tp + fn else None,
        "edge_f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
        "exact_graph_accuracy": sum(row["exact_graph"] for row in rows) / n,
        "missing_edges_per_image": fn / n,
        "invented_edges_per_image": fp / n,
        "macro_edge_f1": float(np.mean([row["f1"] for row in rows])),
        "invalid_merge_images": sum(bool(row.get("invalid_merges")) for row in rows),
    }


def image_metrics(image: np.ndarray, reference: np.ndarray, threshold: float = 0.5) -> dict:
    pred, truth = normalized_image(image), normalized_image(reference)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and reference sizes differ")
    p, t = pred < threshold, truth < threshold
    denom = int(p.sum() + t.sum())
    pb = p & ~ndi.binary_erosion(p, structure=NEIGHBORS)
    tb = t & ~ndi.binary_erosion(t, structure=NEIGHBORS)
    bp = np.count_nonzero(pb & ndi.binary_dilation(tb, structure=NEIGHBORS))
    br = np.count_nonzero(tb & ndi.binary_dilation(pb, structure=NEIGHBORS))
    precision = bp / pb.sum() if pb.any() else 0.0
    recall = br / tb.sum() if tb.any() else 0.0
    boundary = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mse = float(np.mean((pred - truth) ** 2))
    return {
        "foreground_dice": 2 * int((p & t).sum()) / denom if denom else 1.0,
        "boundary_f1_1px": float(boundary) if pb.any() or tb.any() else 1.0,
        "psnr_db": float(-10 * np.log10(mse)) if mse else None,
        "identical_pixels": mse == 0,
        "ssim": float(structural_similarity(pred, truth, data_range=1.0)),
    }
