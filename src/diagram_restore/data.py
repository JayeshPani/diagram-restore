"""Deterministic pilot generator. Split clean parents before assigning corruption."""

import json
import math
import tomllib
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .geometry import NEIGHBORS, Node, node_mask, node_regions, polyline_mask, render
from .io import hash_bytes, hash_json, save_mask, source_fingerprint, write_json

SPLITS = ("train", "validation", "test")


def read_config(path: Path) -> dict:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config["image_size"] != 64:
        raise ValueError("pilot-v1 geometry is defined for 64px; version larger layouts separately")
    if not 2 <= config["min_nodes"] <= config["max_nodes"] <= 5:
        raise ValueError("pilot-v1 supports 2–5 nodes")
    if any(config[split] <= 0 for split in SPLITS):
        raise ValueError("Every split must contain diagrams")
    weights = config["corruption_weights"]
    if set(weights) != {"clean", "noise", "blur", "gap"}:
        raise ValueError("Expected clean, noise, blur and gap strata")
    if any(v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("Corruption weights must be nonnegative and sum to one")
    if not 0 < config["max_removed_fraction"] <= 0.2:
        raise ValueError("Pilot erasure fraction must be in (0, 0.2]")
    return config


def _candidate_paths(a: Node, b: Node, rng: np.random.Generator) -> list:
    x0, y0 = a.center
    x1, y1 = b.center
    paths = [[(x0, y0), (x1, y1)], [(x0, y0), (x1, y0), (x1, y1)], [(x0, y0), (x0, y1), (x1, y1)]]
    rng.shuffle(paths)
    return paths


def sample_parent(seed: int, renderer_seed: int, config: dict) -> dict:
    """Geometric rejection, independent of the scored edge extraction algorithm."""
    rng = np.random.default_rng(seed)
    renderer_rng = np.random.default_rng(renderer_seed)
    size = config["image_size"]
    for attempt in range(200):
        n = int(rng.integers(config["min_nodes"], config["max_nodes"] + 1))
        sites = [(x, y) for y in (11, 32, 53) for x in (11, 32, 53)]
        indices = rng.choice(len(sites), n, replace=False)
        nodes = []
        for i, index in enumerate(indices):
            cx, cy = sites[index]
            cx, cy = cx + int(rng.integers(-2, 3)), cy + int(rng.integers(-2, 3))
            half = int(renderer_rng.integers(4, 7))
            shape = str(renderer_rng.choice(["box", "circle"]))
            nodes.append(Node(chr(65 + i), shape, (cx - half, cy - half, cx + half, cy + half)))
        fills, bands = node_regions(nodes, size)
        width = int(renderer_rng.choice(config["line_widths"]))
        occupied = np.zeros((size, size), dtype=bool)
        edges, masks = [], []
        pairs = list(combinations(range(n), 2))
        rng.shuffle(pairs)
        desired = int(rng.integers(1, min(len(pairs), n + 1) + 1))
        for a, b in pairs:
            for points in _candidate_paths(nodes[a], nodes[b], rng):
                raw = polyline_mask(points, size, width)
                # A connector cannot enter or closely approach any unrelated shape.
                forbidden = np.zeros_like(fills)
                for j, node in enumerate(nodes):
                    if j not in (a, b):
                        forbidden |= ndi.binary_dilation(
                            node_mask(node, size), structure=NEIGHBORS, iterations=3
                        )
                connector = raw & ~fills
                if (raw & forbidden).any() or not connector.any():
                    continue
                if (
                    connector & ndi.binary_dilation(occupied, structure=NEIGHBORS, iterations=2)
                ).any():
                    continue
                labels, count = ndi.label(connector, structure=NEIGHBORS)
                if (
                    count != 1
                    or not (connector & bands[a]).any()
                    or not (connector & bands[b]).any()
                ):
                    continue
                # Each endpoint has one compact contact, with sufficient non-node interior.
                interior = connector & ~ndi.binary_dilation(
                    fills, structure=NEIGHBORS, iterations=3
                )
                if interior.sum() < 4:
                    continue
                edges.append(
                    {
                        "nodes": sorted([nodes[a].id, nodes[b].id]),
                        "polyline": [list(p) for p in points],
                        "width": width,
                    }
                )
                masks.append(connector)
                occupied |= connector
                break
            if len(edges) == desired:
                break
        if edges:
            clean = render(nodes, masks, size)
            return {
                "nodes": nodes,
                "edges": edges,
                "masks": masks,
                "clean": clean,
                "graph_seed": seed,
                "renderer_seed": renderer_seed,
                "geometry_attempt": attempt,
            }
    raise RuntimeError(f"Could not construct a valid parent for seed {seed}")


def _make_gap(parent: dict, rng: np.random.Generator, config: dict) -> tuple[np.ndarray, dict]:
    size = config["image_size"]
    fills, _ = node_regions(parent["nodes"], size)
    for edge_index in rng.permutation(len(parent["masks"])):
        mask = parent["masks"][edge_index]
        edge = parent["edges"][edge_index]
        skel = skeletonize(mask)
        candidates = np.argwhere(
            skel & ~ndi.binary_dilation(fills, structure=NEIGHBORS, iterations=3)
        )
        rng.shuffle(candidates)
        max_len = min(config["max_gap_length"], max(1, int(skel.sum() * 0.2)))
        lengths = list(range(1, max_len + 1))
        rng.shuffle(lengths)
        for y, x in candidates:
            # Perpendicular strip removes the full stroke thickness, including width-two ink.
            near = np.argwhere(skel[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4])
            axis = 1 if np.ptp(near[:, 1]) >= np.ptp(near[:, 0]) else 0
            coords = np.indices((size, size))[axis]
            center = x if axis == 1 else y
            for length in lengths:
                low = int(center) - length // 2
                gap = mask & (coords >= low) & (coords < low + length)
                # Avoid removing another leg of a bent connector far from the chosen center.
                yy, xx = np.indices((size, size))
                gap &= (abs(xx - int(x)) <= 4) & (abs(yy - int(y)) <= 4)
                removed = int((gap & skel).sum())
                if not removed or removed / int(skel.sum()) > config["max_removed_fraction"]:
                    continue
                if (gap & ndi.binary_dilation(fills, structure=NEIGHBORS, iterations=2)).any():
                    continue
                remaining = mask & ~gap
                _, components = ndi.label(remaining, structure=NEIGHBORS)
                if components != 2:
                    continue
                return gap, {
                    "edge": edge["nodes"],
                    "axis": axis,
                    "strip_pixels": length,
                    "removed_skeleton_pixels": removed,
                    "edge_skeleton_pixels": int(skel.sum()),
                    "removed_fraction": removed / int(skel.sum()),
                }
    raise RuntimeError(f"No admissible short gap for parent seed {parent['graph_seed']}")


def corrupt(parent: dict, kind: str, seed: int, config: dict) -> tuple:
    rng = np.random.default_rng(seed)
    image = parent["clean"].astype(np.float32) / 255
    gap = np.zeros_like(image, dtype=bool)
    info = {"kind": kind, "seed": seed, "noise_sigma": 0.0, "blur_sigma": 0.0}
    if kind == "gap":
        gap, info["gap"] = _make_gap(parent, rng, config)
        image[gap] = 1.0
    if kind in {"blur", "gap"}:
        info["blur_sigma"] = float(rng.uniform(0.2, config["blur_sigma_max"]))
        image = ndi.gaussian_filter(image, info["blur_sigma"], mode="nearest")
    if kind in {"noise", "gap"}:
        info["noise_sigma"] = float(rng.uniform(0.01, config["noise_sigma_max"]))
        image = image + rng.normal(0, info["noise_sigma"], image.shape)
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8), gap, info


def _strata(n: int, weights: dict, rng: np.random.Generator) -> list[str]:
    kinds = list(weights)
    targets = np.array([weights[kind] * n for kind in kinds])
    counts = np.floor(targets).astype(int)
    for index in np.argsort(-(targets - counts))[: n - int(counts.sum())]:
        counts[index] += 1
    result = [kind for kind, count in zip(kinds, counts, strict=True) for _ in range(count)]
    rng.shuffle(result)
    return result


def generate(config_path: Path, output: Path) -> dict:
    config = read_config(config_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty dataset directory: {output}")
    rng = np.random.default_rng(config["seed"])
    total = sum(config[split] for split in SPLITS)
    parents, seen = [], set()
    while len(parents) < total:
        seeds = rng.integers(0, 2**32, 2).tolist()
        parent = sample_parent(*seeds, config)
        clean_hash = hash_bytes(parent["clean"].tobytes())
        if clean_hash not in seen:
            seen.add(clean_hash)
            parents.append(parent)
    # No damaged images exist before this assignment of clean parents to splits.
    rng.shuffle(parents)
    assignments = []
    index = 0
    for split in SPLITS:
        kinds = _strata(config[split], config["corruption_weights"], rng)
        for kind in kinds:
            assignments.append((parents[index], split, kind, int(rng.integers(0, 2**32))))
            index += 1
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (parent, split, kind, corruption_seed) in enumerate(assignments):
        diagram_id = f"d{index:05d}"
        damaged, gap, corruption = corrupt(parent, kind, corruption_seed, config)
        folder = output / split / diagram_id
        folder.mkdir(parents=True)
        Image.fromarray(parent["clean"]).save(folder / "clean.png")
        Image.fromarray(damaged).save(folder / "damaged.png")
        fills, bands = node_regions(parent["nodes"], config["image_size"])
        connector_mask = np.logical_or.reduce(parent["masks"])
        attachments = connector_mask & np.logical_or.reduce(bands)
        masks = {
            "connectors": connector_mask,
            "gap": gap,
            "attachments": attachments,
            "skeleton": skeletonize(parent["clean"] < 128),
            "node_regions": fills,
        }
        for name, mask in masks.items():
            save_mask(folder / f"{name}.png", mask)
        edge_annotations = []
        for i, (edge, mask) in enumerate(zip(parent["edges"], parent["masks"], strict=True)):
            save_mask(folder / f"edge_{i}.png", mask)
            edge_annotations.append({**edge, "mask": f"{split}/{diagram_id}/edge_{i}.png"})
        nodes = [node.to_dict() for node in parent["nodes"]]
        geometry = {"nodes": nodes, "edges": parent["edges"]}
        record = {
            "diagram_id": diagram_id,
            "parent_family_id": hash_json(geometry),
            "split": split,
            "nodes": nodes,
            "edges": [e["nodes"] for e in parent["edges"]],
            "connector_annotations": edge_annotations,
            "clean": f"{split}/{diagram_id}/clean.png",
            "damaged": f"{split}/{diagram_id}/damaged.png",
            "masks": {name: f"{split}/{diagram_id}/{name}.png" for name in masks},
            "clean_pixel_sha256": hash_bytes(parent["clean"].tobytes()),
            "damaged_pixel_sha256": hash_bytes(damaged.tobytes()),
            "geometry_sha256": hash_json(geometry),
            "graph_seed": parent["graph_seed"],
            "renderer_seed": parent["renderer_seed"],
            "geometry_attempt": parent["geometry_attempt"],
            "corruption": corruption,
            "renderer_version": "raster-v1",
            "size": config["image_size"],
        }
        write_json(folder / "annotation.json", record)
        records.append(record)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    summary = {
        "version": config["version"],
        "config": config,
        "images": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "strata": {
            s: dict(Counter(r["corruption"]["kind"] for r in records if r["split"] == s))
            for s in SPLITS
        },
        "manifest_sha256": hash_bytes(manifest.read_bytes()),
        "source_sha256": source_fingerprint(),
        "split_before_corruption": True,
        "mask_convention": "255=positive; image convention is 0=black ink, 255=white",
    }
    write_json(output / "dataset.json", summary)
    return summary


def read_manifest(root: Path, split: str | None = None) -> list[dict]:
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    return [record for record in records if split is None or record["split"] == split]
