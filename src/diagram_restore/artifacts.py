"""Reproducible integrity audits and inspectable images for the first milestone."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import read_manifest
from .evaluation import edge_metrics, extract_edges, image_metrics, summarize
from .fixtures import all_fixtures
from .geometry import Node
from .io import hash_bytes, load_image, write_json


def fixture_report(output: Path) -> dict:
    rows = []
    for fixture in all_fixtures():
        extraction = extract_edges(fixture.image, fixture.nodes)
        expected = {tuple(sorted(edge)) for edge in fixture.expected}
        passed = set(extraction.edges) == expected and (
            len(extraction.invalid_merges) == fixture.expected_merges
        )
        rows.append(
            {
                "name": fixture.name,
                "passed": passed,
                "expected_edges": sorted(expected),
                "actual": extraction.to_dict(),
                "against_reference": edge_metrics(extraction.edges, fixture.reference),
            }
        )
    result = {
        "total": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "cases": rows,
        "drawing": "independent hand-specified Pillow fixtures, not the dataset renderer",
    }
    write_json(output / "fixtures.json", result)
    return result


def audit(root: Path, output: Path) -> dict:
    records = read_manifest(root)
    failures = []
    for key in ("diagram_id", "parent_family_id", "clean_pixel_sha256", "geometry_sha256"):
        if len({row[key] for row in records}) != len(records):
            failures.append({"type": "duplicate_identity", "key": key})
    clean_rows = []
    for row in records:
        nodes = [Node.from_dict(n) for n in row["nodes"]]
        clean = load_image(root / row["clean"])
        if hash_bytes(clean.tobytes()) != row["clean_pixel_sha256"]:
            failures.append({"id": row["diagram_id"], "type": "clean_hash_mismatch"})
        # Reserved damaged images are read for byte integrity only, never scored here.
        damaged = load_image(root / row["damaged"])
        if hash_bytes(damaged.tobytes()) != row["damaged_pixel_sha256"]:
            failures.append({"id": row["diagram_id"], "type": "damaged_hash_mismatch"})
        extraction = extract_edges(clean, nodes)
        metrics = edge_metrics(extraction.edges, row["edges"])
        clean_rows.append(metrics)
        if not metrics["exact_graph"]:
            failures.append(
                {
                    "id": row["diagram_id"],
                    "type": "clean_graph_mismatch",
                    "expected": row["edges"],
                    "actual": extraction.edges,
                }
            )
        gap = load_image(root / row["masks"]["gap"]) > 0
        if row["corruption"]["kind"] == "gap":
            info = row["corruption"]["gap"]
            if not gap.any() or info["removed_fraction"] > 0.2:
                failures.append({"id": row["diagram_id"], "type": "invalid_gap"})
        elif gap.any():
            failures.append({"id": row["diagram_id"], "type": "unexpected_gap"})
    fixture_results = fixture_report(output)
    result = {
        "records": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "identity_hashes_unique": not any(f.get("type") == "duplicate_identity" for f in failures),
        "clean_graph_recovery": summarize(clean_rows),
        "failures": failures,
        "fixture_passed": fixture_results["passed"],
        "fixture_total": fixture_results["total"],
        "integrity_passed": not failures and fixture_results["passed"] == fixture_results["total"],
        "test_access": "clean-render graph integrity and image hashes only; no damaged-test scores",
    }
    write_json(output / "integrity.json", result)
    return result


def evaluate_observations(root: Path, output: Path, split: str = "validation") -> dict:
    if split not in {"train", "validation"}:
        raise ValueError("Development command does not evaluate the reserved test set")
    rows = []
    for row in read_manifest(root, split):
        nodes = [Node.from_dict(n) for n in row["nodes"]]
        clean, damaged = load_image(root / row["clean"]), load_image(root / row["damaged"])
        extracted = extract_edges(damaged, nodes)
        rows.append(
            {
                "diagram_id": row["diagram_id"],
                "kind": row["corruption"]["kind"],
                "expected_edges": row["edges"],
                "predicted_edges": extracted.edges,
                **edge_metrics(extracted.edges, row["edges"]),
                **image_metrics(damaged, clean),
                "invalid_merges": extracted.invalid_merges,
            }
        )
    thresholds = {}
    for threshold in (0.4, 0.6):
        alternative = []
        for row in read_manifest(root, split):
            nodes = [Node.from_dict(n) for n in row["nodes"]]
            pred = extract_edges(load_image(root / row["damaged"]), nodes, threshold)
            alternative.append(edge_metrics(pred.edges, row["edges"]))
        thresholds[str(threshold)] = summarize(alternative)
    result = {
        "split": split,
        "model": "B0_unchanged_damaged_input",
        "threshold": 0.5,
        "overall": summarize(rows),
        "by_corruption": {
            kind: summarize([r for r in rows if r["kind"] == kind])
            for kind in sorted({r["kind"] for r in rows})
        },
        "threshold_sensitivity": thresholds,
        "mean_foreground_dice": float(np.mean([r["foreground_dice"] for r in rows])),
        "mean_ssim": float(np.mean([r["ssim"] for r in rows])),
        "per_image": rows,
    }
    write_json(output / f"{split}_damaged_input.json", result)
    return result


def contact_sheets(root: Path, output: Path) -> list[str]:
    records = read_manifest(root)
    development = [r for r in records if r["split"] in {"train", "validation"}]
    by_kind = {
        kind: [r for r in development if r["corruption"]["kind"] == kind][:9]
        for kind in ("clean", "noise", "blur", "gap")
    }
    # Interleave the four kinds, making each page a mixture of corruption types.
    # A small dataset may have fewer than 9 examples of some kind, so this pads to
    # whatever is actually available instead of assuming a fixed 9-per-kind, 36-total shape.
    slots = max((len(rows) for rows in by_kind.values()), default=0)
    selected = [
        by_kind[kind][i]
        for i in range(slots)
        for kind in ("clean", "noise", "blur", "gap")
        if i < len(by_kind[kind])
    ]
    font = ImageFont.load_default(size=12)
    paths = []
    pages = -(-len(selected) // 18) if selected else 0
    for page in range(pages):
        sheet = Image.new("RGB", (960, 1200), "#eef0f3")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected[page * 18 : (page + 1) * 18]):
            x, y = (i % 3) * 320, (i // 3) * 200
            draw.text(
                (x + 10, y + 6),
                f"{row['diagram_id']}  {row['corruption']['kind']}",
                fill="black",
                font=font,
            )
            for j, key in enumerate(("clean", "damaged")):
                image = Image.fromarray(load_image(root / row[key])).convert("RGB")
                image = image.resize((128, 128), Image.Resampling.NEAREST)
                sheet.paste(image, (x + 10 + j * 154, y + 40))
                draw.text((x + 10 + j * 154, y + 24), key, fill="#405164", font=font)
            nodes = [Node.from_dict(n) for n in row["nodes"]]
            pred = extract_edges(load_image(root / row["damaged"]), nodes)
            score = edge_metrics(pred.edges, row["edges"])
            draw.text(
                (x + 10, y + 176),
                f"edges {len(row['edges'])}; missing {score['fn']}; invented {score['fp']}",
                fill="#405164",
                font=font,
            )
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"contact_sheet_{page + 1}.png"
        sheet.save(path)
        paths.append(str(path))
    return paths


def fixture_sheet(output: Path) -> str:
    from .fixtures import CASES, make_fixture

    sheet = Image.new("RGB", (1040, 950), "#eef0f3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=12)
    for i, case in enumerate(CASES):
        f = make_fixture(case)
        x, y = i % 4 * 260, i // 4 * 190
        draw.text((x + 8, y + 8), case, fill="black", font=font)
        image = Image.fromarray(f.image).convert("RGB").resize((128, 128), Image.Resampling.NEAREST)
        sheet.paste(image, (x + 8, y + 30))
        draw.text(
            (x + 8, y + 166),
            "expected " + ",".join("-".join(e) for e in f.expected),
            fill="#405164",
            font=font,
        )
    output.mkdir(parents=True, exist_ok=True)
    path = output / "evaluator_fixtures.png"
    sheet.save(path)
    return str(path)


def overlay_audit(root: Path, output: Path) -> dict:
    """Development contact overlays, including node IDs, components and missing edges."""
    rows = read_manifest(root, "validation")
    selected = []
    for kind in ("clean", "noise", "blur", "gap"):
        selected.extend([r for r in rows if r["corruption"]["kind"] == kind][:8])
    font = ImageFont.load_default(size=12)
    index = []
    paths = []
    for page in range(2):
        sheet = Image.new("RGB", (1000, 960), "#eef0f3")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(selected[page * 16 : (page + 1) * 16]):
            x, y = (i % 4) * 250, (i // 4) * 240
            gray = load_image(root / row["damaged"])
            nodes = [Node.from_dict(n) for n in row["nodes"]]
            image = (
                Image.fromarray(gray).convert("RGB").resize((192, 192), Image.Resampling.NEAREST)
            )
            painter = ImageDraw.Draw(image)
            for node in nodes:
                box = tuple(v * 3 for v in node.bbox)
                painter.rectangle(box, outline="#3987b9", width=1)
                painter.text(
                    (node.center[0] * 3 - 3, node.center[1] * 3 - 6),
                    node.id,
                    fill="#145078",
                    font=font,
                )
            sheet.paste(image, (x + 8, y + 23))
            extraction = extract_edges(gray, nodes)
            metrics = edge_metrics(extraction.edges, row["edges"])
            draw.text(
                (x + 8, y + 3),
                f"{row['diagram_id']} {row['corruption']['kind']}",
                fill="black",
                font=font,
            )
            missing = ",".join("-".join(e) for e in metrics["missing_edges"]) or "none"
            draw.text((x + 8, y + 219), f"missing: {missing}", fill="#94302d", font=font)
            index.append(
                {
                    "id": row["diagram_id"],
                    "reference": row["edges"],
                    "extracted": extraction.edges,
                    "missing": metrics["missing_edges"],
                    "invented": metrics["invented_edges"],
                }
            )
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"validation_overlays_{page + 1}.png"
        sheet.save(path)
        paths.append(str(path))
    report = {
        "examples": index,
        "count": len(index),
        "sheets": paths,
        "human_review": "pending visual inspection; not implied by automatic generation",
    }
    write_json(output / "visual_audit_index.json", report)
    return report
