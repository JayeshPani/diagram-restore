"""Small deterministic artifact helpers; generated outputs are not source edits."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_json(value) -> str:
    return hash_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")).copy()


def save_mask(path: Path, mask: np.ndarray) -> None:
    """Annotation masks use white=positive, unlike the black-ink image convention."""
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def source_fingerprint() -> str:
    root = Path(__file__).parent
    return hash_bytes(
        b"".join(p.name.encode() + p.read_bytes() for p in sorted(root.rglob("*.py")))
    )


def git_revision() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None
