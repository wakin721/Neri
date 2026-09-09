"""Materialize the reviewed DINOv3 classifier seed from text-safe chunks."""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
from pathlib import Path
from typing import Sequence

DINO_CLASSIFIER_FILENAME = "dinov3_classifier_merged_reviewed_20260908.pt"
DINO_CLASSIFIER_SHA256 = "b2f334da61c9feee51cff51bcded0af16878dcd5d72ea4f2e3e47b7f2adab76a"
SEED_PART_PREFIX = DINO_CLASSIFIER_FILENAME + ".b64."


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_classifier_seed(seed_dir: str | Path, *, force: bool = False) -> Path:
    """Reconstruct the reviewed .pt seed and verify its canonical SHA-256."""

    root = Path(seed_dir).resolve()
    target = root / DINO_CLASSIFIER_FILENAME
    if target.is_file() and not force:
        if _sha256(target) == DINO_CLASSIFIER_SHA256:
            return target
        raise RuntimeError("Existing DINOv3 classifier seed has an unexpected SHA-256.")

    parts = sorted(root.glob(SEED_PART_PREFIX + "*"), key=lambda path: path.name)
    if not parts:
        raise FileNotFoundError(f"DINOv3 classifier seed chunks are missing under {root}")

    encoded = b"".join(part.read_bytes().strip() for part in parts)
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError("DINOv3 classifier seed Base64 is invalid.") from exc

    root.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(payload)
    actual = _sha256(temporary)
    if actual != DINO_CLASSIFIER_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "DINOv3 classifier seed SHA-256 mismatch: "
            f"expected {DINO_CLASSIFIER_SHA256}, got {actual}"
        )
    temporary.replace(target)
    return target


def _run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("seed_dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    path = materialize_classifier_seed(args.seed_dir, force=args.force)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
