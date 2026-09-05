"""Prepare new, metadata-free upload copies without modifying source media."""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from system.config import APP_VERSION
from .policy import MAX_IMAGE_BYTES, MAX_IMAGE_EDGE, JPEG_QUALITY, AGREEMENT_VERSION


def prepare_sample(path: Path, job: dict) -> tuple[bytes, bytes]:
    with Image.open(path) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated_image_excluded")
        # Loading enforces Pillow's decompression bomb protection before allocating copies.
        source.load()
        orientation = source.getexif().get(274, 1)
        width, height = source.size
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        # A fresh pixel-only object prevents EXIF/XMP/ICC/comments being copied by plugins.
        clean = Image.frombytes("RGB", image.size, image.tobytes())
    output = io.BytesIO()
    clean.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    photo = output.getvalue()
    if len(photo) > MAX_IMAGE_BYTES:
        raise ValueError("compressed_image_too_large")
    payload = job["payload"]
    predictions = []
    # Older inference paths may have handled EXIF differently. Omit ambiguous boxes.
    if orientation == 1:
        for box in payload.get("model_boxes", []):
            x1, y1, x2, y2 = box["bbox"]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue
            predictions.append({"species": box["species"], "confidence": box["confidence"],
                                "bbox_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height]})
    annotation = {
        "schema_version": 1, "agreement_version": AGREEMENT_VERSION,
        "sample_id": job["sample_id"], "revision": job["revision"],
        "species": payload["species"], "species_counts": payload.get("species_counts", []),
        "confirmed_empty": payload.get("empty", False),
        "human_verified_fields": ["species", "species_counts", "confirmed_empty"],
        "image": {"format": "jpeg", "width": clean.width, "height": clean.height},
        "model_predictions": predictions,
        "model_boxes_are_human_verified": False,
        "application_version": APP_VERSION,
        "privacy_processing": "metadata_removed_visual_identity_not_guaranteed",
    }
    return photo, json.dumps(annotation, ensure_ascii=False, allow_nan=False).encode("utf-8")
