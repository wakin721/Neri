"""Prepare metadata-free JPEGs plus per-species YOLO annotations."""
from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image, ImageOps

from system.config import APP_VERSION
from .policy import AGREEMENT_VERSION, JPEG_QUALITY, MAX_IMAGE_BYTES, MAX_IMAGE_EDGE


def _yolo_line(box: dict, width: int, height: int) -> str | None:
    try:
        x1, y1, x2, y2 = [float(x) for x in box["bbox"][:4]]
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def prepare_sample(path: Path, job: dict) -> tuple[bytes, bytes]:
    with Image.open(path) as source:
        if source.format == "MPO":
            source.seek(0)
        elif getattr(source, "n_frames", 1) != 1:
            raise ValueError("animated_image_excluded")
        source.load()
        orientation = source.getexif().get(274, 1)
        width, height = source.size
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        clean = Image.frombytes("RGB", image.size, image.tobytes())

    output = io.BytesIO()
    clean.save(output, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    photo = output.getvalue()
    if len(photo) > MAX_IMAGE_BYTES:
        raise ValueError("compressed_image_too_large")

    payload = job["payload"]
    species = list(payload["species"])
    training_names = dict(payload.get("training_names", {}))
    predictions = []
    yolo_labels: dict[str, str] = {}
    yolo_classes: dict[str, str] = {}

    for name in species:
        yolo_classes[name] = f"{training_names.get(name, name)}\n"
        yolo_labels[name] = ""

    if orientation == 1 and not payload.get("empty", False):
        lines_by_species: dict[str, list[str]] = {name: [] for name in species}
        for box in payload.get("model_boxes", []):
            name = str(box.get("species", "")).strip()
            line = _yolo_line(box, width, height)
            if line is None:
                continue
            x1, y1, x2, y2 = [float(x) for x in box["bbox"][:4]]
            predictions.append(
                {
                    "species": name,
                    "confidence": box.get("confidence", 0),
                    "bbox_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
                }
            )
            if name in lines_by_species:
                lines_by_species[name].append(line)
        for name, lines in lines_by_species.items():
            yolo_labels[name] = "".join(lines)

    package = {
        "schema_version": 2,
        "agreement_version": AGREEMENT_VERSION,
        "sample_id": job["sample_id"],
        "revision": job["revision"],
        "species": species,
        "upload_species": species,
        "species_counts": payload.get("species_counts", []),
        "confirmed_empty": payload.get("empty", False),
        "human_verified_fields": ["species", "species_counts", "confirmed_empty"],
        "image": {"format": "jpeg", "width": clean.width, "height": clean.height},
        "model_predictions": predictions,
        "model_boxes_are_human_verified": False,
        "yolo_labels": yolo_labels,
        "yolo_classes": yolo_classes,
        "application_version": APP_VERSION,
        "privacy_processing": "metadata_removed_visual_identity_not_guaranteed",
    }
    return photo, json.dumps(package, ensure_ascii=False, allow_nan=False).encode("utf-8")
