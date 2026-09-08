"""Validation and immutable representation of Neri's reviewed DINOv3 head."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

import torch

DINO_BACKBONE = "dinov3_vitb16"
DINO_FEATURE_DIM = 768
DINO_PREPROCESSING = "letterbox224_imagenet"
DINO_EVENT_AGGREGATION = "mean_l2_normalized_crop_embeddings"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CheckpointValidationError(ValueError):
    """Raised when a classifier checkpoint is incompatible with Neri."""


@dataclass(frozen=True)
class DinoV3Checkpoint:
    path: Path | None
    backbone: str
    feature_dim: int
    classes: tuple[str, ...]
    head_weight: torch.Tensor
    head_bias: torch.Tensor
    prototypes: torch.Tensor
    threshold: float
    encoder_weights: str
    encoder_sha256: str
    preprocessing: str
    event_aggregation: str
    fingerprint: str


def _tensor(value: Any, field: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"{field} must be a torch.Tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise CheckpointValidationError(f"{field} contains non-finite values")
    return result


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(payload.get("backbone", "")).encode("utf-8"))
    digest.update(str(payload.get("feature_dim", "")).encode("ascii"))
    for name in payload.get("classes", ()):
        digest.update(str(name).encode("utf-8")); digest.update(b"\0")
    for field in ("weight", "bias"):
        value = payload.get("head_state", {}).get(field)
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    prototypes = payload.get("prototypes")
    if isinstance(prototypes, torch.Tensor):
        digest.update(prototypes.detach().cpu().contiguous().numpy().tobytes())
    digest.update(repr(payload.get("threshold")).encode("ascii"))
    digest.update(str(payload.get("encoder_sha256", "")).encode("ascii"))
    return digest.hexdigest()


def validate_checkpoint(saved: Mapping[str, Any], *, path: Path | None = None,
                        fingerprint: str | None = None) -> DinoV3Checkpoint:
    if not isinstance(saved, Mapping):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    backbone = str(saved.get("backbone", ""))
    feature_dim = saved.get("feature_dim")
    if backbone != DINO_BACKBONE or feature_dim != DINO_FEATURE_DIM:
        raise CheckpointValidationError(
            "Expected dinov3_vitb16 with 768-dimensional features; feature spaces are not interchangeable"
        )
    raw_classes = saved.get("classes")
    if not isinstance(raw_classes, (list, tuple)):
        raise CheckpointValidationError("classes must be a list")
    classes = tuple(str(item).strip() for item in raw_classes)
    if len(classes) < 2 or any(not item for item in classes) or len(set(classes)) != len(classes):
        raise CheckpointValidationError("Invalid class mapping")
    head_state = saved.get("head_state")
    if not isinstance(head_state, Mapping):
        raise CheckpointValidationError("head_state must be a mapping")
    weight = _tensor(head_state.get("weight"), "head_state.weight")
    bias = _tensor(head_state.get("bias"), "head_state.bias")
    expected = (len(classes), DINO_FEATURE_DIM)
    if tuple(weight.shape) != expected or tuple(bias.shape) != (len(classes),):
        raise CheckpointValidationError(f"Invalid head shape; expected {expected} and {(len(classes),)}")
    prototypes = _tensor(saved.get("prototypes"), "prototypes")
    if tuple(prototypes.shape) != expected:
        raise CheckpointValidationError(f"Invalid prototype shape; expected {expected}")
    try:
        threshold = float(saved.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError("threshold must be finite") from exc
    if not math.isfinite(threshold):
        raise CheckpointValidationError("threshold must be finite")
    preprocessing = str(saved.get("preprocessing", ""))
    if preprocessing != DINO_PREPROCESSING:
        raise CheckpointValidationError(f"Unsupported preprocessing: {preprocessing or '<missing>'}")
    aggregation = str(saved.get("event_aggregation", ""))
    if aggregation != DINO_EVENT_AGGREGATION:
        raise CheckpointValidationError(f"Unsupported event aggregation: {aggregation or '<missing>'}")
    encoder_weights = str(saved.get("encoder_weights", "")).strip()
    encoder_sha256 = str(saved.get("encoder_sha256", "")).strip().lower()
    if not encoder_weights:
        raise CheckpointValidationError("encoder_weights is required")
    if not _SHA256_RE.fullmatch(encoder_sha256):
        raise CheckpointValidationError("encoder_sha256 must be a 64-character SHA-256")
    return DinoV3Checkpoint(
        path=path, backbone=backbone, feature_dim=DINO_FEATURE_DIM, classes=classes,
        head_weight=weight, head_bias=bias, prototypes=prototypes, threshold=threshold,
        encoder_weights=encoder_weights, encoder_sha256=encoder_sha256,
        preprocessing=preprocessing, event_aggregation=aggregation,
        fingerprint=fingerprint or _payload_fingerprint(saved),
    )


def load_checkpoint(path: str | Path) -> DinoV3Checkpoint:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DINOv3 classifier checkpoint not found: {resolved}")
    try:
        saved = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(resolved, map_location="cpu")
    with resolved.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    return validate_checkpoint(saved, path=resolved, fingerprint=fingerprint)
