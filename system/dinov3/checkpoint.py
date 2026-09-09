"""Validation and immutable representation of Neri's DINOv3 classifiers."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping

DINO_BACKBONE = "dinov3_vitb16"
DINO_FEATURE_DIM = 768
DINO_PREPROCESSING = "letterbox224_imagenet"
DINO_EVENT_AGGREGATION = "mean_l2_normalized_crop_embeddings"
DINO_MULTI_PROTOTYPE_HEAD = "multi_prototype"
DINO_SELECTION_K = 3
DINO_DECISION = "squared_euclidean_to_nearest_prototype"
DINO_REJECTION_SCORE = "cosine_similarity_to_winning_prototype"
DINO_ENCODER_CODE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CheckpointValidationError(ValueError):
    """Raised when a classifier checkpoint is incompatible with Neri."""


@dataclass(frozen=True)
class DinoV3Checkpoint:
    path: Path | None
    backbone: str
    feature_dim: int
    classes: tuple[str, ...]
    head_type: str
    feature_center: Any
    prototypes: Any
    prototype_class_indices: Any
    prototypes_per_class: tuple[int, ...]
    selection_k: int
    decision: str
    rejection_score: str
    threshold: float
    encoder_weights: str
    encoder_sha256: str
    encoder_code_commit: str
    preprocessing: str
    event_aggregation: str
    fingerprint: str
    # Temporary compatibility fields for the pre-migration classifier. They
    # are never required by the Multi-prototype checkpoint format.
    head_weight: Any | None = None
    head_bias: Any | None = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load or validate a DINOv3 checkpoint"
        ) from exc
    return torch


def _tensor(value: Any, field: str) -> Any:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"{field} must be a torch.Tensor")
    result = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise CheckpointValidationError(f"{field} contains non-finite values")
    return result


def _index_tensor(value: Any, field: str) -> Any:
    torch = _require_torch()
    if not isinstance(value, torch.Tensor):
        raise CheckpointValidationError(f"{field} must be a torch.Tensor")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise CheckpointValidationError(f"{field} must contain integer indices")
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _threshold(saved: Mapping[str, Any]) -> float:
    try:
        threshold = float(saved.get("threshold"))
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError("threshold must be finite") from exc
    if not math.isfinite(threshold):
        raise CheckpointValidationError("threshold must be finite")
    return threshold


def _common_metadata(saved: Mapping[str, Any]) -> tuple[str, str, str, str]:
    preprocessing = str(saved.get("preprocessing", ""))
    if preprocessing != DINO_PREPROCESSING:
        raise CheckpointValidationError(
            f"Unsupported preprocessing: {preprocessing or '<missing>'}"
        )
    aggregation = str(saved.get("event_aggregation", ""))
    if aggregation != DINO_EVENT_AGGREGATION:
        raise CheckpointValidationError(
            f"Unsupported event aggregation: {aggregation or '<missing>'}"
        )
    encoder_weights = str(saved.get("encoder_weights", "")).strip()
    encoder_sha256 = str(saved.get("encoder_sha256", "")).strip().lower()
    if not encoder_weights:
        raise CheckpointValidationError("encoder_weights is required")
    if not _SHA256_RE.fullmatch(encoder_sha256):
        raise CheckpointValidationError(
            "encoder_sha256 must be a 64-character SHA-256"
        )
    return preprocessing, aggregation, encoder_weights, encoder_sha256


def _payload_fingerprint(payload: Mapping[str, Any]) -> str:
    """Generate deterministic fingerprints for in-memory test checkpoints.

    Real files use the checkpoint file SHA-256 in ``load_checkpoint``. The
    legacy branch preserves the previous fingerprint algorithm so existing
    development registries are not gratuitously invalidated during migration.
    """

    torch = _require_torch()
    digest = hashlib.sha256()
    digest.update(str(payload.get("backbone", "")).encode("utf-8"))
    digest.update(str(payload.get("feature_dim", "")).encode("ascii"))
    for name in payload.get("classes", ()):
        digest.update(str(name).encode("utf-8"))
        digest.update(b"\0")

    if str(payload.get("head_type", "")).strip() == DINO_MULTI_PROTOTYPE_HEAD:
        for field in ("feature_center", "prototypes", "prototype_class_indices"):
            value = payload.get(field)
            if isinstance(value, torch.Tensor):
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        for value in payload.get("prototypes_per_class", ()):
            digest.update(str(value).encode("ascii"))
            digest.update(b"\0")
        digest.update(str(payload.get("selection_k", "")).encode("ascii"))
        digest.update(str(payload.get("decision", "")).encode("utf-8"))
        digest.update(str(payload.get("rejection_score", "")).encode("utf-8"))
        digest.update(str(payload.get("encoder_code_commit", "")).encode("ascii"))
    else:
        head_state = payload.get("head_state")
        if isinstance(head_state, Mapping):
            for field in ("weight", "bias"):
                value = head_state.get(field)
                if isinstance(value, torch.Tensor):
                    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        prototypes = payload.get("prototypes")
        if isinstance(prototypes, torch.Tensor):
            digest.update(prototypes.detach().cpu().contiguous().numpy().tobytes())

    digest.update(repr(payload.get("threshold")).encode("ascii"))
    digest.update(str(payload.get("encoder_sha256", "")).encode("ascii"))
    return digest.hexdigest()


def _validate_multi_prototype(
    saved: Mapping[str, Any], classes: tuple[str, ...]
) -> tuple[Any, Any, Any, tuple[int, ...], int, str, str, str]:
    torch = _require_torch()
    if saved.get("schema_version") != 1:
        raise CheckpointValidationError("Unsupported Multi-prototype schema_version")

    feature_center = _tensor(saved.get("feature_center"), "feature_center")
    if tuple(feature_center.shape) != (DINO_FEATURE_DIM,):
        raise CheckpointValidationError(
            f"Invalid feature_center shape; expected {(DINO_FEATURE_DIM,)}"
        )

    prototypes = _tensor(saved.get("prototypes"), "prototypes")
    if prototypes.ndim != 2 or prototypes.shape[1] != DINO_FEATURE_DIM or len(prototypes) == 0:
        raise CheckpointValidationError(
            "Invalid prototype shape; expected (M, 768) with M > 0"
        )

    indices = _index_tensor(
        saved.get("prototype_class_indices"),
        "prototype_class_indices",
    )
    if tuple(indices.shape) != (len(prototypes),):
        raise CheckpointValidationError(
            "prototype_class_indices must have one index per prototype"
        )
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= len(classes))):
        raise CheckpointValidationError("prototype_class_indices contains an invalid class index")

    raw_counts = saved.get("prototypes_per_class")
    if not isinstance(raw_counts, (list, tuple)) or len(raw_counts) != len(classes):
        raise CheckpointValidationError(
            "prototypes_per_class must contain one count per class"
        )
    counts: list[int] = []
    for value in raw_counts:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CheckpointValidationError(
                "prototypes_per_class must contain positive integers"
            )
        counts.append(value)
    actual_counts = torch.bincount(indices, minlength=len(classes)).tolist()
    if counts != actual_counts:
        raise CheckpointValidationError(
            "prototypes_per_class does not match prototype_class_indices"
        )

    selection_k = saved.get("selection_k")
    if isinstance(selection_k, bool) or selection_k != DINO_SELECTION_K:
        raise CheckpointValidationError(
            f"selection_k must be {DINO_SELECTION_K} for Multi-prototype checkpoints"
        )
    decision = str(saved.get("decision", ""))
    if decision != DINO_DECISION:
        raise CheckpointValidationError(f"Unsupported decision rule: {decision or '<missing>'}")
    rejection_score = str(saved.get("rejection_score", ""))
    if rejection_score != DINO_REJECTION_SCORE:
        raise CheckpointValidationError(
            f"Unsupported rejection score: {rejection_score or '<missing>'}"
        )
    encoder_code_commit = str(saved.get("encoder_code_commit", "")).strip().lower()
    if encoder_code_commit != DINO_ENCODER_CODE_COMMIT:
        raise CheckpointValidationError("encoder_code_commit does not match DINOv3 runtime")

    return (
        None,
        None,
        feature_center,
        tuple(counts),
        int(selection_k),
        decision,
        rejection_score,
        encoder_code_commit,
    )


def _validate_legacy_linear(
    saved: Mapping[str, Any], classes: tuple[str, ...]
) -> tuple[Any, Any, Any, Any, tuple[int, ...], int, str, str, str]:
    torch = _require_torch()
    head_state = saved.get("head_state")
    if not isinstance(head_state, Mapping):
        raise CheckpointValidationError("head_state must be a mapping")
    weight = _tensor(head_state.get("weight"), "head_state.weight")
    bias = _tensor(head_state.get("bias"), "head_state.bias")
    expected = (len(classes), DINO_FEATURE_DIM)
    if tuple(weight.shape) != expected or tuple(bias.shape) != (len(classes),):
        raise CheckpointValidationError(
            f"Invalid head shape; expected {expected} and {(len(classes),)}"
        )
    prototypes = _tensor(saved.get("prototypes"), "prototypes")
    if tuple(prototypes.shape) != expected:
        raise CheckpointValidationError(f"Invalid prototype shape; expected {expected}")
    return (
        weight,
        bias,
        torch.zeros(DINO_FEATURE_DIM, dtype=torch.float32),
        torch.arange(len(classes), dtype=torch.int64),
        (1,) * len(classes),
        1,
        "linear_head_with_prototype_consistency",
        "cosine_similarity_to_nearest_prototype",
        str(saved.get("encoder_code_commit", "")).strip().lower(),
    )


def validate_checkpoint(
    saved: Mapping[str, Any],
    *,
    path: Path | None = None,
    fingerprint: str | None = None,
) -> DinoV3Checkpoint:
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
    if (
        len(classes) < 2
        or any(not item for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise CheckpointValidationError("Invalid class mapping")

    threshold = _threshold(saved)
    preprocessing, aggregation, encoder_weights, encoder_sha256 = _common_metadata(saved)
    head_type = str(saved.get("head_type", "")).strip().lower()

    if head_type == DINO_MULTI_PROTOTYPE_HEAD:
        (
            weight,
            bias,
            feature_center,
            counts,
            selection_k,
            decision,
            rejection_score,
            encoder_code_commit,
        ) = _validate_multi_prototype(saved, classes)
        prototypes = _tensor(saved.get("prototypes"), "prototypes")
        indices = _index_tensor(
            saved.get("prototype_class_indices"),
            "prototype_class_indices",
        )
    elif head_type in {"", "linear", "linear_head", "joint_linear_head"}:
        head_type = "linear"
        (
            weight,
            bias,
            feature_center,
            indices,
            counts,
            selection_k,
            decision,
            rejection_score,
            encoder_code_commit,
        ) = _validate_legacy_linear(saved, classes)
        prototypes = _tensor(saved.get("prototypes"), "prototypes")
    else:
        raise CheckpointValidationError(f"Unsupported DINOv3 head_type: {head_type}")

    return DinoV3Checkpoint(
        path=path,
        backbone=backbone,
        feature_dim=DINO_FEATURE_DIM,
        classes=classes,
        head_type=head_type,
        feature_center=feature_center,
        prototypes=prototypes,
        prototype_class_indices=indices,
        prototypes_per_class=counts,
        selection_k=selection_k,
        decision=decision,
        rejection_score=rejection_score,
        threshold=threshold,
        encoder_weights=encoder_weights,
        encoder_sha256=encoder_sha256,
        encoder_code_commit=encoder_code_commit,
        preprocessing=preprocessing,
        event_aggregation=aggregation,
        fingerprint=fingerprint or _payload_fingerprint(saved),
        head_weight=weight,
        head_bias=bias,
    )


def load_checkpoint(path: str | Path) -> DinoV3Checkpoint:
    torch = _require_torch()
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
