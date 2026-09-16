"""Validation and immutable representation of Neri's DINOv2 classifier."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

DINO_BACKBONE = "facebook/dinov2-base"
DINO_ARCHITECTURE_ID = "dinov2_vitb14"
DINO_FEATURE_DIM = 768
DINO_PREPROCESSING = "letterbox224_imagenet"
DINO_EVENT_AGGREGATION = "mean_l2_normalized_crop_embeddings"
DINO_MULTI_PROTOTYPE_HEAD = "multi_prototype"
DINO_REJECTION_MODE = "multi_dual_margin"
DINO_MODEL_REVISION = "f9e44c814b77203eaa57a6bdbbd535f21ede1415"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class CheckpointValidationError(ValueError):
    """Raised when a classifier checkpoint is incompatible with Neri."""


@dataclass(frozen=True)
class DinoV2Rejection:
    mode: str
    cosine_threshold: float
    adjusted_distance_score_threshold: float
    margin_weight: float
    class_distance_scales: tuple[float, ...] | None = None


@dataclass(frozen=True)
class DinoV2Checkpoint:
    path: Path | None
    backbone: str
    feature_dim: int
    classes: tuple[str, ...]
    feature_center: Any
    prototypes: Any
    prototype_class_indices: Any
    prototypes_per_class: tuple[int, ...]
    threshold: float
    encoder_weights: str
    encoder_sha256: str
    preprocessing: str
    event_aggregation: str
    rejection: DinoV2Rejection
    fingerprint: str
    head_type: str = DINO_MULTI_PROTOTYPE_HEAD
    selection_k: int = 0
    adaptive_k: bool = True
    k_by_class: dict[str, int] | None = None
    adaptive_k_selection: Any = None
    adaptive_k_folds: int | None = None
    prototype_norm_power: float = 0.0
    data_fingerprint: str = ""
    threshold_calibration: str = ""
    max_known_frr: float | None = None
    selected_by: str = ""
    optimization_seed: int | None = None
    rejection_score: str = ""
    calibration: Any = None
    scale_training: Any = None


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load or validate a DINOv2 checkpoint") from exc
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
    if value.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}:
        raise CheckpointValidationError(f"{field} must contain integer indices")
    return value.detach().to(device="cpu", dtype=torch.int64).contiguous()


def _finite_float(value: Any, field: str, *, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointValidationError(f"{field} must be finite") from exc
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise CheckpointValidationError(f"{field} must be {qualifier}")
    return result


def _classes(saved: Mapping[str, Any]) -> tuple[str, ...]:
    raw = saved.get("classes")
    if not isinstance(raw, (list, tuple)):
        raise CheckpointValidationError("classes must be a list")
    classes = tuple(str(item).strip() for item in raw)
    if len(classes) < 2 or any(not item for item in classes) or len(set(classes)) != len(classes):
        raise CheckpointValidationError("Invalid class mapping")
    return classes


def _metadata(saved: Mapping[str, Any]) -> tuple[str, str, str, str]:
    preprocessing = str(saved.get("preprocessing", ""))
    if preprocessing != DINO_PREPROCESSING:
        raise CheckpointValidationError(f"Unsupported preprocessing: {preprocessing or '<missing>'}")
    aggregation = str(saved.get("event_aggregation", ""))
    if aggregation != DINO_EVENT_AGGREGATION:
        raise CheckpointValidationError(f"Unsupported event aggregation: {aggregation or '<missing>'}")
    encoder_weights = str(saved.get("encoder_weights", "")).strip()
    if not encoder_weights:
        raise CheckpointValidationError("encoder_weights is required")
    encoder_sha256 = str(saved.get("encoder_sha256", "")).strip().lower()
    if not _SHA256_RE.fullmatch(encoder_sha256):
        raise CheckpointValidationError("encoder_sha256 must be a 64-character SHA-256")
    return preprocessing, aggregation, encoder_weights, encoder_sha256


def _rejection(saved: Mapping[str, Any], class_count: int) -> DinoV2Rejection:
    raw = saved.get("rejection")
    if not isinstance(raw, Mapping):
        raise CheckpointValidationError("rejection must be a mapping")
    mode = str(raw.get("mode", ""))
    if mode != DINO_REJECTION_MODE:
        raise CheckpointValidationError("rejection.mode must be multi_dual_margin")
    cosine = _finite_float(raw.get("cosine_threshold"), "cosine_threshold")
    adjusted = _finite_float(
        raw.get("adjusted_distance_score_threshold"),
        "adjusted_distance_score_threshold",
    )
    weight = _finite_float(raw.get("margin_weight"), "margin_weight", nonnegative=True)
    scales_raw = raw.get("class_distance_scales")
    scales: tuple[float, ...] | None = None
    if scales_raw is not None:
        if not isinstance(scales_raw, (list, tuple)) or len(scales_raw) != class_count:
            raise CheckpointValidationError("class_distance_scales must contain one value per class")
        parsed = []
        for value in scales_raw:
            scale = _finite_float(value, "class_distance_scales")
            if scale <= 0:
                raise CheckpointValidationError("class_distance_scales must contain positive finite values")
            parsed.append(scale)
        scales = tuple(parsed)
    return DinoV2Rejection(
        mode=mode,
        cosine_threshold=cosine,
        adjusted_distance_score_threshold=adjusted,
        margin_weight=weight,
        class_distance_scales=scales,
    )


def _prototype_contract(saved: Mapping[str, Any], classes: tuple[str, ...]):
    torch = _require_torch()
    center = _tensor(saved.get("feature_center"), "feature_center")
    if tuple(center.shape) != (DINO_FEATURE_DIM,):
        raise CheckpointValidationError("feature_center must have shape (768,)")
    prototypes = _tensor(saved.get("prototypes"), "prototypes")
    if prototypes.ndim != 2 or tuple(prototypes.shape[1:]) != (DINO_FEATURE_DIM,) or len(prototypes) == 0:
        raise CheckpointValidationError("prototypes must have shape (M, 768) with M > 0")
    if bool(torch.any(torch.linalg.vector_norm(prototypes, dim=1) <= 1e-12)):
        raise CheckpointValidationError("prototypes contains a zero-length prototype")
    indices = _index_tensor(saved.get("prototype_class_indices"), "prototype_class_indices")
    if tuple(indices.shape) != (len(prototypes),):
        raise CheckpointValidationError("prototype_class_indices must have one index per prototype")
    if bool(torch.any(indices < 0)) or bool(torch.any(indices >= len(classes))):
        raise CheckpointValidationError("prototype_class_indices contains an invalid class index")
    if set(indices.tolist()) != set(range(len(classes))):
        raise CheckpointValidationError("prototype_class_indices must represent every class")

    raw_counts = saved.get("prototypes_per_class")
    if not isinstance(raw_counts, (list, tuple)) or len(raw_counts) != len(classes):
        raise CheckpointValidationError("prototypes_per_class must contain one count per class")
    counts: list[int] = []
    for value in raw_counts:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CheckpointValidationError("prototypes_per_class must contain positive integers")
        counts.append(value)
    if counts != torch.bincount(indices, minlength=len(classes)).tolist():
        raise CheckpointValidationError("prototypes_per_class does not match prototype_class_indices")

    adaptive_k = saved.get("adaptive_k", False)
    if not isinstance(adaptive_k, bool):
        raise CheckpointValidationError("adaptive_k must be boolean")
    selection_k = saved.get("selection_k", 0 if adaptive_k else None)
    if isinstance(selection_k, bool) or not isinstance(selection_k, int):
        raise CheckpointValidationError("selection_k must be an integer")
    if adaptive_k:
        if selection_k < 0:
            raise CheckpointValidationError("selection_k must be nonnegative for adaptive-k checkpoints")
    elif selection_k <= 0:
        raise CheckpointValidationError("selection_k must be positive for fixed-k checkpoints")

    raw_k_by_class = saved.get("k_by_class")
    k_by_class: dict[str, int] | None = None
    if adaptive_k:
        if not isinstance(raw_k_by_class, Mapping) or set(raw_k_by_class) != set(classes):
            raise CheckpointValidationError("k_by_class must contain every class for adaptive-k checkpoints")
        k_by_class = {}
        for name, count in zip(classes, counts):
            value = raw_k_by_class.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value != count:
                raise CheckpointValidationError("k_by_class does not match prototypes_per_class")
            k_by_class[name] = value

    norm_power = _finite_float(saved.get("prototype_norm_power", 0.0), "prototype_norm_power", nonnegative=True)
    return center, prototypes, indices, tuple(counts), adaptive_k, selection_k, k_by_class, norm_power


def _json_fingerprint_value(value: Any):
    torch = _require_torch()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(k): _json_fingerprint_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_fingerprint_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _payload_fingerprint(saved: Mapping[str, Any]) -> str:
    fields = {
        key: saved.get(key)
        for key in (
            "backbone",
            "feature_dim",
            "classes",
            "feature_center",
            "head_type",
            "prototypes",
            "prototype_class_indices",
            "prototypes_per_class",
            "selection_k",
            "adaptive_k",
            "k_by_class",
            "prototype_norm_power",
            "encoder_weights",
            "encoder_sha256",
            "preprocessing",
            "event_aggregation",
            "threshold",
            "rejection",
        )
    }
    encoded = json.dumps(_json_fingerprint_value(fields), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_checkpoint(
    saved: Mapping[str, Any],
    *,
    path: Path | None = None,
    fingerprint: str | None = None,
) -> DinoV2Checkpoint:
    if not isinstance(saved, Mapping):
        raise CheckpointValidationError("checkpoint payload must be a mapping")
    backbone = str(saved.get("backbone", ""))
    feature_dim = saved.get("feature_dim")
    if backbone != DINO_BACKBONE or feature_dim != DINO_FEATURE_DIM:
        raise CheckpointValidationError(
            "Expected facebook/dinov2-base with 768-dimensional features; feature spaces are not interchangeable"
        )
    classes = _classes(saved)
    head_type = str(saved.get("head_type", "")).strip().lower()
    if head_type != DINO_MULTI_PROTOTYPE_HEAD:
        raise CheckpointValidationError("DINOv2 requires head_type=multi_prototype")
    preprocessing, aggregation, encoder_weights, encoder_sha256 = _metadata(saved)
    center, prototypes, indices, counts, adaptive_k, selection_k, k_by_class, norm_power = _prototype_contract(saved, classes)
    rejection = _rejection(saved, len(classes))
    threshold = _finite_float(saved.get("threshold"), "threshold")
    if not math.isclose(threshold, rejection.cosine_threshold, rel_tol=0.0, abs_tol=1e-12):
        raise CheckpointValidationError("threshold must match rejection cosine_threshold")

    return DinoV2Checkpoint(
        path=path,
        backbone=backbone,
        feature_dim=DINO_FEATURE_DIM,
        classes=classes,
        feature_center=center,
        prototypes=prototypes,
        prototype_class_indices=indices,
        prototypes_per_class=counts,
        threshold=threshold,
        encoder_weights=encoder_weights,
        encoder_sha256=encoder_sha256,
        preprocessing=preprocessing,
        event_aggregation=aggregation,
        rejection=rejection,
        fingerprint=fingerprint or _payload_fingerprint(saved),
        head_type=head_type,
        selection_k=selection_k,
        adaptive_k=adaptive_k,
        k_by_class=k_by_class,
        adaptive_k_selection=saved.get("adaptive_k_selection"),
        adaptive_k_folds=saved.get("adaptive_k_folds"),
        prototype_norm_power=norm_power,
        data_fingerprint=str(saved.get("data_fingerprint", "")),
        threshold_calibration=str(saved.get("threshold_calibration", "")),
        max_known_frr=(None if saved.get("max_known_frr") is None else _finite_float(saved.get("max_known_frr"), "max_known_frr", nonnegative=True)),
        selected_by=str(saved.get("selected_by", "")),
        optimization_seed=(saved.get("optimization_seed") if isinstance(saved.get("optimization_seed"), int) and not isinstance(saved.get("optimization_seed"), bool) else None),
        rejection_score=str(saved.get("rejection_score", "")),
        calibration=saved.get("calibration"),
        scale_training=saved.get("scale_training"),
    )


def load_checkpoint(path: str | Path) -> DinoV2Checkpoint:
    torch = _require_torch()
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"DINOv2 classifier checkpoint not found: {resolved}")
    try:
        saved = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        saved = torch.load(resolved, map_location="cpu")
    with resolved.open("rb") as handle:
        fingerprint = hashlib.file_digest(handle, "sha256").hexdigest()
    return validate_checkpoint(saved, path=resolved, fingerprint=fingerprint)
