"""Validated, frozen training-free Memory head for DINOv2 inference."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .checkpoint import (
    DINO_BACKBONE,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
    DinoV2Rejection,
)


@dataclass(frozen=True)
class MemoryCheckpoint:
    path: Path
    classes: tuple[str, ...]
    feature_center: Any
    features: np.ndarray
    labels: np.ndarray
    cameras: np.ndarray
    neighbors: int
    margin_weight: float
    threshold: float
    encoder_sha256: str
    fingerprint: str
    calibration: dict[str, Any]
    backbone: str = DINO_BACKBONE
    feature_dim: int = DINO_FEATURE_DIM
    preprocessing: str = DINO_PREPROCESSING
    event_aggregation: str = DINO_EVENT_AGGREGATION
    encoder_weights: str = "model/model.safetensors"
    head_type: str = "memory_no_centroid"
    prototype_norm_power: float = 1.0

    @property
    def prototypes(self) -> np.ndarray:
        """Expose exemplar count through the existing component status field."""
        return self.features

    @property
    def rejection(self) -> DinoV2Rejection:
        # Feedback persistence still needs a threshold-bearing rejection record.
        return DinoV2Rejection(
            mode="multi_dual_margin",
            cosine_threshold=self.threshold,
            adjusted_distance_score_threshold=self.threshold,
            margin_weight=self.margin_weight,
        )


def _unit_rows(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DINO_FEATURE_DIM or not len(array):
        raise ValueError(f"{name} must have shape (N, 768)")
    if not np.isfinite(array).all() or not np.allclose(
        np.linalg.norm(array, axis=1), 1.0, atol=1e-4
    ):
        raise ValueError(f"{name} must contain finite unit vectors")
    array.setflags(write=False)
    return array


def load_memory_checkpoint(path: str | Path) -> MemoryCheckpoint:
    import torch

    resolved = Path(path).expanduser().resolve()
    with np.load(resolved, allow_pickle=False) as archive:
        required = {"metadata", "center", "features", "labels", "cameras", "classes", "centroids"}
        if set(archive.files) != required:
            raise ValueError("Memory checkpoint fields are incomplete or unexpected")
        metadata = json.loads(str(archive["metadata"].item()))
        center = np.asarray(archive["center"], dtype=np.float32).copy()
        features = _unit_rows(archive["features"].copy(), "Memory features")
        labels = archive["labels"].astype(str).copy()
        cameras = archive["cameras"].astype(str).copy()
        classes = archive["classes"].astype(str).copy()
        _unit_rows(archive["centroids"], "Memory centroids")
    if not isinstance(metadata, dict) or metadata.get("version") != 1:
        raise ValueError("Unsupported Memory checkpoint version")
    config = metadata.get("config")
    if not isinstance(config, dict) or set(config) != {
        "neighbors", "centroid_weight", "camera_pooling", "margin_weight"
    }:
        raise ValueError("Invalid Memory config")
    if config["centroid_weight"] != 0.0:
        raise ValueError("Memory centroid_weight must be zero")
    if config["neighbors"] not in (1, 3, 5) or isinstance(config["neighbors"], bool):
        raise ValueError("Memory neighbors must be 1, 3 or 5")
    if config["camera_pooling"] is not True:
        raise ValueError("Memory camera_pooling must be true")
    margin_weight = float(config["margin_weight"])
    if not np.isfinite(margin_weight) or margin_weight < 0:
        raise ValueError("Memory margin_weight must be finite and nonnegative")
    threshold = float(metadata.get("threshold"))
    if not np.isfinite(threshold):
        raise ValueError("Memory threshold must be finite")
    provenance = metadata.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("preprocessing") != DINO_PREPROCESSING:
        raise ValueError("Memory preprocessing does not match DINOv2")
    if provenance.get("feature_dim") != DINO_FEATURE_DIM or provenance.get("gradient_updates") != 0:
        raise ValueError("Memory feature or training provenance is invalid")
    encoder_sha256 = provenance.get("encoder_sha256")
    if not isinstance(encoder_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", encoder_sha256):
        raise ValueError("Memory encoder SHA-256 is invalid")
    if center.shape != (DINO_FEATURE_DIM,) or not np.isfinite(center).all():
        raise ValueError("Memory feature center is invalid")
    if labels.shape != (len(features),) or cameras.shape != labels.shape:
        raise ValueError("Memory exemplar labels or cameras are misaligned")
    if len(classes) < 2 or not np.array_equal(np.unique(labels), classes):
        raise ValueError("Memory classes do not match exemplar labels")
    if any(not name.strip() for name in classes) or any(not camera.strip() for camera in cameras):
        raise ValueError("Memory classes and cameras must be nonempty")
    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict) or not calibration.get("images"):
        raise ValueError("Memory threshold must have calibration evidence")
    with resolved.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    labels.setflags(write=False)
    cameras.setflags(write=False)
    return MemoryCheckpoint(
        path=resolved,
        classes=tuple(classes),
        feature_center=torch.from_numpy(center),
        features=features,
        labels=labels,
        cameras=cameras,
        neighbors=int(config["neighbors"]),
        margin_weight=margin_weight,
        threshold=threshold,
        encoder_sha256=encoder_sha256.lower(),
        fingerprint=fingerprint,
        calibration=calibration,
    )
