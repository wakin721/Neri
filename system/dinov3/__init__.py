"""Native DINOv3 classification and open-world species registration."""
from .checkpoint import (
    DINO_BACKBONE,
    DINO_FEATURE_DIM,
    CheckpointValidationError,
    DinoV3Checkpoint,
    load_checkpoint,
    validate_checkpoint,
)
from .component import (
    DINO_BACKBONE_SHA256,
    DINO_CLASSIFIER_SHA256,
    DINO_SOURCE_COMMIT,
    DinoV3ComponentPaths,
    dinov3_component_paths,
    dinov3_component_status,
    remove_dinov3_component,
)

__all__ = [
    "DINO_BACKBONE",
    "DINO_FEATURE_DIM",
    "DINO_BACKBONE_SHA256",
    "DINO_CLASSIFIER_SHA256",
    "DINO_SOURCE_COMMIT",
    "CheckpointValidationError",
    "DinoV3Checkpoint",
    "DinoV3ComponentPaths",
    "load_checkpoint",
    "validate_checkpoint",
    "dinov3_component_paths",
    "dinov3_component_status",
    "remove_dinov3_component",
]
