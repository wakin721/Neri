"""DINOv2 classification and open-world species registration."""
from .checkpoint import (
    DINO_ARCHITECTURE_ID,
    DINO_BACKBONE,
    DINO_FEATURE_DIM,
    DINO_MODEL_REVISION,
    CheckpointValidationError,
    DinoV2Checkpoint,
    DinoV2Rejection,
    load_checkpoint,
    validate_checkpoint,
)

__all__ = [
    "DINO_ARCHITECTURE_ID",
    "DINO_BACKBONE",
    "DINO_FEATURE_DIM",
    "DINO_MODEL_REVISION",
    "CheckpointValidationError",
    "DinoV2Checkpoint",
    "DinoV2Rejection",
    "load_checkpoint",
    "validate_checkpoint",
]
