from __future__ import annotations

import torch

from system.dinov3.checkpoint import (
    DINO_BACKBONE,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
)

ENCODER_SHA256 = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
ENCODER_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def make_multi_prototype_payload(
    *,
    classes: tuple[str, ...] = ("A", "B"),
    threshold: float = 0.31,
) -> dict[str, object]:
    feature_center = torch.zeros(DINO_FEATURE_DIM, dtype=torch.float32)
    prototypes: list[torch.Tensor] = []
    prototype_class_indices: list[int] = []
    for class_index in range(len(classes)):
        for prototype_index in range(3):
            vector = torch.zeros(DINO_FEATURE_DIM, dtype=torch.float32)
            vector[class_index * 3 + prototype_index] = 1.0
            prototypes.append(vector)
            prototype_class_indices.append(class_index)
    return {
        "schema_version": 1,
        "backbone": DINO_BACKBONE,
        "feature_dim": DINO_FEATURE_DIM,
        "classes": list(classes),
        "feature_center": feature_center,
        "head_type": "multi_prototype",
        "selection_k": 3,
        "prototypes": torch.stack(prototypes),
        "prototype_class_indices": torch.tensor(
            prototype_class_indices,
            dtype=torch.int64,
        ),
        "prototypes_per_class": [3] * len(classes),
        "threshold": threshold,
        "decision": "squared_euclidean_to_nearest_prototype",
        "rejection_score": "cosine_similarity_to_winning_prototype",
        "encoder_weights": (
            "dinov3-vitb16/"
            "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        ),
        "encoder_sha256": ENCODER_SHA256,
        "encoder_code_commit": ENCODER_COMMIT,
        "preprocessing": DINO_PREPROCESSING,
        "event_aggregation": DINO_EVENT_AGGREGATION,
    }
