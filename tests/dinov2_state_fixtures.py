from __future__ import annotations

import torch

from system.dinov2.checkpoint import (
    DINO_BACKBONE,
    DINO_EVENT_AGGREGATION,
    DINO_FEATURE_DIM,
    DINO_PREPROCESSING,
    DINO_REJECTION_MODE,
    DinoV2Rejection,
)
from system.dinov2.feedback import HumanFeedbackStore as _HumanFeedbackStore

ENCODER_SHA256 = "0" * 64


def rejection_for_threshold(threshold: float) -> DinoV2Rejection:
    return DinoV2Rejection(
        mode=DINO_REJECTION_MODE,
        cosine_threshold=float(threshold),
        adjusted_distance_score_threshold=-1_000_000.0,
        margin_weight=0.0,
        class_distance_scales=None,
    )


class HumanFeedbackStore(_HumanFeedbackStore):
    """Test adapter preserving the old threshold= fixture ergonomics."""

    def __init__(
        self,
        path,
        *,
        model_fingerprint,
        checkpoint_classes,
        threshold=None,
        rejection=None,
        prototype_norm_power=0.0,
    ):
        if rejection is None:
            rejection = rejection_for_threshold(0.31 if threshold is None else threshold)
        super().__init__(
            path,
            model_fingerprint=model_fingerprint,
            checkpoint_classes=checkpoint_classes,
            rejection=rejection,
            prototype_norm_power=prototype_norm_power,
        )


def make_multi_prototype_payload(
    *,
    classes: tuple[str, ...] = ("A", "B"),
    threshold: float = 0.31,
    selection_k: int = 3,
) -> dict[str, object]:
    feature_center = torch.zeros(DINO_FEATURE_DIM, dtype=torch.float32)
    prototypes: list[torch.Tensor] = []
    prototype_class_indices: list[int] = []
    for class_index in range(len(classes)):
        for prototype_index in range(selection_k):
            vector = torch.zeros(DINO_FEATURE_DIM, dtype=torch.float32)
            vector[class_index * selection_k + prototype_index] = 1.0
            prototypes.append(vector)
            prototype_class_indices.append(class_index)
    return {
        "backbone": DINO_BACKBONE,
        "feature_dim": DINO_FEATURE_DIM,
        "classes": list(classes),
        "feature_center": feature_center,
        "head_type": "multi_prototype",
        "selection_k": selection_k,
        "adaptive_k": False,
        "prototypes": torch.stack(prototypes),
        "prototype_class_indices": torch.tensor(
            prototype_class_indices,
            dtype=torch.int64,
        ),
        "prototypes_per_class": [selection_k] * len(classes),
        "threshold": float(threshold),
        "rejection": {
            "mode": DINO_REJECTION_MODE,
            "cosine_threshold": float(threshold),
            "adjusted_distance_score_threshold": -1_000_000.0,
            "margin_weight": 0.0,
        },
        "rejection_score": "cosine_AND_distance_plus_different_class_margin",
        "prototype_norm_power": 0.0,
        "encoder_weights": "dinov2-base/model.safetensors",
        "encoder_sha256": ENCODER_SHA256,
        "preprocessing": DINO_PREPROCESSING,
        "event_aggregation": DINO_EVENT_AGGREGATION,
    }
