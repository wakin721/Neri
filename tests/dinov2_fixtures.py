from __future__ import annotations

import copy
import torch


def make_dinov2_payload(*, with_scales: bool = True):
    dim = 768
    classes = ["A", "B"]
    center = torch.zeros(dim, dtype=torch.float32)
    prototypes = torch.zeros((4, dim), dtype=torch.float32)
    prototypes[0, 0] = 1.0
    prototypes[1, 1] = 1.0
    prototypes[2, 2] = 1.0
    prototypes[3, 3] = 1.0
    rejection = {
        "mode": "multi_dual_margin",
        "cosine_threshold": 0.31,
        "adjusted_distance_score_threshold": -0.80,
        "margin_weight": 2.0,
    }
    if with_scales:
        rejection["class_distance_scales"] = [1.0, 1.25]
    return {
        "backbone": "facebook/dinov2-base",
        "feature_dim": dim,
        "preprocessing": "letterbox224_imagenet",
        "event_aggregation": "mean_l2_normalized_crop_embeddings",
        "encoder_weights": "models/dinov2-base/model.safetensors",
        "encoder_sha256": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
        "data_fingerprint": "1" * 64,
        "classes": classes,
        "feature_center": center,
        "head_type": "multi_prototype",
        "threshold": 0.31,
        "prototypes": prototypes,
        "prototype_class_indices": torch.tensor([0, 1, 1, 1], dtype=torch.int64),
        "prototypes_per_class": [1, 3],
        "selection_k": 0,
        "adaptive_k": True,
        "k_by_class": {"A": 1, "B": 3},
        "adaptive_k_selection": [],
        "adaptive_k_folds": 3,
        "prototype_norm_power": 1.0,
        "rejection": rejection,
        "threshold_calibration": "dual_cosine_l2_exact",
        "max_known_frr": 0.05,
        "selected_by": "validation only",
        "optimization_seed": 20260916,
        "rejection_score": "cosine_AND_distance_plus_different_class_margin",
        "calibration": {"method": "matched_frr_class_scaled_margin"},
        "scale_training": {"strength": 0.0},
    }


def clone_payload(payload):
    return copy.deepcopy(payload)
