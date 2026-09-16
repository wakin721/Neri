from __future__ import annotations

import numpy as np
import pytest
import torch

from system.dinov2.checkpoint import DinoV2Rejection, validate_checkpoint
from system.dinov2.classifier import DinoV2Classifier
from system.dinov2.rejection import adjusted_distance_score, accepts_margin_rejection
from tests.dinov2_fixtures import make_dinov2_payload


def _rejection():
    return DinoV2Rejection(
        mode="multi_dual_margin",
        cosine_threshold=0.8,
        adjusted_distance_score_threshold=0.5,
        margin_weight=2.0,
        class_distance_scales=(2.0, 1.0),
    )


def test_adjusted_score_matches_benchmark_operation_order():
    score = adjusted_distance_score(
        winner_squared_distance=0.4,
        class_margin=1.2,
        margin_weight=2.0,
        class_scale=2.0,
    )
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("cosine", "score", "expected"),
    [
        (0.81, 0.51, True),
        (0.79, 0.51, False),
        (0.81, 0.49, False),
        (0.79, 0.49, False),
    ],
)
def test_acceptance_requires_both_gates(cosine, score, expected):
    assert accepts_margin_rejection(cosine, score, _rejection()) is expected


def test_batch_geometry_matches_benchmark_matrix_multiplication():
    rng = np.random.default_rng(1)
    payload = make_dinov2_payload(with_scales=False)
    prototypes = rng.normal(size=(20, 768)).astype(np.float32)
    payload["prototypes"] = torch.from_numpy(prototypes.copy())
    payload["prototype_class_indices"] = torch.tensor([0] * 10 + [1] * 10, dtype=torch.int64)
    payload["prototypes_per_class"] = [10, 10]
    payload["k_by_class"] = {"A": 10, "B": 10}
    payload["feature_center"] = torch.zeros(768, dtype=torch.float32)
    classifier = DinoV2Classifier(validate_checkpoint(payload))

    features = rng.normal(size=(32, 768)).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    predictions = classifier.classify_features(features)

    x = features - payload["feature_center"].numpy()[None, :]
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    dot = x @ prototypes.T
    distances = np.maximum(
        np.square(x).sum(axis=1, keepdims=True)
        + np.square(prototypes).sum(axis=1)[None, :]
        - 2 * dot,
        0.0,
    )
    winner = distances.argmin(axis=1)

    for row, prediction in enumerate(predictions):
        assert prediction.nearest_prototype_index == int(winner[row])
        assert prediction.squared_distance == pytest.approx(
            float(distances[row, winner[row]]), abs=1e-7
        )
