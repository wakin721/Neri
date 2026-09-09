from __future__ import annotations

import numpy as np
import pytest
import torch

from system.dinov3.checkpoint import validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def _unit_feature(index: int = 0) -> np.ndarray:
    feature = np.zeros(768, dtype=np.float32)
    feature[index] = 1.0
    return feature


def test_classifier_uses_centered_squared_euclidean_to_choose_prototype():
    payload = make_multi_prototype_payload(threshold=-1.0)
    payload["feature_center"][0] = 0.25
    payload["prototypes"][0] = torch.zeros(768, dtype=torch.float32)
    payload["prototypes"][0, 0] = 0.75
    checkpoint = validate_checkpoint(payload)

    prediction = DinoV3Classifier(checkpoint).classify_features(
        _unit_feature(0)[None, :]
    )[0]

    assert prediction.species == "A"
    assert prediction.accepted is True
    assert prediction.nearest_prototype_index == 0
    assert prediction.squared_distance == pytest.approx(0.0, abs=1e-7)
    assert prediction.known_score == pytest.approx(1.0, abs=1e-7)


def test_classifier_candidates_collapse_three_prototypes_into_each_species():
    checkpoint = validate_checkpoint(
        make_multi_prototype_payload(threshold=-1.0)
    )

    prediction = DinoV3Classifier(checkpoint).classify_features(
        _unit_feature(0)[None, :]
    )[0]

    assert [candidate["name"] for candidate in prediction.candidates] == ["A", "B"]
    assert len({candidate["name"] for candidate in prediction.candidates}) == 2
    assert prediction.candidates[0]["nearest_prototype_index"] == 0
    assert prediction.candidates[0]["squared_distance"] == pytest.approx(0.0)
    assert prediction.candidates[0]["cosine_score"] == pytest.approx(1.0)


def test_classifier_rejects_using_cosine_to_winning_prototype():
    payload = make_multi_prototype_payload(threshold=0.8)
    payload["prototypes"][0] = torch.zeros(768, dtype=torch.float32)
    payload["prototypes"][0, 0] = 1.0
    payload["prototypes"][0, 1] = 1.0
    checkpoint = validate_checkpoint(payload)

    prediction = DinoV3Classifier(checkpoint).classify_features(
        _unit_feature(0)[None, :]
    )[0]

    assert prediction.nearest_prototype_index == 0
    assert prediction.known_score == pytest.approx(2 ** -0.5, rel=1e-6)
    assert prediction.accepted is False
    assert prediction.species == "Unknown"
    assert prediction.best_known_species == "A"
