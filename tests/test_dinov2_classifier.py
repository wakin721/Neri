from __future__ import annotations

import numpy as np
import pytest
import torch

from system.dinov2.checkpoint import validate_checkpoint
from system.dinov2.classifier import DinoV2Classifier, aggregate_event_embeddings
from system.dinov2.prototype_bank import PrototypeBank, PrototypeRecord
from tests.dinov2_fixtures import make_dinov2_payload


def _unit(index: int) -> np.ndarray:
    value = np.zeros(768, dtype=np.float32)
    value[index] = 1.0
    return value


def _geometry_payload():
    payload = make_dinov2_payload()
    prototypes = torch.zeros((3, 768), dtype=torch.float32)
    prototypes[0, 0] = 1.0
    prototypes[1, 0] = 0.8
    prototypes[1, 1] = 0.6
    prototypes[2, 1] = 1.0
    payload["prototypes"] = prototypes
    payload["prototype_class_indices"] = torch.tensor([0, 0, 1], dtype=torch.int64)
    payload["prototypes_per_class"] = [2, 1]
    payload["k_by_class"] = {"A": 2, "B": 1}
    payload["threshold"] = 0.5
    payload["rejection"] = {
        "mode": "multi_dual_margin",
        "cosine_threshold": 0.5,
        "adjusted_distance_score_threshold": 1.5,
        "margin_weight": 2.0,
        "class_distance_scales": [2.0, 1.0],
    }
    return payload


def test_cl2n_nearest_prototype_and_margin_ignore_same_class_prototype():
    classifier = DinoV2Classifier(validate_checkpoint(_geometry_payload()))
    prediction = classifier.classify_features(_unit(0)[None, :])[0]
    assert prediction.best_known_species == "A"
    assert prediction.species == "A"
    assert prediction.accepted is True
    assert prediction.nearest_prototype_index == 0
    assert prediction.squared_distance == pytest.approx(0.0, abs=1e-7)
    assert prediction.class_margin == pytest.approx(2.0, abs=1e-6)
    assert prediction.adjusted_distance_score == pytest.approx(2.0, abs=1e-6)
    assert prediction.known_score == pytest.approx(1.0, abs=1e-7)


def test_score_gate_can_reject_even_when_cosine_passes():
    payload = _geometry_payload()
    payload["rejection"]["adjusted_distance_score_threshold"] = 2.1
    classifier = DinoV2Classifier(validate_checkpoint(payload))
    prediction = classifier.classify_features(_unit(0)[None, :])[0]
    assert prediction.known_score == pytest.approx(1.0)
    assert prediction.adjusted_distance_score == pytest.approx(2.0)
    assert prediction.accepted is False
    assert prediction.species == "Unknown"
    assert prediction.best_known_species == "A"


def test_centering_is_followed_by_l2_normalization():
    payload = _geometry_payload()
    payload["feature_center"][0] = 0.25
    classifier = DinoV2Classifier(validate_checkpoint(payload))
    prediction = classifier.classify_features(_unit(0)[None, :])[0]
    assert prediction.squared_distance == pytest.approx(0.0, abs=1e-7)
    assert prediction.known_score == pytest.approx(1.0, abs=1e-7)


def test_zero_vector_after_centering_is_rejected():
    payload = _geometry_payload()
    payload["feature_center"] = torch.from_numpy(_unit(0))
    classifier = DinoV2Classifier(validate_checkpoint(payload))
    with pytest.raises(ValueError, match="zero"):
        classifier.classify_features(_unit(0)[None, :])


def test_input_features_must_be_l2_normalized():
    classifier = DinoV2Classifier(validate_checkpoint(_geometry_payload()))
    bad = _unit(0) * 2.0
    with pytest.raises(ValueError, match="L2-normalized"):
        classifier.classify_features(bad[None, :])


def test_event_aggregation_means_then_l2_normalizes():
    crops = np.stack([_unit(0), _unit(1)])
    event = aggregate_event_embeddings(crops)
    assert event[0] == pytest.approx(2 ** -0.5)
    assert event[1] == pytest.approx(2 ** -0.5)
    assert np.linalg.norm(event) == pytest.approx(1.0)


class _Provider:
    def __init__(self, bank):
        self.bank = bank

    def prototype_bank(self, center):
        return self.bank


def test_provisional_novel_species_is_assistive_not_formally_accepted():
    payload = _geometry_payload()
    payload["rejection"]["adjusted_distance_score_threshold"] = 0.5
    provisional = PrototypeRecord(
        species="C",
        embedding=_unit(2),
        source="overlay",
        registry_id=7,
        registration_status="provisional",
    )
    classifier = DinoV2Classifier(
        validate_checkpoint(payload),
        registry=_Provider(PrototypeBank(formal=(), provisional=(provisional,))),
    )
    prediction = classifier.classify_features(_unit(2)[None, :])[0]
    assert prediction.species == "C"
    assert prediction.accepted is False
    assert prediction.assistive_match is True
    assert prediction.source == "overlay"
    assert prediction.registry_id == 7
    assert prediction.best_known_species in {"A", "B"}


def test_explanation_exposes_margin_rejection_and_projection():
    classifier = DinoV2Classifier(validate_checkpoint(_geometry_payload()))
    explanation = classifier.explain_feature(_unit(0))
    assert explanation["rejection"]["mode"] == "multi_dual_margin"
    assert explanation["class_margin"] == pytest.approx(2.0)
    assert explanation["adjusted_distance_score"] == pytest.approx(2.0)
    assert explanation["projection"]["method"] == "nearest_two_species_axis"
    assert any(point["kind"] == "current" for point in explanation["projection"]["points"])


class _FakeEncoder:
    def encode(self, crops, *, array_color="rgb"):
        return np.stack([_unit(index) for index in range(len(crops))])


def test_classify_crops_uses_attached_encoder():
    classifier = DinoV2Classifier(validate_checkpoint(_geometry_payload()), encoder=_FakeEncoder())
    predictions = classifier.classify_crops([object()])
    assert len(predictions) == 1
    assert predictions[0].best_known_species == "A"


def test_classify_event_aggregates_encoder_features():
    payload = _geometry_payload()
    payload["rejection"]["adjusted_distance_score_threshold"] = -10.0
    classifier = DinoV2Classifier(validate_checkpoint(payload), encoder=_FakeEncoder())
    prediction = classifier.classify_event([object(), object()])
    assert prediction.best_known_species in {"A", "B"}
