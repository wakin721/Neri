from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from system.dinov3.checkpoint import load_checkpoint, validate_checkpoint
from system.dinov3.classifier import DinoV3Classifier
from system.dinov3.multi_dual_classifier import MultiDualDinoV3Classifier
from system.dinov3.prototype_bank import PrototypeBank, PrototypeRecord
from system.dinov3.rejection import MultiDualRejectionConfig, parse_multi_dual_rejection
from system.dinov3.runtime import DinoV3ManifestError, load_dinov3_model, resolve_dinov3_manifest
from tests.dinov3_multi_prototype_fixtures import make_multi_prototype_payload


def _unit_feature(index: int = 0) -> np.ndarray:
    feature = np.zeros(768, dtype=np.float32)
    feature[index] = 1.0
    return feature


class _FakeEncoder:
    def __init__(self, checkpoint, **kwargs) -> None:
        self.checkpoint = checkpoint


class _StaticBankProvider:
    def __init__(self, bank: PrototypeBank) -> None:
        self.bank = bank

    def prototype_bank(self, _center) -> PrototypeBank:
        return self.bank


def _write_runtime_fixture(tmp_path, *, rejection=None, checkpoint_threshold=0.31):
    payload = make_multi_prototype_payload(threshold=checkpoint_threshold)
    checkpoint_path = tmp_path / "multi_prototype.pt"
    torch.save(payload, checkpoint_path)
    manifest = {
        "schema_version": 1,
        "backend": "dinov3",
        "display_name": "test",
        "checkpoint": checkpoint_path.name,
        "architecture": "dinov3_vitb16",
        "feature_dim": 768,
        "encoder_sha256": payload["encoder_sha256"],
        "preprocessing": payload["preprocessing"],
        "event_aggregation": payload["event_aggregation"],
    }
    if rejection is not None:
        manifest["rejection"] = rejection
    manifest_path = tmp_path / "multi_prototype.neri.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, checkpoint_path


def test_dual_gate_requires_cosine_and_squared_distance():
    config = MultiDualRejectionConfig(
        cosine_threshold=0.30,
        squared_distance_threshold=0.40,
    )

    assert config.accepts(cosine_score=0.31, squared_distance=0.39) is True
    assert config.accepts(cosine_score=0.29, squared_distance=0.39) is False
    assert config.accepts(cosine_score=0.31, squared_distance=0.41) is False


def test_rejection_manifest_requires_both_calibrated_thresholds():
    with pytest.raises(ValueError, match="cosine_threshold"):
        parse_multi_dual_rejection(
            {"mode": "multi_dual", "squared_distance_threshold": 0.4}
        )
    with pytest.raises(ValueError, match="squared_distance_threshold"):
        parse_multi_dual_rejection(
            {"mode": "multi_dual", "cosine_threshold": 0.3}
        )


def test_rejection_manifest_rejects_invalid_values():
    invalid = (
        {"mode": "multi_dual", "cosine_threshold": 2.0, "squared_distance_threshold": 0.4},
        {"mode": "multi_dual", "cosine_threshold": 0.3, "squared_distance_threshold": -1.0},
        {"mode": "multi_dual", "cosine_threshold": math.nan, "squared_distance_threshold": 0.4},
    )
    for value in invalid:
        with pytest.raises(ValueError):
            parse_multi_dual_rejection(value)


def test_multi_dual_rejects_when_cosine_passes_but_l2_fails():
    payload = make_multi_prototype_payload(threshold=-1.0)
    payload["prototypes"][0] = torch.zeros(768, dtype=torch.float32)
    payload["prototypes"][0, 0] = 0.5
    checkpoint = validate_checkpoint(payload)
    classifier = MultiDualDinoV3Classifier(
        checkpoint,
        rejection=MultiDualRejectionConfig(
            cosine_threshold=0.8,
            squared_distance_threshold=0.10,
        ),
    )

    prediction = classifier.classify_features(_unit_feature(0)[None, :])[0]

    assert prediction.nearest_prototype_index == 0
    assert prediction.known_score == pytest.approx(1.0)
    assert prediction.squared_distance == pytest.approx(0.25)
    assert prediction.accepted is False
    assert prediction.species == "Unknown"
    assert prediction.best_known_species == "A"


def test_multi_dual_accepts_when_both_gates_pass():
    payload = make_multi_prototype_payload(threshold=-1.0)
    payload["prototypes"][0] = torch.zeros(768, dtype=torch.float32)
    payload["prototypes"][0, 0] = 0.5
    checkpoint = validate_checkpoint(payload)
    classifier = MultiDualDinoV3Classifier(
        checkpoint,
        rejection=MultiDualRejectionConfig(
            cosine_threshold=0.8,
            squared_distance_threshold=0.30,
        ),
    )

    prediction = classifier.classify_features(_unit_feature(0)[None, :])[0]

    assert prediction.species == "A"
    assert prediction.accepted is True
    assert prediction.threshold == pytest.approx(0.8)
    assert prediction.squared_distance == pytest.approx(0.25)


def test_multi_dual_preserves_provisional_assistive_match_semantics():
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=-1.0))
    provisional = PrototypeRecord(
        species="new-species",
        embedding=_unit_feature(20),
        source="overlay",
        registry_id=7,
        registration_status="provisional",
    )
    provider = _StaticBankProvider(PrototypeBank(formal=(), provisional=(provisional,)))
    classifier = MultiDualDinoV3Classifier(
        checkpoint,
        rejection=MultiDualRejectionConfig(
            cosine_threshold=0.8,
            squared_distance_threshold=0.10,
        ),
        registry=provider,
    )

    prediction = classifier.classify_features(_unit_feature(20)[None, :])[0]

    assert prediction.species == "new-species"
    assert prediction.accepted is False
    assert prediction.assistive_match is True
    assert prediction.source == "overlay"
    assert prediction.registry_id == 7
    assert prediction.registration_status == "provisional"


def test_multi_dual_explanation_exposes_both_thresholds():
    checkpoint = validate_checkpoint(make_multi_prototype_payload(threshold=-1.0))
    classifier = MultiDualDinoV3Classifier(
        checkpoint,
        rejection=MultiDualRejectionConfig(
            cosine_threshold=0.3,
            squared_distance_threshold=0.4,
        ),
    )

    explanation = classifier.explain_feature(_unit_feature(0))

    assert explanation["rejection"] == {
        "mode": "multi_dual",
        "cosine_threshold": 0.3,
        "squared_distance_threshold": 0.4,
    }


def test_runtime_uses_multi_dual_only_when_manifest_enables_it(tmp_path):
    manifest, _checkpoint = _write_runtime_fixture(
        tmp_path,
        rejection={
            "mode": "multi_dual",
            "cosine_threshold": 0.42,
            "squared_distance_threshold": 0.55,
        },
    )
    runtime = load_dinov3_model(
        manifest,
        state_root=tmp_path / "state",
        encoder_factory=_FakeEncoder,
    )
    try:
        assert isinstance(runtime.classifier, MultiDualDinoV3Classifier)
        assert runtime.feedback.threshold == pytest.approx(0.42)
        assert runtime.classifier.rejection_metadata["squared_distance_threshold"] == pytest.approx(0.55)
    finally:
        runtime.close()


def test_runtime_keeps_legacy_cosine_classifier_without_rejection_block(tmp_path):
    manifest, _checkpoint = _write_runtime_fixture(tmp_path, checkpoint_threshold=0.27)
    runtime = load_dinov3_model(
        manifest,
        state_root=tmp_path / "state",
        encoder_factory=_FakeEncoder,
    )
    try:
        assert type(runtime.classifier) is DinoV3Classifier
        assert runtime.feedback.threshold == pytest.approx(0.27)
    finally:
        runtime.close()


def test_runtime_rejects_incomplete_multi_dual_manifest(tmp_path):
    manifest, _checkpoint = _write_runtime_fixture(
        tmp_path,
        rejection={"mode": "multi_dual", "cosine_threshold": 0.3},
    )

    with pytest.raises(DinoV3ManifestError, match="squared_distance_threshold"):
        load_dinov3_model(
            manifest,
            state_root=tmp_path / "state",
            encoder_factory=_FakeEncoder,
        )


def test_runtime_still_resolves_manifest_when_checkpoint_path_is_selected(tmp_path):
    manifest, checkpoint_path = _write_runtime_fixture(tmp_path)

    assert resolve_dinov3_manifest(checkpoint_path) == manifest.resolve()
    assert load_checkpoint(checkpoint_path).feature_dim == 768
