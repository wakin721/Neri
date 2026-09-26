from __future__ import annotations

import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from system.dinov2.checkpoint import load_checkpoint
from system.dinov2.runtime import load_dinov2_model


def _unit(index: int) -> np.ndarray:
    vector = np.zeros(768, dtype=np.float32)
    vector[index] = 1.0
    return vector


def _memory_model(tmp_path, *, weight=0.0, threshold=1.0, encoder_sha="a" * 64):
    classes = np.array(["A", "B"])
    features = np.stack([_unit(0), _unit(0), _unit(1), _unit(2), _unit(3)])
    labels = np.array(["A", "A", "A", "B", "B"])
    cameras = np.array(["a1", "a1", "a2", "b1", "b2"])
    metadata = {
        "version": 1,
        "config": {
            "neighbors": 3,
            "centroid_weight": weight,
            "camera_pooling": True,
            "margin_weight": 1.0,
        },
        "threshold": threshold,
        "calibration": {"images": 100, "target_frr": 0.04},
        "provenance": {
            "encoder_sha256": encoder_sha,
            "preprocessing": "letterbox224_imagenet",
            "feature_dim": 768,
            "gradient_updates": 0,
        },
    }
    model = tmp_path / "memory_no_centroid.npz"
    np.savez_compressed(
        model,
        metadata=json.dumps(metadata),
        center=np.zeros(768, dtype=np.float32),
        features=features,
        labels=labels,
        cameras=cameras,
        classes=classes,
        centroids=np.stack([_unit(0), _unit(2)]),
    )
    manifest = tmp_path / "classifier.neri.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "backend": "dinov2",
                "checkpoint": model.name,
                "architecture": "dinov2_vitb14",
                "feature_dim": 768,
                "encoder_sha256": encoder_sha,
                "preprocessing": "letterbox224_imagenet",
                "event_aggregation": "mean_l2_normalized_crop_embeddings",
            }
        ),
        encoding="utf-8",
    )
    return model, manifest


def test_memory_no_centroid_scores_camera_maxima_and_margin(tmp_path):
    model, manifest = _memory_model(tmp_path)

    class Encoder:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint

    class Store:
        def __init__(self, fingerprint):
            self.model_fingerprint = fingerprint

    checkpoint = load_checkpoint(model)
    runtime = load_dinov2_model(
        manifest,
        registry=Store(checkpoint.fingerprint),
        feedback=Store(checkpoint.fingerprint),
        encoder_factory=Encoder,
    )
    prediction = runtime.classifier.classify_features(_unit(0)[None, :])[0]
    assert checkpoint.head_type == "memory_no_centroid"
    assert prediction.species == "A"
    assert prediction.accepted is True
    assert prediction.known_score == pytest.approx(1.0)
    assert prediction.threshold == pytest.approx(1.0)
    assert prediction.candidates[0]["name"] == "A"


def test_memory_threshold_tie_accepts_and_higher_threshold_rejects(tmp_path):
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier

    model, _ = _memory_model(tmp_path, threshold=1.0)
    classifier = MemoryDinoV2Classifier(load_checkpoint(model))
    assert classifier.classify_features(_unit(0)[None, :])[0].accepted is True

    model, _ = _memory_model(tmp_path, threshold=1.01)
    classifier = MemoryDinoV2Classifier(load_checkpoint(model))
    result = classifier.classify_features(_unit(0)[None, :])[0]
    assert result.accepted is False
    assert result.species == "Unknown"
    assert result.best_known_species == "A"


def test_formal_memory_evidence_enters_classification_without_retraining(tmp_path):
    from system.dinov2.memory_bank import MemoryBank, MemoryExample
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier

    class Registry:
        def memory_bank(self):
            return MemoryBank(
                formal=(MemoryExample("C", _unit(4), "new-camera", "overlay", registry_id=7),),
            )

    model, _ = _memory_model(tmp_path)
    classifier = MemoryDinoV2Classifier(load_checkpoint(model), registry=Registry())
    result = classifier.classify_features(_unit(4)[None, :])[0]
    assert result.species == "C"
    assert result.accepted is True
    assert result.source == "overlay"
    assert result.registry_id == 7


def test_provisional_memory_evidence_is_assistive_only(tmp_path):
    from system.dinov2.memory_bank import MemoryBank, MemoryExample
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier

    class Registry:
        def memory_bank(self):
            return MemoryBank(
                formal=(),
                provisional=(MemoryExample("C", _unit(4), "new-camera", "overlay", registry_id=7),),
            )

    model, _ = _memory_model(tmp_path)
    classifier = MemoryDinoV2Classifier(load_checkpoint(model), registry=Registry())
    result = classifier.classify_features(_unit(4)[None, :])[0]
    assert result.species == "C"
    assert result.accepted is False
    assert result.assistive_match is True
    assert result.best_known_species in {"A", "B"}


def test_registry_memory_evidence_uses_event_cameras_after_registration(tmp_path):
    from system.dinov2.registry import SpeciesRegistry

    registry = SpeciesRegistry(tmp_path / "registry.sqlite3", model_fingerprint="9" * 64)
    try:
        start = datetime(2026, 9, 23)
        entry = registry.record_unknown(
            _unit(4), camera_id="cam-1", captured_at=start, source_path="first.jpg"
        )
        for index in range(1, 4):
            registry.record_observation(
                entry.id,
                _unit(4),
                camera_id=f"cam-{index + 1}",
                captured_at=start + timedelta(hours=index),
                source_path=f"{index}.jpg",
            )
        registry.set_identity(entry.id, common_name="C")
        assert registry.memory_bank().formal == ()
        assert registry.memory_bank().provisional == ()
        registry.register(entry.id)
        bank = registry.memory_bank()
        assert len(bank.provisional) == 4
        assert {sample.camera_id for sample in bank.provisional} == {
            "cam-1", "cam-2", "cam-3", "cam-4"
        }
        assert all(sample.registry_id == entry.id for sample in bank.provisional)
    finally:
        registry.close()


def test_feedback_memory_bank_keeps_activated_evidence_snapshot(tmp_path, monkeypatch):
    from system.dinov2.feedback import HumanFeedbackStore

    model, _ = _memory_model(tmp_path)
    checkpoint = load_checkpoint(model)
    store = HumanFeedbackStore(
        tmp_path / "feedback.sqlite3",
        model_fingerprint=checkpoint.fingerprint,
        checkpoint_classes=checkpoint.classes,
        rejection=checkpoint.rejection,
        memory_checkpoint=checkpoint,
    )
    events = [SimpleNamespace(embedding=_unit(4), camera_id="review-camera")]
    monkeypatch.setattr(store, "_evidence_observations", lambda *args, **kwargs: [])
    monkeypatch.setattr(store, "_group_events", lambda _: events)
    try:
        store._activate_generation(
            "A", status="confirmed", prototypes=_unit(4)[None, :],
            quality_passed=True, positive_coverage=1.0, false_accept_rate=0.0,
        )
        events.append(SimpleNamespace(embedding=_unit(5), camera_id="unreviewed"))
        bank = store.memory_bank()
        assert len(bank.formal) == 1
        assert bank.formal[0].species == "A"
        assert bank.formal[0].camera_id == "review-camera"
        assert bank.formal[0].source == "feedback"
    finally:
        store.close()


def test_feedback_memory_quality_uses_winner_and_margin(tmp_path):
    from system.dinov2.feedback import HumanFeedbackStore

    model, _ = _memory_model(tmp_path, threshold=0.5)
    checkpoint = load_checkpoint(model)
    store = HumanFeedbackStore.__new__(HumanFeedbackStore)
    store.memory_checkpoint = checkpoint
    store.threshold = checkpoint.threshold
    store.prototype_norm_power = 1.0
    positives = [SimpleNamespace(embedding=_unit(4), camera_id="new") for _ in range(4)]
    negatives = [SimpleNamespace(embedding=_unit(0), camera_id="old")]

    _, coverage, false_accept_rate, quality = store._build_candidate(
        positives, negatives, np.zeros(768, dtype=np.float32),
        status="confirmed", species="A",
    )

    assert coverage == 1.0
    assert false_accept_rate == 1.0
    assert quality is False


def test_default_component_replaces_distributed_prototype_with_memory(tmp_path):
    from system.dinov2.component import (
        _replace_default_classifier,
        dinov2_component_status,
    )
    from tests.test_dinov2_component import _sha, _write_component

    root = tmp_path / "DINOv2"
    root.mkdir()
    _write_component(root)
    asset_dir = tmp_path / "asset"
    asset_dir.mkdir()
    asset, _ = _memory_model(
        asset_dir, encoder_sha=_sha(root / "model" / "model.safetensors")
    )

    _replace_default_classifier(root, memory_asset=asset)

    status = dinov2_component_status(root=root)
    assert status["healthy"] is True
    assert status["classifier_head_type"] == "memory_no_centroid"
    assert status["classifier_filename"] == "memory_no_centroid.npz"
    assert not (root / "classifier.pt").exists()


def test_registration_duplicate_guard_uses_memory_classifier(tmp_path, monkeypatch):
    from system.dinov2 import api

    model, _ = _memory_model(tmp_path)
    checkpoint = load_checkpoint(model)
    monkeypatch.setattr(
        api,
        "_registry_service",
        lambda: SimpleNamespace(load_checkpoint_for_model=lambda _: checkpoint),
    )

    class Registry:
        def register(self, entry_id, *, formal_matcher):
            assert formal_matcher(_unit(0)) == "A"
            return SimpleNamespace(as_dict=lambda: {"id": entry_id})

    assert api._register_with_duplicate_guard(Registry(), 7, str(model)) == {"id": 7}


def test_memory_explanation_identifies_score_and_projects_current_sample(tmp_path):
    from system.dinov2.memory_classifier import MemoryDinoV2Classifier

    model, _ = _memory_model(tmp_path)
    explanation = MemoryDinoV2Classifier(load_checkpoint(model)).explain_feature(_unit(0))
    assert explanation["rejection"]["mode"] == "memory_no_centroid"
    assert explanation["nearest_species"][0]["cosine_score"] == pytest.approx(0.5)
    assert explanation["projection"]["species"] == ["A", "B"]
    assert any(point["kind"] == "current" for point in explanation["projection"]["points"])


def test_memory_catalog_groups_exemplars_by_species(tmp_path):
    from system.dinov2.api import DinoV2RegistryEntryResponse, build_registry_catalog

    model, _ = _memory_model(tmp_path)
    registry = SimpleNamespace(list=lambda: [])
    catalog = build_registry_catalog(load_checkpoint(model), registry)

    assert len(catalog) == 2
    assert catalog[0]["prototype_count"] == 3
    assert len(catalog[0]["clusters"]) == 1
    assert catalog[0]["clusters"][0]["sample_count"] == 3
    assert catalog[0]["clusters"][0]["camera_count"] == 2
    response = DinoV2RegistryEntryResponse(**catalog[0])
    assert response.clusters[0].prototype_index is None


@pytest.mark.parametrize("weight", [0.1, 0.5, 1.0, -0.1])
def test_memory_loader_rejects_nonzero_centroid_weight(tmp_path, weight):
    model, _ = _memory_model(tmp_path, weight=weight)
    with pytest.raises(ValueError, match="centroid_weight"):
        load_checkpoint(model)
